"""Container-style isolation for commands, built on bubblewrap (bwrap).

bwrap uses unprivileged Linux namespaces: a new mount namespace exposes
only the read-only system directories plus the workspace (the home
directory, SSH keys and other files simply do not exist inside), a new PID
namespace hides every other process, and a new network namespace has no
interfaces. If bwrap is missing or user namespaces are disabled, commands
fall back to resource limits only, and the broker then refuses programs
that can run arbitrary code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

SYSTEM_DIRS = ("/usr", "/etc", "/opt")
ROOT_LINKS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")


class SandboxError(Exception):
    pass


@dataclass
class Sandbox:
    mode: str  # "bwrap" | "none"
    reason: str
    bwrap: str | None = None

    @property
    def active(self) -> bool:
        return self.mode == "bwrap"

    def wrap(self, program: str, argv: list[str], *, cwd: str, workspace: str, network: bool,
             masks: list[str] = (), mounts: list[tuple[str, bool]] = ()) -> tuple[str, list[str]]:
        """Return the (path, argv) to execute, wrapped in bwrap when active.

        ``mounts`` are extra (directory, writable) pairs besides the workspace.
        ``masks`` are paths inside the mounted directories that the policy
        denies; inside the sandbox they are covered (files by /dev/null,
        directories by an empty tmpfs), so no program can read them.
        """
        if not self.active:
            return program, [program, *argv[1:]]
        assert self.bwrap
        args = bwrap_args(cwd=cwd, workspace=workspace, network=network)
        chdir = args.index("--chdir")
        extra: list[str] = []
        for directory, writable in mounts:
            extra += ["--bind" if writable else "--ro-bind", directory, directory]
        args[chdir:chdir] = extra  # mount before --chdir so cwd can be inside them
        for path in masks:
            if os.path.isdir(path) and not os.path.islink(path):
                args += ["--tmpfs", path]
            elif not os.path.islink(path):
                args += ["--ro-bind", "/dev/null", path]
        return self.bwrap, [self.bwrap, *args, "--", program, *argv[1:]]


def bwrap_args(*, cwd: str, workspace: str, network: bool) -> list[str]:
    args = ["--die-with-parent", "--new-session", "--unshare-all", "--unshare-user", "--disable-userns"]
    if network:
        args.append("--share-net")
    for d in SYSTEM_DIRS:
        if os.path.isdir(d):
            args += ["--ro-bind", d, d]
    for link in ROOT_LINKS:
        if os.path.islink(link):
            args += ["--symlink", os.readlink(link), link]
        elif os.path.isdir(link):
            args += ["--ro-bind", link, link]
    args += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run"]
    if network and os.path.isdir("/run/systemd/resolve"):
        args += ["--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve"]
    args += ["--bind", workspace, workspace, "--chdir", cwd, "--hostname", "agentlayer-sandbox"]
    return args


def detect(mode: str = "auto", workspace: str | None = None) -> Sandbox:
    """Pick the sandbox for this host. ``mode`` is auto, bwrap or none."""
    if mode == "none":
        return Sandbox("none", "disabled by configuration")
    path = shutil.which("bwrap")
    if path is None:
        if mode == "bwrap":
            raise SandboxError("the policy requires bubblewrap, but 'bwrap' is not installed (apt install bubblewrap)")
        return Sandbox("none", "bubblewrap is not installed")
    probe_dir = workspace or "/"
    try:
        result = subprocess.run(
            [path, *bwrap_args(cwd=probe_dir, workspace=probe_dir, network=False), "--", "/bin/true"],
            capture_output=True, timeout=15,
        )
        ok, detail = result.returncode == 0, result.stderr.decode(errors="replace").strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        ok, detail = False, str(exc)
    if ok:
        return Sandbox("bwrap", "bubblewrap namespaces (mount, pid, net, ipc, uts, user)", path)
    if mode == "bwrap":
        raise SandboxError(f"bubblewrap is installed but cannot create namespaces: {detail}")
    return Sandbox("none", f"bubblewrap cannot create namespaces here ({detail or 'unknown error'})")
