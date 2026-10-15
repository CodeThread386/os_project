"""Privileged tools: run allow-listed admin programs as root, install and remove software.

Every action here needs a live approval from the person at the keyboard (a
broker invariant that no policy can switch off, and that ``--yes`` does not
cover). Root commands run through sudo outside the sandbox, because they
exist to change the system. The password, when sudo needs one, is typed
into AgentLayer's hidden prompt and written straight into sudo's standard
input through a pipe: the model, the audit log and the disk never see it.
"""

from __future__ import annotations

import os
import pwd
import re
import shutil

from agentlayer.broker import Requirement
from agentlayer.executor import CommandSyntaxError, RunResult, run_pipeline, split_pipeline
from agentlayer.policy import PACKAGE_MANAGERS, ExecLimits
from agentlayer.tools.base import Tool, ToolArgError, ToolContext, ToolOutput, param, register

SAFE_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin"
SUDO: str | None = None  # resolved lazily; tests point it at a fake
ROOT_LIMITS = ExecLimits(cpu_seconds=3600, memory_mb=8192, file_size_mb=16384, open_files=4096)
_PACKAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+._:@=-]{0,127}")


def _sudo() -> str:
    path = SUDO or shutil.which("sudo", path=SAFE_PATH)
    if not path:
        raise ToolArgError("sudo is not installed")
    return path


def _env() -> dict[str, str]:
    return {"PATH": SAFE_PATH, "LANG": os.environ.get("LANG", "C.UTF-8"), "TERM": "dumb",
            "HOME": os.path.expanduser("~"), "DEBIAN_FRONTEND": "noninteractive"}


def _user() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return str(os.getuid())


def _run(argv: list[str], timeout: float, stdin: bytes | None = None, cwd: str = "/", stream=None) -> RunResult:
    return run_pipeline([(argv[0], argv)], cwd=cwd, env=_env(), timeout=timeout, max_output=200_000,
                        limits=ROOT_LIMITS, stdin_data=stdin, on_output=stream)


def run_privileged(command: list[str], ctx: ToolContext, timeout: float) -> RunResult | ToolOutput:
    """Run ``command`` (absolute program path first) as root via sudo."""
    sudo = _sudo()
    if _run([sudo, "-n", "true"], timeout=15).exit_code == 0:
        return _run([sudo, "-n", *command], timeout, stream=ctx.stream)  # sudo's credential cache is still valid
    if ctx.ask_secret is None:
        return ToolOutput.fail("sudo needs your password, but there is no terminal to ask for it; nothing was run")
    password = ctx.ask_secret(f"[sudo] password for {_user()}")
    if not password:
        return ToolOutput.fail("no password was entered; nothing was run")
    secret = (password + "\n").encode()
    del password  # keep the plain-text password around no longer than needed
    result = _run([sudo, "-S", "-p", "", *command], timeout, stdin=secret, stream=ctx.stream)
    del secret
    if result.exit_code != 0 and ("incorrect password" in result.stderr or "try again" in result.stderr.lower()):
        return ToolOutput.fail("sudo did not accept the password; nothing was run")
    return result


def _tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "…" + text[-limit:]


def _report(result: RunResult, what: str, ctx: ToolContext) -> ToolOutput:
    budget = max(1500, ctx.policy.max_tool_output - 800)
    data = {"exit_code": result.exit_code, "status": result.status, "output_tail": _tail(result.stdout, budget)}
    if result.stderr.strip():
        data["stderr_tail"] = _tail(result.stderr, budget // 3)
    ok = result.exit_code == 0
    display = _tail(result.stdout + ("\n" + result.stderr if result.stderr else ""), 3000).rstrip()
    return ToolOutput(ok, data, f"{what}: {result.status} ({result.duration:.0f}s)", display or None)


# --------------------------------------------------------------- run_as_root


def _root_argv(args: dict) -> list[str]:
    try:
        stages = split_pipeline(args["command"])
    except CommandSyntaxError as exc:
        raise ToolArgError(str(exc)) from exc
    if len(stages) != 1:
        raise ToolArgError("pipes are not allowed for root commands; run one program")
    return [word for word, _ in stages[0]]


def _root_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    argv = _root_argv(args)
    return [Requirement("sys.admin", op="run", program=argv[0], argv=argv)]


def _run_as_root(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    argv = decision.approved().argv
    program = shutil.which(argv[0], path=SAFE_PATH)
    if program is None:
        return ToolOutput.fail(f"'{argv[0]}' is not installed")
    grant = decision.grant("sys.admin")
    result = run_privileged([program, *argv[1:]], ctx, grant.timeout)
    if isinstance(result, ToolOutput):
        return result
    return _report(result, " ".join(argv)[:60], ctx)


register(Tool(
    name="run_as_root",
    description="Run one allow-listed admin program as root via sudo (e.g. 'systemctl restart nginx'). "
                "The user approves each one and types the password if sudo asks.",
    rights=("sys.admin",),
    params={"command": param("string", "The command, without 'sudo'")},
    required=("command",),
    requires=_root_requires,
    run=_run_as_root,
    preview=lambda a, c: f"$ sudo {a['command']}\nRuns as ROOT, outside the sandbox.",
    destructive=True,
))


# ------------------------------------------------------------------ packages

_COMMANDS = {
    ("apt", "install"): ["apt-get", "install", "-y"],
    ("apt", "remove"): ["apt-get", "remove", "-y"],
    ("snap", "install"): ["snap", "install"],
    ("snap", "remove"): ["snap", "remove"],
    ("flatpak", "install"): ["flatpak", "install", "-y", "--noninteractive", "flathub"],
    ("flatpak", "remove"): ["flatpak", "uninstall", "-y", "--noninteractive"],
    ("pipx", "install"): ["pipx", "install"],
    ("pipx", "remove"): ["pipx", "uninstall"],
}
_SEARCH = {
    "apt": ["apt-cache", "search", "--names-only"],
    "snap": ["snap", "find"],
    "flatpak": ["flatpak", "search"],
}


def _names(args: dict) -> list[str]:
    names = [n for n in re.split(r"[\s,]+", args.get("names") or "") if n]
    if not names:
        raise ToolArgError("give one or more package names")
    bad = [n for n in names if not _PACKAGE.fullmatch(n)]
    if bad:
        raise ToolArgError(f"invalid package name(s): {', '.join(bad)}")
    if args["manager"] == "pipx" and len(names) > 1 and args["action"] != "search":
        raise ToolArgError("pipx handles one package per call")
    return names


def _packages_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    names = _names(args)
    if args["action"] == "search":
        return [Requirement("sys.read", op="package-search")]
    return [Requirement("sys.admin", op=args["action"], program=args["manager"], packages=names)]


def _packages(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    names, manager, action = _names(args), args["manager"], args["action"]
    if action == "search":
        base = _SEARCH.get(manager)
        if base is None:
            return ToolOutput.fail(f"{manager} has no search command; try apt, snap or flatpak")
        program = shutil.which(base[0], path=SAFE_PATH)
        if program is None:
            return ToolOutput.fail(f"{base[0]} is not installed")
        result = _run([program, *base[1:], *names], timeout=60)
        lines = result.stdout.splitlines()
        text = "\n".join(lines[:40]) + (f"\n… {len(lines) - 40} more" if len(lines) > 40 else "")
        return ToolOutput(result.exit_code == 0, {"results": text or "(no matches)", "matches": len(lines)},
                          f"{len(lines)} {manager} package(s) match", text or None)

    base = _COMMANDS[(manager, action)]
    program = shutil.which(base[0], path=SAFE_PATH)
    if program is None:
        return ToolOutput.fail(f"{base[0]} is not installed on this system")
    command = [program, *base[1:], *names]
    grant = decision.grant("sys.admin")
    if manager == "pipx":  # installs into the user's home: no root needed
        result = _run(command, grant.timeout, cwd=os.path.expanduser("~"), stream=ctx.stream)
    else:
        result = run_privileged(command, ctx, grant.timeout)
        if isinstance(result, ToolOutput):
            return result
    return _report(result, f"{manager} {action} {' '.join(names)}", ctx)


register(Tool(
    name="packages",
    description="Search for, install or remove software with apt, snap, flatpak or pipx. "
                "Install/remove need the user's approval (and sudo password).",
    rights=("sys.admin",),
    params={
        "action": param("string", "search, install or remove", enum=["search", "install", "remove"]),
        "names": param("string", "Package name(s) or search words, space-separated"),
        "manager": param("string", "Package manager", enum=list(PACKAGE_MANAGERS), default="apt"),
    },
    required=("action", "names"),
    requires=_packages_requires,
    run=_packages,
    preview=lambda a, c: (f"{a['action']} with {a['manager']}: {a['names']}\n"
                          + ("Runs as ROOT via sudo; this can take a few minutes." if a["manager"] != "pipx"
                             else "Installs into your home directory with pipx.")),
    destructive=True,
))
