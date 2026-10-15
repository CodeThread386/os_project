"""Security policy: the access matrix enforced by the capability broker.

A policy is a TOML file. It names a sandboxed workspace, path deny rules
that override everything else, and a list of grants. Each grant gives the
agent one *right* (``fs.read``, ``exec``, ``proc.signal``, ...) over a set of
*objects* (path globs, program names, a process scope, host globs). Read
together, the grants form the agent's row of Lampson's access matrix, which
is the same thing as the agent's capability list.
"""

from __future__ import annotations

import fnmatch
import os
import re
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from agentlayer.config import STATE_DIR

RIGHTS: dict[str, str] = {
    "fs.read": "read files, list directories, inspect metadata",
    "fs.write": "create and modify files and directories",
    "fs.delete": "delete files and directories",
    "proc.read": "list and inspect processes through /proc",
    "proc.signal": "send signals to processes",
    "proc.priority": "change a process's nice value",
    "exec": "run allow-listed programs",
    "net.read": "read-only HTTP GET lookups",
    "sys.read": "read system information",
    "gui.launch": "launch desktop apps, open files/URLs in them, show notifications",
    "gui.windows": "list, focus, minimise, maximise and close windows",
    "gui.screen": "take screenshots",
    "gui.clipboard": "read and write the clipboard",
    "sys.admin": "run allow-listed admin programs as root, install and remove software",
    "memory": "keep long-term notes across sessions",
    "sys.logs": "read service status logs from the systemd journal",
    "task.schedule": "schedule unattended tasks with systemd timers",
    "cap.request": "ask the user for temporary, expiring capabilities",
}

NETWORK_MODES = ("never", "confirm", "always")
PACKAGE_MANAGERS = ("apt", "snap", "flatpak", "pipx")

SIGNAL_NAMES = ("SIGTERM", "SIGINT", "SIGHUP", "SIGKILL", "SIGSTOP", "SIGCONT", "SIGUSR1", "SIGUSR2", "SIGQUIT")

_COMMON_KEYS = {"right", "confirm", "note"}
_RIGHT_KEYS: dict[str, set[str]] = {
    "fs.read": {"paths"},
    "fs.write": {"paths", "confirm_overwrite"},
    "fs.delete": {"paths"},
    "proc.read": {"scope"},
    "proc.signal": {"scope", "signals", "protect"},
    "proc.priority": {"scope", "protect"},
    "exec": {
        "programs", "sandboxed_only", "workspace_programs", "confirm_programs", "deny_args", "sandbox",
        "network", "extra_dirs", "timeout", "max_timeout", "max_output", "limits",
    },
    "net.read": {"hosts", "confirm_unlisted", "allow_private", "max_bytes", "timeout"},
    "sys.read": set(),
    "gui.launch": {"apps", "confirm_unlisted"},
    "gui.windows": {"confirm_close"},
    "gui.screen": set(),
    "gui.clipboard": {"confirm_read"},
    "sys.admin": {"programs", "managers", "deny_args", "timeout"},
    "memory": {"max_items"},
    "sys.logs": set(),
    "task.schedule": {"max_tasks"},
    "cap.request": {"max_minutes"},
}

BUNDLED = {
    "phase1": "phase1-readonly",
    "phase2": "phase2-write",
    "phase3": "phase3-process",
    "phase4": "phase4-full",
    "phase5": "phase5-desktop",
}


class PolicyError(Exception):
    pass


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Compile a path glob: ``*`` stays inside one path segment, ``**`` crosses them.

    A trailing ``/**`` also matches the directory itself, so ``/ws/**`` covers
    ``/ws`` and everything below it.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("/**", i) and i + 3 == len(pattern):
            out.append("(?:/.*)?")
            i += 3
        elif pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out), re.DOTALL)


def canonical_pattern(pattern: str, workspace: Path) -> str:
    """Expand ``~``/``{workspace}`` and resolve symlinks in the literal prefix.

    Paths are compared after ``realpath()``, so a pattern such as
    ``/etc/os-release`` (a symlink on many distributions) must be resolved the
    same way or it would never match.
    """
    expanded = os.path.expanduser(pattern.replace("{workspace}", str(workspace)))
    if expanded.startswith("**"):
        return expanded
    if not expanded.startswith("/"):
        raise PolicyError(f"path pattern must be absolute or start with '**': {pattern!r}")
    parts = expanded.split("/")
    split = next((i for i, part in enumerate(parts) if "*" in part or "?" in part), len(parts))
    prefix = os.path.realpath("/".join(parts[:split]) or "/")
    rest = "/".join(parts[split:])
    if not rest:
        return prefix
    return ("" if prefix == "/" else prefix) + "/" + rest


@dataclass
class PathRule:
    raw: str  # as written in the policy file
    pattern: str  # canonical form
    regex: re.Pattern[str]

    def matches(self, path: str) -> bool:
        return self.regex.fullmatch(path) is not None


@dataclass
class ExecLimits:
    cpu_seconds: int = 30
    memory_mb: int = 2048
    file_size_mb: int = 64
    open_files: int = 256


@dataclass
class Grant:
    right: str
    confirm: bool = False
    note: str = ""
    # fs.*
    paths: list[PathRule] = field(default_factory=list)
    confirm_overwrite: bool = False
    # proc.*
    scope: str = "own"
    signals: list[str] = field(default_factory=lambda: ["SIGTERM", "SIGINT", "SIGHUP", "SIGKILL"])
    protect: list[str] = field(default_factory=list)
    # exec
    programs: list[str] = field(default_factory=list)
    sandboxed_only: list[str] = field(default_factory=list)
    workspace_programs: bool = False  # run executables built inside the workspace (sandbox only)
    confirm_programs: list[str] = field(default_factory=list)
    deny_args: dict[str, list[str]] = field(default_factory=dict)
    sandbox: str = "auto"
    network: str = "never"  # never | confirm | always: may a command ask for internet access?
    extra_dirs: list[str] = field(default_factory=list)  # besides the workspace, mounted into the sandbox
    timeout: int = 30
    max_timeout: int = 120
    max_output: int = 16_000
    limits: ExecLimits = field(default_factory=ExecLimits)
    # net.read
    hosts: list[str] = field(default_factory=list)
    confirm_unlisted: bool = True
    allow_private: bool = False
    max_bytes: int = 2_000_000
    # gui.*
    apps: list[str] = field(default_factory=list)
    confirm_close: bool = True
    confirm_read: bool = True
    # sys.admin
    managers: list[str] = field(default_factory=list)
    # memory
    max_items: int = 200
    # task.schedule
    max_tasks: int = 20
    # cap.request
    max_minutes: int = 120

    def match_path(self, path: str) -> PathRule | None:
        return next((rule for rule in self.paths if rule.matches(path)), None)

    def host_listed(self, host: str) -> bool:
        host = host.lower().rstrip(".")
        return any(fnmatch.fnmatchcase(host, pat.lower()) for pat in self.hosts)

    def app_listed(self, app_id: str) -> bool:
        app_id = app_id.lower().removesuffix(".desktop")
        return any(fnmatch.fnmatchcase(app_id, pat.lower()) for pat in self.apps)


@dataclass
class Policy:
    name: str
    description: str
    source: Path
    workspace: Path
    deny: list[PathRule]
    grants: list[Grant]
    audit_path: Path
    max_steps: int = 12
    max_read_bytes: int = 64 * 1024
    max_tool_output: int = 6000

    def grants_for(self, right: str) -> list[Grant]:
        return [g for g in self.grants if g.right == right]

    def grant(self, right: str) -> Grant | None:
        found = self.grants_for(right)
        return found[0] if found else None

    def has(self, right: str) -> bool:
        return any(g.right == right for g in self.grants)

    def denied_by(self, path: str) -> PathRule | None:
        return next((rule for rule in self.deny if rule.matches(path)), None)

    def display_path(self, path: str) -> str:
        """Shorten a path for humans and prompts (workspace and home prefixes)."""
        ws = str(self.workspace)
        if path == ws or path.startswith(ws + "/"):
            return "{workspace}" + path[len(ws):]
        home = os.path.expanduser("~")
        if path == home or path.startswith(home + "/"):
            return "~" + path[len(home):]
        return path


def _str_list(value, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise PolicyError(f"{where} must be a list of strings")
    return list(value)


def _parse_grant(raw: dict, workspace: Path, index: int) -> Grant:
    right = raw.get("right")
    where = f"grant #{index + 1}"
    if right not in RIGHTS:
        raise PolicyError(f"{where}: unknown right {right!r} (known: {', '.join(RIGHTS)})")
    where = f"grant #{index + 1} ({right})"
    unknown = set(raw) - _COMMON_KEYS - _RIGHT_KEYS[right]
    if unknown:
        raise PolicyError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")

    g = Grant(right=right, confirm=bool(raw.get("confirm", False)), note=str(raw.get("note", "")))
    if right.startswith("fs."):
        if "paths" not in raw:
            raise PolicyError(f"{where}: 'paths' is required")
        for p in _str_list(raw["paths"], f"{where}.paths"):
            canon = canonical_pattern(p, workspace)
            g.paths.append(PathRule(p, canon, glob_to_regex(canon)))
        g.confirm_overwrite = bool(raw.get("confirm_overwrite", False))
    if right.startswith("proc."):
        g.scope = raw.get("scope", "own")
        if g.scope not in ("own", "all"):
            raise PolicyError(f"{where}: scope must be 'own' or 'all'")
        g.protect = _str_list(raw.get("protect", []), f"{where}.protect")
        if "signals" in raw:
            g.signals = [s.upper() if s.upper().startswith("SIG") else "SIG" + s.upper()
                         for s in _str_list(raw["signals"], f"{where}.signals")]
            bad = [s for s in g.signals if s not in SIGNAL_NAMES]
            if bad:
                raise PolicyError(f"{where}: unsupported signal(s) {', '.join(bad)}")
    if right == "exec":
        g.programs = _str_list(raw.get("programs", []), f"{where}.programs")
        g.sandboxed_only = _str_list(raw.get("sandboxed_only", []), f"{where}.sandboxed_only")
        g.workspace_programs = bool(raw.get("workspace_programs", False))
        g.confirm_programs = _str_list(raw.get("confirm_programs", []), f"{where}.confirm_programs")
        deny_args = raw.get("deny_args", {})
        if not isinstance(deny_args, dict):
            raise PolicyError(f"{where}.deny_args must be a table of program = [patterns]")
        g.deny_args = {prog: _str_list(v, f"{where}.deny_args.{prog}") for prog, v in deny_args.items()}
        g.sandbox = raw.get("sandbox", "auto")
        if g.sandbox not in ("auto", "bwrap", "none"):
            raise PolicyError(f"{where}: sandbox must be 'auto', 'bwrap' or 'none'")
        network = raw.get("network", "never")
        g.network = {False: "never", True: "always"}.get(network, network) if isinstance(network, bool) else network
        if g.network not in NETWORK_MODES:
            raise PolicyError(f"{where}: network must be one of {', '.join(NETWORK_MODES)} (or true/false)")
        g.extra_dirs = [canonical_pattern(d, workspace) for d in _str_list(raw.get("extra_dirs", []), f"{where}.extra_dirs")]
        if any("*" in d or "?" in d for d in g.extra_dirs):
            raise PolicyError(f"{where}.extra_dirs must be plain directories, not globs")
        g.timeout = int(raw.get("timeout", 30))
        g.max_timeout = int(raw.get("max_timeout", max(120, g.timeout)))
        g.max_output = int(raw.get("max_output", 16_000))
        limits = raw.get("limits", {})
        unknown_limits = set(limits) - set(ExecLimits.__dataclass_fields__)
        if unknown_limits:
            raise PolicyError(f"{where}.limits: unknown key(s) {', '.join(sorted(unknown_limits))}")
        g.limits = ExecLimits(**{k: int(v) for k, v in limits.items()})
    if right == "net.read":
        g.hosts = _str_list(raw.get("hosts", []), f"{where}.hosts")
        g.confirm_unlisted = bool(raw.get("confirm_unlisted", True))
        g.allow_private = bool(raw.get("allow_private", False))
        g.max_bytes = int(raw.get("max_bytes", 2_000_000))
        g.timeout = int(raw.get("timeout", 15))
    if right == "gui.launch":
        g.apps = _str_list(raw.get("apps", []), f"{where}.apps")
        g.confirm_unlisted = bool(raw.get("confirm_unlisted", True))
    if right == "gui.windows":
        g.confirm_close = bool(raw.get("confirm_close", True))
    if right == "gui.clipboard":
        g.confirm_read = bool(raw.get("confirm_read", True))
    if right == "sys.admin":
        g.programs = _str_list(raw.get("programs", []), f"{where}.programs")
        g.managers = _str_list(raw.get("managers", []), f"{where}.managers")
        bad = [m for m in g.managers if m not in PACKAGE_MANAGERS]
        if bad:
            raise PolicyError(f"{where}: unknown package manager(s) {', '.join(bad)} (known: {', '.join(PACKAGE_MANAGERS)})")
        deny_args = raw.get("deny_args", {})
        g.deny_args = {prog: _str_list(v, f"{where}.deny_args.{prog}") for prog, v in deny_args.items()}
        g.timeout = int(raw.get("timeout", 900))
        g.confirm = True  # invariant: privileged actions always need a person's approval
    if right == "memory":
        g.max_items = int(raw.get("max_items", 200))
    if right == "task.schedule":
        g.max_tasks = int(raw.get("max_tasks", 20))
        g.confirm = True  # invariant: creating unattended automation always needs the user's approval
    if right == "cap.request":
        g.max_minutes = max(1, min(240, int(raw.get("max_minutes", 120))))
    return g


def resolve_policy_path(name_or_path: str) -> Path:
    """Accept a file path, a bundled policy name (``phase4-full``) or alias (``phase4``)."""
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    stem = BUNDLED.get(name_or_path, name_or_path).removesuffix(".toml")
    bundled = resources.files("agentlayer") / "policies" / f"{stem}.toml"
    if bundled.is_file():
        return Path(str(bundled)).resolve()
    names = ", ".join(sorted(BUNDLED.values()))
    raise PolicyError(f"policy {name_or_path!r} not found (bundled policies: {names})")


def bundled_policy_names() -> list[str]:
    return [BUNDLED[k] for k in sorted(BUNDLED)]


def load_policy(name_or_path: str, workspace: str | None = None) -> Policy:
    source = resolve_policy_path(name_or_path)
    try:
        data = tomllib.loads(source.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise PolicyError(f"{source}: {exc}") from exc

    known_sections = {"policy", "workspace", "agent", "audit", "deny", "grant"}
    unknown = set(data) - known_sections
    if unknown:
        raise PolicyError(f"{source.name}: unknown section(s) {', '.join(sorted(unknown))}")

    meta = data.get("policy", {})
    ws_raw = workspace or data.get("workspace", {}).get("root", "~/agentlayer-workspace")
    ws_path = Path(os.path.expanduser(ws_raw))
    if not ws_path.is_absolute():
        raise PolicyError("workspace root must be an absolute path (or start with ~)")
    ws_path.mkdir(parents=True, exist_ok=True)
    ws_path = Path(os.path.realpath(ws_path))
    if ws_path == Path("/") or ws_path == Path(os.path.realpath(os.path.expanduser("~"))):
        raise PolicyError("the workspace must be a dedicated directory, not / or your home directory")

    deny: list[PathRule] = []
    for p in _str_list(data.get("deny", {}).get("paths", []), "deny.paths"):
        canon = canonical_pattern(p, ws_path)
        deny.append(PathRule(p, canon, glob_to_regex(canon)))

    grants = [_parse_grant(raw, ws_path, i) for i, raw in enumerate(data.get("grant", []))]
    agent = data.get("agent", {})
    audit_raw = data.get("audit", {}).get("path")
    audit_path = Path(os.path.expanduser(audit_raw)) if audit_raw else STATE_DIR / "audit.jsonl"

    return Policy(
        name=meta.get("name", source.stem),
        description=meta.get("description", ""),
        source=source,
        workspace=ws_path,
        deny=deny,
        grants=grants,
        audit_path=audit_path,
        max_steps=int(agent.get("max_steps", 12)),
        max_read_bytes=int(agent.get("max_read_bytes", 64 * 1024)),
        max_tool_output=int(agent.get("max_tool_output", 6000)),
    )
