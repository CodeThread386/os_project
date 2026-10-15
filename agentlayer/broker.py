"""The capability broker: the reference monitor between the LLM and the OS.

Tools never decide whether they may run. Each tool translates its arguments
into a list of :class:`Requirement` objects (a right plus the concrete
object it will touch: a canonical path, a PID, a program, a host). The
broker checks every requirement against the policy and returns one
:class:`Decision`: allow, deny, or allow only after the user confirms.

Policy is data (the TOML file); the checks below are the mechanism. A few
invariants are part of the mechanism and hold under every policy, for
example that the broker's own policy file and audit log can never be
modified by the agent it is supervising.
"""

from __future__ import annotations

import fnmatch
import ipaddress
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from agentlayer import procfs
from agentlayer.config import STATE_DIR
from agentlayer.policy import Grant, PathRule, Policy, PolicyError, canonical_pattern, glob_to_regex

if TYPE_CHECKING:
    from agentlayer.tools.base import Tool, ToolContext


class Verdict(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


# Files under /proc that are never readable: they expose other programs'
# secrets (environment variables hold API keys) or raw memory.
NEVER_READABLE = [
    "/proc/*/environ", "/proc/*/task/*/environ", "/proc/*/mem", "/proc/*/task/*/mem",
    "/proc/kcore", "/proc/kmem", "/proc/*/auxv",
]
_SUBTREE_SCAN_LIMIT = 20_000
GRANTABLE = ("fs.read", "fs.write", "fs.delete", "net.read", "exec", "gui.launch")
_SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"


@dataclass
class Requirement:
    """One right the action needs over one concrete object."""

    right: str
    path: str | None = None  # canonical path the operation touches
    lexical: str | None = None  # the path as requested, normalised
    op: str = ""  # read | list | write | append | mkdir | move-from | move-to | copy-to | delete
    recursive: bool = False  # deleting a directory tree
    overwrite: bool = False  # replaces existing data
    pid: int | None = None
    signal: str | None = None
    program: str | None = None
    argv: list[str] | None = None
    cwd: str | None = None
    url: str | None = None
    network: bool = False  # a command that asks for internet access
    target: str | None = None  # gui: app id, window title, clipboard/screen operation
    packages: list[str] | None = None  # sys.admin package operations
    minutes: int | None = None  # cap.request: how long the capability should last
    reason: str = ""  # cap.request: why the agent needs it

    def describe(self) -> str:
        if self.path is not None:
            return f"{self.right} {self.path}"
        if self.pid is not None:
            return f"{self.right} pid {self.pid}" + (f" ({self.signal})" if self.signal else "")
        if self.program is not None:
            return f"{self.right} {self.program}"
        if self.url is not None:
            return f"{self.right} {self.url}"
        if self.target is not None:
            return f"{self.right} {self.op} {self.target}".replace("  ", " ")
        return f"{self.right} {self.op}".strip()


@dataclass
class Check:
    requirement: Requirement
    verdict: Verdict
    reason: str
    grant: Grant | None = None
    human: bool = False  # must be confirmed by a person each time (no --yes, no "always")


@dataclass
class TemporaryGrant:
    """A capability the user granted for a limited time (just-in-time least privilege)."""

    id: str
    right: str
    target: str  # as the agent asked for it
    pattern: str  # canonical path glob, host glob, program or app glob
    reason: str
    expires: float | None  # epoch seconds; None = until this session ends
    created: float = field(default_factory=time.time)
    rule: PathRule | None = None

    def matches(self, value: str) -> bool:
        if self.rule is not None:
            return self.rule.matches(value)
        return fnmatch.fnmatchcase(value.lower(), self.pattern)

    def remaining(self) -> str:
        if self.expires is None:
            return "until this session ends"
        left = max(0, int(self.expires - time.time()))
        return f"{left // 60} min {left % 60} s left"


@dataclass
class Decision:
    verdict: Verdict
    reason: str
    checks: list[Check] = field(default_factory=list)

    def grant(self, right: str) -> Grant | None:
        return next((c.grant for c in self.checks if c.requirement.right == right and c.grant), None)

    def approved(self, index: int = 0) -> Requirement:
        """The exact object the broker checked; tools act on this, never re-resolve."""
        return self.checks[index].requirement

    @property
    def rights(self) -> list[str]:
        out: list[str] = []
        for c in self.checks:
            if c.requirement.right not in out:
                out.append(c.requirement.right)
        return out

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    @property
    def privileged(self) -> bool:
        """Runs as root."""
        return any(c.requirement.right == "sys.admin" for c in self.checks)

    @property
    def requires_human(self) -> bool:
        """Needs a live approval every time: never auto-approved, never 'always for this request'."""
        return self.privileged or any(c.human for c in self.checks)


def _deny(req: Requirement, reason: str) -> Check:
    return Check(req, Verdict.DENY, reason)


class CapabilityBroker:
    def __init__(self, policy: Policy, *, sandbox_active: bool = False) -> None:
        self.policy = policy
        self.sandbox_active = sandbox_active
        # The trusted computing base: files the supervised agent must never alter.
        self.protected = {
            os.path.realpath(policy.source),
            os.path.realpath(policy.audit_path),
            os.path.realpath(policy.audit_path) + ".lock",
            os.path.realpath(STATE_DIR),  # memory, saved sessions, quota state, history
        }
        # The agent process and its ancestors (the terminal, the shell, ...).
        self.lineage = set(procfs.lineage(os.getpid()))
        self._never_readable = [glob_to_regex(p) for p in NEVER_READABLE]
        self.temporary: list[TemporaryGrant] = []

    # ------------------------------------------------- temporary capabilities

    def active_grants(self) -> list[TemporaryGrant]:
        now = time.time()
        self.temporary = [g for g in self.temporary if g.expires is None or g.expires > now]
        return list(self.temporary)

    def _temp(self, right: str, value: str) -> TemporaryGrant | None:
        return next((g for g in self.active_grants() if g.right == right and g.matches(value)), None)

    def plan_grant(self, right: str, target: str, minutes: int | None, reason: str) -> TemporaryGrant:
        """Validate a capability request; raises ValueError explaining why it cannot be granted."""
        request = self.policy.grant("cap.request")
        if request is None:
            raise ValueError("this policy does not allow capability requests")
        if right not in GRANTABLE:
            raise ValueError(f"'{right}' cannot be granted temporarily (grantable: {', '.join(GRANTABLE)})")
        if not self.policy.has(right):
            raise ValueError(f"this policy grants no {right} at all; temporary grants only widen existing rights")
        if minutes is not None and not 1 <= minutes <= request.max_minutes:
            raise ValueError(f"duration must be between 1 and {request.max_minutes} minutes")
        target = (target or "").strip()
        if not target:
            raise ValueError("say what the capability should cover")
        expires = time.time() + minutes * 60 if minutes else None
        grant = TemporaryGrant(uuid.uuid4().hex[:6], right, target, "", reason.strip()[:300], expires)
        if right.startswith("fs."):
            raw = os.path.expanduser(target)
            if not os.path.isabs(raw):
                raw = os.path.join(str(self.policy.workspace), raw)
            if "*" not in raw and os.path.isdir(raw):
                raw = raw.rstrip("/") + "/**"
            try:
                pattern = canonical_pattern(raw, self.policy.workspace)
            except PolicyError as exc:
                raise ValueError(str(exc)) from exc
            literal = pattern.split("*", 1)[0].rstrip("/")
            depth = len([p for p in literal.split("/") if p])
            if depth < (3 if right != "fs.read" else 2):
                raise ValueError(f"'{target}' is too broad; ask for a specific folder")
            if self.policy.denied_by(literal or "/"):
                raise ValueError(f"'{target}' is covered by a deny rule, which temporary grants cannot override")
            grant.pattern, grant.rule = pattern, PathRule(target, pattern, glob_to_regex(pattern))
        elif right == "net.read":
            host = target.lower().removeprefix("https://").removeprefix("http://").split("/")[0]
            if "." not in host or host.replace("*", "").strip(".") == "":
                raise ValueError(f"'{target}' is not a specific host name")
            grant.pattern = host
        elif right == "exec":
            if "/" in target or not shutil.which(target, path=_SAFE_PATH):
                raise ValueError(f"'{target}' is not an installed program name")
            if not self.sandbox_active:
                raise ValueError("extra programs are only granted inside the sandbox, which is not active")
            grant.pattern = target.lower()
        else:  # gui.launch
            if target.strip("*. ") == "":
                raise ValueError(f"'{target}' is too broad")
            grant.pattern = target.lower().removesuffix(".desktop")
        return grant

    def add_grant(self, grant: TemporaryGrant) -> None:
        self.temporary.append(grant)

    def revoke(self, grant_id: str = "all") -> list[TemporaryGrant]:
        gone = [g for g in self.temporary if grant_id in ("all", g.id)]
        self.temporary = [g for g in self.temporary if g not in gone]
        return gone

    # ------------------------------------------------------------------ API

    def authorize(self, tool: "Tool", args: dict, ctx: "ToolContext") -> Decision:
        from agentlayer.tools.base import ToolArgError

        try:
            requirements = tool.requires(args, ctx)
        except ToolArgError as exc:
            return Decision(Verdict.DENY, f"invalid request: {exc}")
        return self.evaluate(requirements)

    def evaluate(self, requirements: list[Requirement]) -> Decision:
        if not requirements:
            return Decision(Verdict.DENY, "the action declared no capability requirements")
        checks = [self.check(r) for r in requirements]
        denied = [c for c in checks if c.verdict is Verdict.DENY]
        if denied:
            return Decision(Verdict.DENY, denied[0].reason, checks)
        confirms = [c for c in checks if c.verdict is Verdict.CONFIRM]
        if confirms:
            reasons = list(dict.fromkeys(c.reason for c in confirms))
            return Decision(Verdict.CONFIRM, "; ".join(reasons), checks)
        reasons = list(dict.fromkeys(c.reason for c in checks))
        return Decision(Verdict.ALLOW, "; ".join(reasons), checks)

    def check(self, req: Requirement) -> Check:
        if req.right.startswith("fs."):
            return self._check_fs(req)
        if req.right.startswith("proc."):
            return self._check_proc(req)
        if req.right == "exec":
            return self._check_exec(req)
        if req.right == "net.read":
            return self._check_net(req)
        if req.right.startswith("gui."):
            return self._check_gui(req)
        if req.right == "sys.admin":
            return self._check_admin(req)
        if req.right == "memory":
            grant = self.policy.grant("memory")
            if grant is None:
                return _deny(req, "this policy grants no memory capability")
            return Check(req, Verdict.ALLOW, f"memory {req.op or 'access'} granted", grant)
        if req.right in ("sys.read", "sys.logs"):
            grant = self.policy.grant(req.right)
            if grant is None:
                return _deny(req, f"this policy grants no {req.right} capability")
            return Check(req, Verdict.CONFIRM if grant.confirm else Verdict.ALLOW,
                         f"{req.right} " + ("needs your confirmation" if grant.confirm else "granted"), grant)
        if req.right == "task.schedule":
            grant = self.policy.grant("task.schedule")
            if grant is None:
                return _deny(req, "this policy grants no task.schedule capability")
            if req.op == "create":
                return Check(req, Verdict.CONFIRM, f"scheduling the unattended task '{req.target}' needs your "
                                                   "confirmation", grant, human=True)
            return Check(req, Verdict.ALLOW, f"task.schedule {req.op} granted", grant)
        if req.right == "cap.request":
            try:
                planned = self.plan_grant(req.op, req.target or "", req.minutes, req.reason)
            except ValueError as exc:
                return _deny(req, str(exc))
            how_long = f"{req.minutes} minutes" if req.minutes else "the rest of this session"
            return Check(req, Verdict.CONFIRM, f"the agent asks for temporary {planned.right} on "
                                               f"'{planned.target}' for {how_long}", self.policy.grant("cap.request"),
                         human=True)
        return _deny(req, f"unknown right {req.right!r}")

    def readable(self, path: str, lexical: str | None = None) -> bool:
        """Quick yes/no used by bulk operations (search, tree walks)."""
        return self._check_fs(Requirement("fs.read", path=path, lexical=lexical or path)).verdict is Verdict.ALLOW

    # ---------------------------------------------------------- filesystem

    def _check_fs(self, req: Requirement) -> Check:
        pol = self.policy
        path = req.path or ""
        for candidate in dict.fromkeys(p for p in (req.lexical, path) if p):
            rule = pol.denied_by(candidate)
            if rule:
                return _deny(req, f"{pol.display_path(candidate)} matches deny rule '{rule.raw}'")

        if req.right == "fs.read" and any(r.fullmatch(path) for r in self._never_readable):
            return _deny(req, f"{path} is never readable: it can expose secrets or raw memory")

        if req.right in ("fs.write", "fs.delete"):
            for protected in self.protected:
                inside = path == protected or path.startswith(protected + "/")
                if inside or (req.recursive and protected.startswith(path.rstrip("/") + "/")):
                    return _deny(req, f"{pol.display_path(protected)} belongs to the broker's trusted "
                                      "computing base and cannot be modified by the agent")
            if path == str(pol.workspace) and req.op in ("delete", "move-from"):
                return _deny(req, "the workspace root itself cannot be deleted or moved")

        if req.recursive:
            blocked = self._scan_subtree(path)
            if blocked:
                return _deny(req, blocked)

        for grant in pol.grants_for(req.right):
            rule = grant.match_path(path)
            if rule is None:
                continue
            shown = pol.display_path(path)
            if grant.confirm:
                return Check(req, Verdict.CONFIRM, f"{req.right} on {shown} needs your confirmation", grant)
            if req.overwrite and grant.confirm_overwrite:
                return Check(req, Verdict.CONFIRM, f"overwriting existing data at {shown} needs your confirmation", grant)
            return Check(req, Verdict.ALLOW, f"{req.right} granted on '{rule.raw}'", grant)
        temp = self._temp(req.right, path)
        if temp is not None:
            shown = pol.display_path(path)
            if req.right == "fs.delete" or (req.right == "fs.write" and req.overwrite):
                return Check(req, Verdict.CONFIRM, f"{req.right} on {shown} (temporary grant {temp.id}) needs "
                                                   "your confirmation", pol.grant(req.right))
            return Check(req, Verdict.ALLOW, f"{req.right} on {shown} by temporary grant {temp.id} "
                                             f"({temp.remaining()})", pol.grant(req.right))
        if not pol.has(req.right):
            return _deny(req, f"this policy grants no {req.right} capability")
        return _deny(req, f"no {req.right} capability covers {pol.display_path(path)}")

    def denied_under(self, root: str, limit: int = _SUBTREE_SCAN_LIMIT) -> list[str] | None:
        """Paths below ``root`` that deny rules cover (to be masked inside the sandbox).

        Returns None when the tree is too large to scan completely: such a
        directory must not be mounted, because unmasked secrets could remain.
        """
        found: list[str] = []
        seen = 0
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in dirnames + filenames:
                seen += 1
                full = os.path.join(dirpath, name)
                if self.policy.denied_by(full):
                    found.append(full)
            dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in found]
            if seen > limit:
                return None
        return found

    def writable(self, path: str) -> bool:
        return self.check(Requirement("fs.write", path=path, lexical=path)).verdict is not Verdict.DENY

    def _scan_subtree(self, root: str) -> str | None:
        """A recursive delete must not sweep away anything individually denied."""
        seen = 0
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in dirnames + filenames:
                seen += 1
                full = os.path.join(dirpath, name)
                rule = self.policy.denied_by(full)
                if rule:
                    return f"the tree contains {self.policy.display_path(full)}, which matches deny rule '{rule.raw}'"
                if seen > _SUBTREE_SCAN_LIMIT:
                    return f"the tree has more than {_SUBTREE_SCAN_LIMIT} entries; refusing a bulk delete that large"
        return None

    # ----------------------------------------------------------- processes

    def _check_proc(self, req: Requirement) -> Check:
        grant = self.policy.grant(req.right)
        if grant is None:
            return _deny(req, f"this policy grants no {req.right} capability")
        if req.pid is None:
            return Check(req, Verdict.ALLOW, f"{req.right} granted ({grant.scope} processes)", grant)

        pid = req.pid
        stat = procfs.read_stat(pid)
        if stat is None:
            return _deny(req, f"no process with PID {pid}")
        if req.right in ("proc.signal", "proc.priority"):
            if pid <= 1:
                return _deny(req, "PID 1 (init) is never a valid target")
            if req.right == "proc.signal" and pid in self.lineage:
                return _deny(req, f"PID {pid} is the agent itself or one of the processes it runs under")
            names = {stat.name}
            cmd = procfs.read_cmdline(pid)
            if cmd:
                names.add(os.path.basename(cmd[0]))
            hit = next((n for n in names for p in grant.protect if fnmatch.fnmatchcase(n, p)), None)
            if hit:
                return _deny(req, f"process '{hit}' (PID {pid}) is protected by the policy")
        if grant.scope == "own":
            uid = procfs.real_uid(pid)
            if uid != os.getuid():
                owner = procfs.username(uid) if uid is not None else "another user"
                return _deny(req, f"PID {pid} belongs to {owner}; this capability covers only your own processes")
        if req.right == "proc.signal" and req.signal not in grant.signals:
            return _deny(req, f"{req.signal} is not in the granted signal set ({', '.join(grant.signals)})")
        label = f"PID {pid} ({stat.name})"
        if grant.confirm:
            what = f"sending {req.signal} to" if req.right == "proc.signal" else "re-prioritising"
            return Check(req, Verdict.CONFIRM, f"{what} {label} needs your confirmation", grant)
        return Check(req, Verdict.ALLOW, f"{req.right} granted on {label}", grant)

    # ---------------------------------------------------------------- exec

    def _check_exec(self, req: Requirement) -> Check:
        grant = self.policy.grant("exec")
        if grant is None:
            return _deny(req, "this policy grants no exec capability")
        prog = req.program or ""
        ws = str(self.policy.workspace)
        cwd = req.cwd or ws
        roots = [ws, *grant.extra_dirs]
        if not any(cwd == r or cwd.startswith(r + "/") for r in roots):
            shown = ", ".join(self.policy.display_path(r) for r in roots)
            return _deny(req, f"commands may only run inside: {shown}")
        if not prog:
            return _deny(req, "empty command")
        if "/" in prog:
            return self._check_workspace_program(req, grant, prog, cwd)
        temporary = prog not in grant.programs and self._temp("exec", prog) is not None
        if prog not in grant.programs and not temporary:
            return _deny(req, f"program '{prog}' is not on the exec allow-list")
        if (prog in grant.sandboxed_only or temporary) and not self.sandbox_active:
            return _deny(req, f"'{prog}' can run arbitrary code, so it is only allowed inside the "
                              "bubblewrap sandbox, which is not active")
        argv = req.argv or [prog]
        # Argument rules stop programs from launching *other* programs (find -exec, xargs ...).
        # Inside the sandbox anything launched stays confined, so they only apply without it.
        for pattern in ([] if self.sandbox_active else grant.deny_args.get(prog, [])):
            hit = next((a for a in argv[1:] if fnmatch.fnmatchcase(a, pattern)), None)
            if hit is not None:
                return _deny(req, f"argument '{hit}' is not allowed for {prog} outside the sandbox")

        if not self.sandbox_active:
            # Without a mount namespace, a path argument is the only way a
            # program reaches files; hold it to the fs.read capability.
            for arg in argv[1:]:
                for value in (arg, arg.partition("=")[2] if arg.startswith("-") else ""):
                    if not value:
                        continue
                    if (value.startswith("/") or ".." in value.split("/")
                            or os.path.lexists(os.path.join(cwd, value))):
                        target = os.path.realpath(os.path.join(cwd, value))
                        if not self.readable(target, os.path.normpath(os.path.join(cwd, value))):
                            return _deny(req, f"argument '{arg}' refers to {self.policy.display_path(target)}, "
                                              "which is outside your fs.read capability")

        network = self._check_network(req, grant)
        if network is not None:
            return network
        if grant.confirm or prog in grant.confirm_programs:
            return Check(req, Verdict.CONFIRM, f"running '{prog}' needs your confirmation", grant)
        where = "in the sandbox" if self.sandbox_active else "unsandboxed"
        return Check(req, Verdict.ALLOW, f"exec granted for '{prog}' ({where})", grant)

    def _check_network(self, req: Requirement, grant: Grant) -> Check | None:
        if not req.network:
            return None
        if grant.network == "never":
            return _deny(req, "commands may not use the internet under this policy")
        if not self.sandbox_active:
            return _deny(req, "internet access for commands is only granted inside the sandbox")
        if grant.network == "confirm":
            return Check(req, Verdict.CONFIRM, f"'{req.program}' wants internet access; this needs your confirmation", grant)
        return None

    def _check_workspace_program(self, req: Requirement, grant: Grant, prog: str, cwd: str) -> Check:
        """A program addressed by path: allowed only if it was built inside the workspace."""
        if not grant.workspace_programs:
            return _deny(req, f"'{prog}': this policy only runs allow-listed programs by bare name")
        if not self.sandbox_active:
            return _deny(req, f"'{prog}': programs from the workspace are arbitrary code and only run inside "
                              "the bubblewrap sandbox, which is not active")
        target = os.path.realpath(os.path.join(cwd, prog))
        roots = [str(self.policy.workspace), *grant.extra_dirs]
        if not any(target.startswith(r + "/") for r in roots):
            return _deny(req, f"'{prog}' resolves to {self.policy.display_path(target)}, outside the workspace")
        if not (os.path.isfile(target) and os.access(target, os.X_OK)):
            return _deny(req, f"'{prog}' is not an executable file in the workspace")
        network = self._check_network(req, grant)
        if network is not None:
            return network
        if grant.confirm:
            return Check(req, Verdict.CONFIRM, f"running '{prog}' needs your confirmation", grant)
        return Check(req, Verdict.ALLOW, f"exec granted for workspace program '{prog}' (in the sandbox)", grant)

    # ----------------------------------------------------------------- desktop

    def _check_gui(self, req: Requirement) -> Check:
        grant = self.policy.grant(req.right)
        if grant is None:
            return _deny(req, f"this policy grants no {req.right} capability")
        what = req.target or req.op
        if req.right == "gui.launch" and req.op == "app":
            if grant.app_listed(req.program or "") or self._temp("gui.launch", req.program or ""):
                verdict = Verdict.CONFIRM if grant.confirm else Verdict.ALLOW
                return Check(req, verdict, f"launching {what} " + ("needs your confirmation" if grant.confirm
                                                                  else "granted (allow-listed app)"), grant)
            if grant.confirm_unlisted:
                return Check(req, Verdict.CONFIRM, f"{what} is not on the app allow-list; launching it needs your confirmation", grant)
            return _deny(req, f"{what} is not on the app allow-list")
        if req.right == "gui.windows" and req.op == "close":
            if req.pid is not None and req.pid in self.lineage:
                return _deny(req, f"the window '{what}' belongs to the program AgentLayer is running in")
            if grant.confirm or grant.confirm_close:
                return Check(req, Verdict.CONFIRM, f"closing '{what}' may lose unsaved work; this needs your confirmation", grant)
        if req.right == "gui.clipboard" and req.op == "read" and (grant.confirm or grant.confirm_read):
            return Check(req, Verdict.CONFIRM, "reading the clipboard (it may hold passwords) needs your confirmation", grant)
        if grant.confirm:
            action = {"gui.screen": "taking a screenshot (the screen may show private things)",
                      "gui.launch": f"launching {what}", "gui.windows": f"{req.op} window {what}",
                      "gui.clipboard": f"clipboard {req.op}"}.get(req.right, f"{req.right} {what}")
            return Check(req, Verdict.CONFIRM, f"{action} needs your confirmation", grant)
        return Check(req, Verdict.ALLOW, f"{req.right} granted ({what})", grant)

    # ------------------------------------------------------------------ admin

    def _check_admin(self, req: Requirement) -> Check:
        grant = self.policy.grant("sys.admin")
        if grant is None:
            return _deny(req, "this policy grants no sys.admin capability")
        if req.op == "run":
            prog = req.program or ""
            if "/" in prog or prog not in grant.programs:
                return _deny(req, f"'{prog}' is not on the list of programs that may run as root")
            for pattern in grant.deny_args.get(prog, []):
                hit = next((a for a in (req.argv or [])[1:] if fnmatch.fnmatchcase(a, pattern)), None)
                if hit is not None:
                    return _deny(req, f"'{prog} {hit}' is not allowed as root")
            return Check(req, Verdict.CONFIRM, f"runs '{' '.join(req.argv or [prog])}' as root", grant)
        manager = req.program or ""
        if manager not in grant.managers:
            return _deny(req, f"package manager '{manager}' is not allowed by this policy")
        names = ", ".join(req.packages or [])
        return Check(req, Verdict.CONFIRM, f"{req.op}s {names} with {manager} (changes installed software)", grant)

    # ------------------------------------------------------------- network

    def _check_net(self, req: Requirement) -> Check:
        grant = self.policy.grant("net.read")
        if grant is None:
            return _deny(req, "this policy grants no net.read capability")
        parts = urlsplit(req.url or "")
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return _deny(req, "only http:// and https:// URLs can be fetched")
        if parts.username or parts.password:
            return _deny(req, "URLs with embedded credentials are not allowed")
        host = parts.hostname.lower()
        if not grant.allow_private:
            reason = _private_address(host)
            if reason:
                return _deny(req, reason)
        if grant.host_listed(host) or self._temp("net.read", host):
            if grant.confirm:
                return Check(req, Verdict.CONFIRM, f"fetching from {host} needs your confirmation", grant)
            return Check(req, Verdict.ALLOW, f"net.read granted for {host}", grant)
        if grant.confirm_unlisted:
            return Check(req, Verdict.CONFIRM, f"{host} is not on the host allow-list; fetching needs your confirmation", grant)
        return _deny(req, f"{host} is not on the host allow-list")


def _private_address(host: str) -> str | None:
    """Refuse hosts that resolve to loopback, LAN or other non-public addresses."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return f"cannot resolve host {host}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            return f"{host} resolves to {ip}, a private or local address; only public hosts can be fetched"
    return None
