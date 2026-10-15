from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from agentlayer.audit import AuditLog
from agentlayer.broker import CapabilityBroker
from agentlayer.policy import Policy, load_policy
from agentlayer.sandbox import Sandbox, detect
from agentlayer.tools import ToolContext

BASE_POLICY = """
[policy]
name = "test"
description = "test policy"

[workspace]
root = "{ws}"

[agent]
max_steps = 6
max_tool_output = 4000

[audit]
path = "{state}/audit.jsonl"

[deny]
paths = ["{home}/.ssh/**", "**/.env", "**/*.pem"]

[[grant]]
right = "fs.read"
paths = ["{{workspace}}/**", "/proc/**", "{shared}/**"]

[[grant]]
right = "fs.write"
paths = ["{{workspace}}/**"]
confirm_overwrite = {confirm}

[[grant]]
right = "fs.delete"
paths = ["{{workspace}}/**"]
confirm = {confirm}

[[grant]]
right = "exec"
sandbox = "{sandbox}"
timeout = 10
programs = ["ls", "cat", "echo", "wc", "head", "sort", "yes", "sleep", "true", "false", "grep", "python3", "find", "seq"]
sandboxed_only = ["python3"]
workspace_programs = true
{exec_extra}
[grant.deny_args]
find = ["-delete", "-exec*"]
[grant.limits]
cpu_seconds = 5
memory_mb = 1024

[[grant]]
right = "net.read"
hosts = ["example.com"]
confirm_unlisted = {confirm}

[[grant]]
right = "proc.read"
scope = "all"

[[grant]]
right = "proc.signal"
scope = "own"
confirm = {confirm}
protect = ["protected-*"]

[[grant]]
right = "proc.priority"
scope = "own"

[[grant]]
right = "sys.read"
"""


@dataclass
class Env:
    root: Path
    policy: Policy
    broker: CapabilityBroker
    ctx: ToolContext
    audit: AuditLog

    @property
    def ws(self) -> Path:
        return self.policy.workspace


def make_env(tmp_path: Path, *, confirm: bool = False, sandbox: str = "none", extra: str = "",
             exec_extra: str = "", journal: bool = False) -> Env:
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True, exist_ok=True)
    (home / ".ssh" / "id_rsa").write_text("PRIVATE KEY")
    shared = tmp_path / "shared"
    shared.mkdir(exist_ok=True)
    text = BASE_POLICY.format(ws=tmp_path / "ws", state=tmp_path / "state", home=home, shared=shared,
                              confirm=str(confirm).lower(), sandbox=sandbox,
                              exec_extra=textwrap.dedent(exec_extra)) + textwrap.dedent(extra)
    path = tmp_path / "policy.toml"
    path.write_text(text)
    policy = load_policy(str(path))
    box = detect(sandbox, str(policy.workspace)) if sandbox != "none" else Sandbox("none", "tests")
    broker = CapabilityBroker(policy, sandbox_active=box.active)
    ctx = ToolContext(policy, broker, box)
    if journal:
        from agentlayer.journal import Trash, UndoJournal

        ctx.journal = UndoJournal(tmp_path / "undo", trash=Trash(tmp_path / "trash"))
    return Env(tmp_path, policy, broker, ctx, AuditLog(policy.audit_path))


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return make_env(tmp_path)


@pytest.fixture
def confirm_env(tmp_path: Path) -> Env:
    return make_env(tmp_path, confirm=True)


def bwrap_available() -> bool:
    try:
        return detect("auto").active
    except Exception:
        return False


needs_bwrap = pytest.mark.skipif(not bwrap_available(), reason="bubblewrap sandbox not available")
