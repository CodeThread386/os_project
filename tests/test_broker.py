import os
import subprocess

import pytest

from agentlayer import procfs
from agentlayer.broker import Requirement, Verdict
from agentlayer.tools import REGISTRY, prepare_args

from .conftest import make_env


def decide(env, tool, **args):
    t = REGISTRY[tool]
    return env.broker.authorize(t, prepare_args(t, args), env.ctx)


def test_workspace_read_allowed(env):
    (env.ws / "a.txt").write_text("x")
    d = decide(env, "read_file", path="a.txt")
    assert d.verdict is Verdict.ALLOW


def test_deny_rule_overrides_grant(env):
    (env.ws / ".env").write_text("SECRET=1")
    d = decide(env, "read_file", path=".env")
    assert d.verdict is Verdict.DENY and "deny rule" in d.reason


def test_ssh_key_denied(env):
    d = decide(env, "read_file", path=str(env.root / "home/.ssh/id_rsa"))
    assert d.verdict is Verdict.DENY


def test_dotdot_traversal_denied(env):
    d = decide(env, "read_file", path="../policy.toml")
    assert d.verdict is Verdict.DENY and "no fs.read capability" in d.reason


def test_symlink_escape_denied(env):
    (env.ws / "innocent.txt").symlink_to(env.root / "home/.ssh/id_rsa")
    d = decide(env, "read_file", path="innocent.txt")
    assert d.verdict is Verdict.DENY


def test_symlink_to_secret_with_safe_name_denied(env):
    outside = env.root / "outside.txt"
    outside.write_text("not granted")
    (env.ws / "link").symlink_to(outside)
    assert decide(env, "read_file", path="link").verdict is Verdict.DENY


def test_lexical_deny_applies_even_if_target_is_allowed(env):
    (env.ws / "ok.txt").write_text("fine")
    (env.ws / "x.pem").symlink_to(env.ws / "ok.txt")
    assert decide(env, "read_file", path="x.pem").verdict is Verdict.DENY


def test_write_outside_workspace_denied(env):
    d = decide(env, "write_file", path=str(env.root / "shared/x.txt"), content="x")
    assert d.verdict is Verdict.DENY


def test_policy_and_audit_log_are_protected(env):
    env.audit.record("test")
    d = decide(env, "write_file", path=str(env.policy.source), content="")
    assert d.verdict is Verdict.DENY
    req = Requirement("fs.delete", path=str(env.policy.audit_path), lexical=str(env.policy.audit_path), op="delete")
    check = env.broker.check(req)
    assert check.verdict is Verdict.DENY and "trusted computing base" in check.reason


def test_audit_log_inside_workspace_still_protected(tmp_path):
    e = make_env(tmp_path)
    e.policy.audit_path = e.ws / "audit.jsonl"
    from agentlayer.broker import CapabilityBroker
    broker = CapabilityBroker(e.policy)
    (e.ws / "audit.jsonl").write_text("{}\n")
    req = Requirement("fs.write", path=str(e.ws / "audit.jsonl"), lexical=str(e.ws / "audit.jsonl"))
    assert broker.check(req).verdict is Verdict.DENY
    tree = Requirement("fs.delete", path=str(e.ws), lexical=str(e.ws), recursive=True, op="delete")
    assert broker.check(tree).verdict is Verdict.DENY


def test_workspace_root_cannot_be_deleted(env):
    (env.ws / "f").write_text("x")
    d = decide(env, "delete_path", path=".", recursive=True)
    assert d.verdict is Verdict.DENY and "workspace root" in d.reason


def test_recursive_delete_cannot_sweep_denied_file(env):
    (env.ws / "proj").mkdir()
    (env.ws / "proj" / ".env").write_text("SECRET")
    d = decide(env, "delete_path", path="proj", recursive=True)
    assert d.verdict is Verdict.DENY and ".env" in d.reason


def test_proc_environ_never_readable(env):
    d = decide(env, "read_file", path=f"/proc/{os.getpid()}/environ")
    assert d.verdict is Verdict.DENY and "never readable" in d.reason
    assert decide(env, "read_file", path="/proc/self/environ").verdict is Verdict.DENY


def test_confirmation_for_delete_and_overwrite(confirm_env):
    env = confirm_env
    (env.ws / "f.txt").write_text("data")
    assert decide(env, "delete_path", path="f.txt").verdict is Verdict.CONFIRM
    assert decide(env, "write_file", path="f.txt", content="new").verdict is Verdict.CONFIRM
    assert decide(env, "write_file", path="new.txt", content="new").verdict is Verdict.ALLOW
    assert decide(env, "write_file", path="f.txt", content="more", mode="append").verdict is Verdict.ALLOW


# ---------------------------------------------------------------- processes


@pytest.fixture
def sleeper():
    proc = subprocess.Popen(["sleep", "30"])
    yield proc
    proc.kill()
    proc.wait()


def test_signal_own_process_allowed(env, sleeper):
    assert decide(env, "signal_process", pid=sleeper.pid, signal="TERM").verdict is Verdict.ALLOW


def test_signal_needs_confirmation_when_configured(confirm_env, sleeper):
    assert decide(confirm_env, "signal_process", pid=sleeper.pid).verdict is Verdict.CONFIRM


def test_signal_pid1_denied(env):
    assert "init" in decide(env, "signal_process", pid=1).reason


def test_signal_self_and_ancestors_denied(env):
    for pid in procfs.lineage(os.getpid())[:2]:
        d = decide(env, "signal_process", pid=pid)
        assert d.verdict is Verdict.DENY and "agent itself" in d.reason


def test_signal_other_users_process_denied(env):
    root_pid = next((p for p in procfs.list_pids() if p > 1 and procfs.real_uid(p) == 0
                     and p not in env.broker.lineage), None)
    if root_pid is None or os.getuid() == 0:
        pytest.skip("no root-owned process to test against")
    d = decide(env, "signal_process", pid=root_pid)
    assert d.verdict is Verdict.DENY and "your own processes" in d.reason


def test_signal_protected_name_denied(env, tmp_path):
    exe = tmp_path / "protected-daemon"
    exe.symlink_to("/usr/bin/sleep")
    proc = subprocess.Popen([str(exe), "30"])
    try:
        d = decide(env, "signal_process", pid=proc.pid)
        assert d.verdict is Verdict.DENY and "protected" in d.reason
    finally:
        proc.kill()
        proc.wait()


def test_signal_not_in_granted_set(tmp_path, sleeper):
    e = make_env(tmp_path)
    e.policy.grant("proc.signal").signals = ["SIGTERM"]
    assert "not in the granted signal set" in decide(e, "signal_process", pid=sleeper.pid, signal="KILL").reason


def test_missing_process(env):
    assert "no process" in decide(env, "process_info", pid=4_000_000).reason


# --------------------------------------------------------------------- exec


def test_exec_allowlist(env):
    assert decide(env, "run_command", command="ls -l | wc -l").verdict is Verdict.ALLOW
    d = decide(env, "run_command", command="curl http://example.com")
    assert d.verdict is Verdict.DENY and "allow-list" in d.reason
    assert decide(env, "run_command", command="ls | curl x").verdict is Verdict.DENY


def test_exec_shell_syntax_rejected(env):
    for cmd in ("ls; rm -rf /", "ls && id", "echo hi > f", "cat < f", "echo $(id)", "echo `id`", "echo $HOME"):
        d = decide(env, "run_command", command=cmd)
        assert d.verdict is Verdict.DENY and "invalid request" in d.reason, cmd


def test_exec_sandboxed_only_without_sandbox(env):
    d = decide(env, "run_command", command="python3 -c 'print(1)'")
    assert d.verdict is Verdict.DENY and "sandbox" in d.reason


def test_exec_deny_args(env):
    d = decide(env, "run_command", command="find . -delete")
    assert d.verdict is Verdict.DENY and "-delete" in d.reason
    assert decide(env, "run_command", command="find . -execdir rm {} +").verdict is Verdict.DENY


def test_exec_path_arguments_checked_without_sandbox(env):
    d = decide(env, "run_command", command=f"cat {env.root}/home/.ssh/id_rsa")
    assert d.verdict is Verdict.DENY
    assert decide(env, "run_command", command="cat ../policy.toml").verdict is Verdict.DENY
    assert decide(env, "run_command", command="grep --file=/etc/hostname x").verdict is Verdict.DENY
    (env.ws / "notes.txt").write_text("x")
    assert decide(env, "run_command", command="cat notes.txt").verdict is Verdict.ALLOW


def test_exec_workspace_program_needs_sandbox(env):
    prog = env.ws / "a.out"
    prog.write_text("#!/bin/sh\n")
    prog.chmod(0o755)
    d = decide(env, "run_command", command="./a.out")
    assert d.verdict is Verdict.DENY and "sandbox" in d.reason


def test_exec_cwd_must_be_in_workspace(env):
    d = decide(env, "run_command", command="ls", cwd="..")
    assert d.verdict is Verdict.DENY and "may only run inside" in d.reason


# ------------------------------------------------------------------ network


def test_net_listed_host_allowed(env):
    assert decide(env, "fetch_url", url="https://example.com/x").verdict is Verdict.ALLOW


def test_net_unlisted_host(env, confirm_env):
    assert decide(env, "fetch_url", url="https://www.iana.org").verdict is Verdict.DENY
    assert decide(confirm_env, "fetch_url", url="https://www.iana.org").verdict is Verdict.CONFIRM


def test_net_private_addresses_denied(tmp_path):
    e = make_env(tmp_path, extra="")
    e.policy.grant("net.read").hosts = ["*"]
    for url in ("http://localhost:11434/api/tags", "http://127.0.0.1/", "http://169.254.169.254/latest/meta-data"):
        d = decide(e, "fetch_url", url=url)
        assert d.verdict is Verdict.DENY and "private or local" in d.reason, url


def test_net_bad_scheme_and_credentials(env):
    assert decide(env, "fetch_url", url="file:///etc/passwd").verdict is Verdict.DENY
    assert decide(env, "fetch_url", url="ftp://example.com/").verdict is Verdict.DENY
    assert "credentials" in decide(env, "fetch_url", url="https://u:p@example.com/").reason


# -------------------------------------------------------------- phase gates


def test_phase1_has_no_write_capability(tmp_path):
    from agentlayer.broker import CapabilityBroker
    from agentlayer.policy import load_policy
    from agentlayer.sandbox import Sandbox
    from agentlayer.tools import ToolContext

    policy = load_policy("phase1", workspace=str(tmp_path / "ws"))
    broker = CapabilityBroker(policy)
    ctx = ToolContext(policy, broker, Sandbox("none", "test"))
    t = REGISTRY["write_file"]
    d = broker.authorize(t, prepare_args(t, {"path": "a", "content": "b"}), ctx)
    assert d.verdict is Verdict.DENY and "grants no fs.write" in d.reason
