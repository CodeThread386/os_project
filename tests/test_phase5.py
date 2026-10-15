"""Phase 5 capabilities: home folders, internet for commands, desktop rights, root, memory."""

import os
import subprocess

import pytest

from agentlayer.broker import Requirement, Verdict
from agentlayer.policy import PolicyError, load_policy
from agentlayer.tools import REGISTRY, prepare_args

from .conftest import make_env, needs_bwrap

GUI_ADMIN = """
[[grant]]
right = "gui.launch"
apps = ["org.example.editor"]
confirm_unlisted = true

[[grant]]
right = "gui.windows"
confirm_close = true

[[grant]]
right = "gui.screen"
confirm = true

[[grant]]
right = "gui.clipboard"
confirm_read = true

[[grant]]
right = "sys.admin"
confirm = false
programs = ["echo", "id", "systemctl"]
managers = ["apt", "pipx"]
[grant.deny_args]
systemctl = ["reboot", "poweroff"]

[[grant]]
right = "memory"
"""


def decide(env, tool, **args):
    t = REGISTRY[tool]
    return env.broker.authorize(t, prepare_args(t, args), env.ctx)


# ------------------------------------------------------------------ policy


def test_phase5_is_superset_of_phase4(tmp_path):
    p4 = {g.right for g in load_policy("phase4", workspace=str(tmp_path / "ws")).grants}
    p5 = load_policy("phase5", workspace=str(tmp_path / "ws"))
    assert p4 <= {g.right for g in p5.grants}
    assert p5.grant("exec").network == "confirm" and p5.grant("exec").extra_dirs
    assert p5.grant("sys.admin").confirm  # forced on


def test_admin_confirmation_cannot_be_disabled(tmp_path):
    env = make_env(tmp_path, extra=GUI_ADMIN)  # the grant says confirm = false
    assert env.policy.grant("sys.admin").confirm is True


@pytest.mark.parametrize("bad", ['network = "sometimes"', 'extra_dirs = ["~/Down*"]'])
def test_exec_option_validation(tmp_path, bad):
    with pytest.raises(PolicyError):
        make_env(tmp_path, exec_extra=bad)


def test_unknown_package_manager_rejected(tmp_path):
    with pytest.raises(PolicyError, match="package manager"):
        make_env(tmp_path, extra='[[grant]]\nright = "sys.admin"\nmanagers = ["brew"]\n')


def test_phase5_home_rules(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for d in ("Downloads", ".config", "projects"):
        (home / d).mkdir(parents=True)
    (home / "Downloads" / "report.pdf").write_text("pdf")
    (home / "projects" / "notes.md").write_text("notes")
    (home / ".bashrc").write_text("secret alias")
    (home / ".config" / "app.conf").write_text("token=1")
    monkeypatch.setenv("HOME", str(home))
    from agentlayer.broker import CapabilityBroker
    from agentlayer.sandbox import Sandbox
    from agentlayer.tools import ToolContext

    policy = load_policy("phase5", workspace=str(tmp_path / "ws"))
    broker = CapabilityBroker(policy)
    ctx = ToolContext(policy, broker, Sandbox("none", "t"))

    def verdict(tool, **args):
        t = REGISTRY[tool]
        return broker.authorize(t, prepare_args(t, args), ctx).verdict

    assert verdict("read_file", path=str(home / "projects/notes.md")) is Verdict.ALLOW
    assert verdict("read_file", path=str(home / ".bashrc")) is Verdict.DENY
    assert verdict("read_file", path=str(home / ".config/app.conf")) is Verdict.DENY
    assert verdict("write_file", path=str(home / "Downloads/new.txt"), content="x") is Verdict.ALLOW
    assert verdict("write_file", path=str(home / "Downloads/report.pdf"), content="x") is Verdict.CONFIRM
    assert verdict("write_file", path=str(home / "projects/notes.md"), content="x") is Verdict.DENY
    assert verdict("delete_path", path=str(home / "Downloads/report.pdf")) is Verdict.CONFIRM
    assert verdict("move_path", source=str(home / "Downloads/report.pdf"),
                   destination=str(home / "projects")) is Verdict.DENY  # ~/projects is read-only
    (home / "Documents").mkdir()
    assert verdict("move_path", source=str(home / "Downloads/report.pdf"),
                   destination=str(home / "Documents")) is Verdict.ALLOW


# ------------------------------------------------------- internet + extra dirs


def test_network_modes(tmp_path):
    never = make_env(tmp_path / "a", exec_extra='network = "never"')
    d = decide(never, "run_command", command="echo hi", network=True)
    assert d.verdict is Verdict.DENY and "internet" in d.reason
    # without the sandbox internet cannot be confined, so it is refused
    confirm = make_env(tmp_path / "b", exec_extra='network = "confirm"')
    assert "only granted inside the sandbox" in decide(confirm, "run_command", command="echo hi", network=True).reason
    assert decide(confirm, "run_command", command="echo hi").verdict is Verdict.ALLOW


def _sandbox_run(env, command, **kw):
    t = REGISTRY["run_command"]
    args = prepare_args(t, {"command": command, **kw})
    d = env.broker.authorize(t, args, env.ctx)
    return d, (t.run(args, env.ctx, d) if d.verdict is Verdict.ALLOW else None)


@needs_bwrap
def test_network_confirm_then_sandbox_has_interfaces(tmp_path):
    env = make_env(tmp_path, sandbox="bwrap", exec_extra='network = "always"')
    count = "python3 -c \"print(len(open('/proc/net/dev').read().splitlines()) - 2)\""
    _, offline = _sandbox_run(env, count)
    _, online = _sandbox_run(env, count, network=True)
    assert int(offline.data["stdout"]) == 1  # loopback only
    assert int(online.data["stdout"]) >= 1 and "internet" in online.data["sandbox"]
    env2 = make_env(tmp_path / "c", sandbox="bwrap", exec_extra='network = "confirm"')
    assert decide(env2, "run_command", command="python3 -V", network=True).verdict is Verdict.CONFIRM


@needs_bwrap
def test_extra_dirs_mounted_with_masking(tmp_path):
    downloads = tmp_path / "Downloads"
    docs = tmp_path / "Docs"
    downloads.mkdir()
    docs.mkdir()
    (downloads / "a.txt").write_text("hello")
    (downloads / "creds.pem").write_text("KEY")
    (docs / "readonly.txt").write_text("ro")
    env = make_env(tmp_path, sandbox="bwrap", exec_extra=f'extra_dirs = ["{downloads}", "{docs}"]',
                   extra=f'\n[[grant]]\nright = "fs.write"\npaths = ["{downloads}/**"]\n')
    d, out = _sandbox_run(env, "cat a.txt", cwd=str(downloads))
    assert out.data["stdout"] == "hello"
    _, out = _sandbox_run(env, "cat creds.pem", cwd=str(downloads))
    assert out.data["exit_code"] != 0 and "KEY" not in out.data["stdout"]
    _, out = _sandbox_run(env, f"python3 -c \"open('{downloads}/made.txt','w').write('ok')\"")
    assert out.ok and (downloads / "made.txt").read_text() == "ok"
    _, out = _sandbox_run(env, f"python3 -c \"open('{docs}/x.txt','w').write('no')\"")
    assert not out.ok and not (docs / "x.txt").exists()  # mounted read-only: no fs.write grant
    assert decide(env, "run_command", command="ls", cwd=str(tmp_path)).verdict is Verdict.DENY


# ---------------------------------------------------------------- desktop


def test_gui_rights(tmp_path):
    env = make_env(tmp_path, extra=GUI_ADMIN)
    check = env.broker.check
    assert check(Requirement("gui.launch", op="app", program="org.example.editor")).verdict is Verdict.ALLOW
    assert check(Requirement("gui.launch", op="app", program="org.other.app")).verdict is Verdict.CONFIRM
    assert check(Requirement("gui.launch", op="notify")).verdict is Verdict.ALLOW
    assert check(Requirement("gui.windows", op="list")).verdict is Verdict.ALLOW
    assert check(Requirement("gui.windows", op="focus", pid=999999)).verdict is Verdict.ALLOW
    assert check(Requirement("gui.windows", op="close", pid=999999)).verdict is Verdict.CONFIRM
    own = env.broker.check(Requirement("gui.windows", op="close", pid=os.getppid(), target="my terminal"))
    assert own.verdict is Verdict.DENY and "AgentLayer is running in" in own.reason
    assert check(Requirement("gui.screen", op="screen")).verdict is Verdict.CONFIRM
    assert check(Requirement("gui.clipboard", op="read")).verdict is Verdict.CONFIRM
    assert check(Requirement("gui.clipboard", op="write")).verdict is Verdict.ALLOW


def test_gui_rights_absent_by_default(env):
    assert env.broker.check(Requirement("gui.launch", op="notify")).verdict is Verdict.DENY


# ------------------------------------------------------------------ admin


def test_admin_decisions(tmp_path):
    env = make_env(tmp_path, extra=GUI_ADMIN)
    d = decide(env, "run_as_root", command="id -u")
    assert d.verdict is Verdict.CONFIRM and d.privileged and "as root" in d.reason
    assert decide(env, "run_as_root", command="systemctl reboot").verdict is Verdict.DENY
    assert decide(env, "run_as_root", command="rm -rf /").verdict is Verdict.DENY
    assert "pipes" in decide(env, "run_as_root", command="echo a | wc").reason
    assert decide(env, "packages", action="install", names="cowsay").verdict is Verdict.CONFIRM
    assert decide(env, "packages", action="install", names="vlc", manager="snap").verdict is Verdict.DENY
    assert decide(env, "packages", action="search", names="editor").verdict is Verdict.ALLOW
    for bad in ("-y", "foo;rm", "a b c$(x)"):
        assert "invalid package" in decide(env, "packages", action="install", names=bad).reason


def test_memory_right(tmp_path):
    env = make_env(tmp_path, extra=GUI_ADMIN)
    assert decide(env, "memory", action="remember", text="likes vim").verdict is Verdict.ALLOW
    assert "give the fact" in decide(env, "memory", action="remember").reason


def test_state_dir_is_protected(env):
    from agentlayer.config import STATE_DIR

    req = Requirement("fs.write", path=str(STATE_DIR / "memory.json"), lexical=str(STATE_DIR / "memory.json"))
    assert env.broker.check(req).verdict is Verdict.DENY


def test_signal_tools_still_guarded_in_phase5(tmp_path):
    proc = subprocess.Popen(["sleep", "30"])
    try:
        policy = load_policy("phase5", workspace=str(tmp_path / "ws"))
        from agentlayer.broker import CapabilityBroker
        broker = CapabilityBroker(policy)
        assert broker.check(Requirement("proc.signal", pid=proc.pid, signal="SIGTERM")).verdict is Verdict.CONFIRM
        assert broker.check(Requirement("proc.signal", pid=1, signal="SIGTERM")).verdict is Verdict.DENY
    finally:
        proc.kill()
        proc.wait()
