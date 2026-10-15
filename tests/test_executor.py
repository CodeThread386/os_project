import os
import shutil

import pytest

from agentlayer.broker import Verdict
from agentlayer.executor import CommandSyntaxError, run_pipeline, split_pipeline
from agentlayer.policy import ExecLimits
from agentlayer.tools import REGISTRY, prepare_args

from .conftest import make_env, needs_bwrap

LIMITS = ExecLimits(cpu_seconds=5, memory_mb=1024)
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def words(command):
    return [[w for w, _ in stage] for stage in split_pipeline(command)]


def test_tokenizer():
    assert words("ls -la") == [["ls", "-la"]]
    assert words("grep 'a b' f | wc -l") == [["grep", "a b", "f"], ["wc", "-l"]]
    assert words('echo "x \\"y\\"" z\\ z') == [["echo", 'x "y"', "z z"]]
    assert words("echo a|b") == [["echo", "a"], ["b"]]
    assert words("echo price=$5") == [["echo", "price=$5"]]
    stage = split_pipeline("ls *.py '*.txt'")[0]
    assert stage == [("ls", True), ("*.py", True), ("*.txt", False)]


@pytest.mark.parametrize("bad", ["ls; id", "a && b", "a || b", "a > f", "a < f", "a &", "echo $(id)",
                                 "echo `id`", "echo $HOME", 'echo "$HOME"', "a |", "| a", "'open", '"open', ""])
def test_tokenizer_rejects_shell_features(bad):
    with pytest.raises(CommandSyntaxError):
        split_pipeline(bad)


def run(stages, tmp_path, **kw):
    resolved = [(shutil.which(argv[0]), argv) for argv in stages]
    opts = {"cwd": str(tmp_path), "env": ENV, "timeout": 10, "max_output": 10_000, "limits": LIMITS}
    opts.update(kw)
    return run_pipeline(resolved, **opts)


def test_simple_command(tmp_path):
    r = run([["echo", "hello"]], tmp_path)
    assert r.stdout == "hello\n" and r.exit_code == 0 and r.status == "exited with status 0"


def test_exit_code_and_stderr(tmp_path):
    r = run([["ls", "/definitely/missing"]], tmp_path)
    assert r.exit_code != 0 and "missing" in r.stderr


def test_pipeline(tmp_path):
    r = run([["seq", "1", "100"], ["sort", "-rn"], ["head", "-3"]], tmp_path)
    assert r.stdout.split() == ["100", "99", "98"] and len(r.stages) == 3


def test_sigpipe_restored(tmp_path):
    """`yes | head -1` only terminates if the child has the default SIGPIPE action."""
    r = run([["yes"], ["head", "-1"]], tmp_path, timeout=5)
    assert r.stdout == "y\n" and not r.timed_out
    assert r.stages[0].signal == "SIGPIPE"


def test_timeout_kills_process_group(tmp_path):
    r = run([["sleep", "30"]], tmp_path, timeout=0.5)
    assert r.timed_out and r.stages[0].signal == "SIGKILL" and r.duration < 5


def test_cpu_limit(tmp_path):
    r = run([["python3", "-c", "while True: pass"]], tmp_path, limits=ExecLimits(cpu_seconds=1), timeout=10)
    assert r.stages[0].signal in ("SIGXCPU", "SIGKILL") and "CPU" in r.status or r.stages[0].signal == "SIGKILL"


def test_output_truncated_but_drained(tmp_path):
    r = run([["seq", "1", "200000"]], tmp_path, max_output=100)
    assert len(r.stdout) == 100 and r.truncated and r.exit_code == 0 and r.stdout_bytes > 1_000_000


def test_exec_failure_reported(tmp_path):
    r = run_pipeline([("/nonexistent/prog", ["prog"])], cwd=str(tmp_path), env=ENV, timeout=5,
                     max_output=1000, limits=LIMITS)
    assert r.exit_code == 127 and "cannot execute" in r.stderr


def test_environment_is_not_inherited(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_must_not_leak")
    r = run([["python3", "-c", "import os; print(sorted(os.environ))"]], tmp_path)
    assert "GROQ_API_KEY" not in r.stdout


def test_resource_usage_collected(tmp_path):
    r = run([["python3", "-c", "sum(range(3_000_000))"]], tmp_path)
    assert r.user_cpu > 0 and r.max_rss_kb > 1000


def test_no_child_left_behind(tmp_path):
    import signal
    run([["sleep", "20"]], tmp_path, timeout=0.3)
    with pytest.raises(ChildProcessError):
        os.waitpid(-1, os.WNOHANG)
    assert signal.getsignal(signal.SIGPIPE) == signal.SIG_IGN  # parent's own disposition untouched


# ------------------------------------------------------------- tool + broker


def run_tool(env, command, **kw):
    t = REGISTRY["run_command"]
    args = prepare_args(t, {"command": command, **kw})
    d = env.broker.authorize(t, args, env.ctx)
    assert d.verdict is Verdict.ALLOW, d.reason
    return t.run(args, env.ctx, d)


def test_run_command_tool_with_glob(env):
    for n in ("a.txt", "b.txt", "c.md"):
        (env.ws / n).write_text(n)
    out = run_tool(env, "ls *.txt")
    assert out.data["stdout"].split() == ["a.txt", "b.txt"]


@needs_bwrap
def test_sandbox_isolation(tmp_path):
    env = make_env(tmp_path, sandbox="bwrap")
    probe = (
        "import os, socket\n"
        "print('home', os.path.exists(%r))\n"
        "print('pids', len([p for p in os.listdir('/proc') if p.isdigit()]))\n"
        "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=2); print('net yes')\n"
        "except OSError: print('net no')\n"
        "try:\n    open('/usr/agentlayer-test', 'w'); print('usr writable')\n"
        "except OSError: print('usr ro')\n"
        "open('made-in-sandbox.txt', 'w').write('ok')\n"
    ) % str(tmp_path / "home")
    (env.ws / "probe.py").write_text(probe)
    out = run_tool(env, "python3 probe.py")
    assert out.ok, out.data
    lines = out.data["stdout"].split("\n")
    assert "home False" in lines and "net no" in lines and "usr ro" in lines
    assert int(lines[1].split()[1]) <= 3
    assert (env.ws / "made-in-sandbox.txt").read_text() == "ok"


@needs_bwrap
def test_sandbox_runs_workspace_program(tmp_path):
    env = make_env(tmp_path, sandbox="bwrap")
    script = env.ws / "hello.sh"
    script.write_text("#!/bin/sh\necho built-here\n")
    script.chmod(0o755)
    assert run_tool(env, "./hello.sh").data["stdout"] == "built-here\n"


def test_denied_workspace_file_unreadable_by_commands_without_sandbox(env):
    (env.ws / ".env").write_text("TOKEN=secret")
    t = REGISTRY["run_command"]
    d = env.broker.authorize(t, prepare_args(t, {"command": "cat .env"}), env.ctx)
    assert d.verdict is Verdict.DENY and "outside your fs.read capability" in d.reason


@needs_bwrap
def test_denied_workspace_files_are_masked_in_sandbox(tmp_path):
    env = make_env(tmp_path, sandbox="bwrap")
    (env.ws / "app").mkdir()
    (env.ws / "app" / ".env").write_text("TOKEN=supersecret")
    (env.ws / "cert.pem").write_text("-----BEGIN PRIVATE KEY-----")
    (env.ws / "readme.txt").write_text("public")
    out = run_tool(env, "python3 -c \"import pathlib; print([p.read_text() for p in sorted(pathlib.Path('.').rglob('*')) if p.is_file()])\"")
    assert "public" in out.data["stdout"]
    assert "supersecret" not in out.data["stdout"] and "PRIVATE KEY" not in out.data["stdout"]
    assert (env.ws / "app" / ".env").read_text() == "TOKEN=supersecret"  # untouched on the host
    t = REGISTRY["run_command"]
    for cmd in ("cat app/.env", "cat cert.pem", "python3 -c \"print(open('app/.env').read())\""):
        args = prepare_args(t, {"command": cmd})
        result = t.run(args, env.ctx, env.broker.authorize(t, args, env.ctx))
        assert result.data["exit_code"] != 0 and "Permission denied" in result.data["stderr"], cmd
        assert "supersecret" not in result.data["stdout"] and "PRIVATE" not in result.data["stdout"]
