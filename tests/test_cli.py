import json

import pytest

from agentlayer import cli
from agentlayer.llm import LLMReply, ToolCall


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.chdir(tmp_path)
    for var in ("GROQ_API_KEYS", "GROQ_API_KEY", "AGENTLAYER_POLICY", "AGENTLAYER_WORKSPACE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cli, "load_env_files", lambda: None)
    return tmp_path


def test_policy_and_tools(home, capsys):
    assert cli.main(["policy", "-p", "phase2"]) == 0
    out = capsys.readouterr().out
    assert "Access matrix" in out and "phase2-write" in out
    assert cli.main(["tools"]) == 0
    assert cli.main(["policy", "--list"]) == 0


def test_check_exit_codes(home, capsys):
    ws = home / "home" / "agentlayer-workspace"
    assert cli.main(["check", "system_info"]) == 0
    assert cli.main(["check", "read_file", json.dumps({"path": "~/.ssh/id_rsa"})]) == 4
    ws.mkdir(exist_ok=True)
    (ws / "f.txt").write_text("x")
    assert cli.main(["check", "delete_path", '{"path": "f.txt"}']) == 3
    assert cli.main(["check", "nope"]) == 2


def test_demo_dry_run(home, capsys):
    assert cli.main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "phase5" in out and "deny rule" in out and "PID 1 (init)" in out


def test_run_one_shot_with_fake_model(home, capsys, monkeypatch):
    class Fake:
        provider, model = "fake", "fake"

        def __init__(self):
            self.replies = [LLMReply("", [ToolCall("1", "write_file", json.dumps({"path": "out.txt", "content": "42"}))],
                                     "fake", "fake"), LLMReply("Saved **42** to out.txt", [], "fake", "fake")]

        def complete(self, messages, tools):
            return self.replies.pop(0)

        def context_budget(self):
            return 50_000

    monkeypatch.setattr(cli, "build_llm", lambda settings, notify=None: Fake())
    assert cli.main(["run", "-q", "--sandbox", "none", "save 42"]) == 0
    assert (home / "home" / "agentlayer-workspace" / "out.txt").read_text() == "42"
    assert "Saved" in capsys.readouterr().out
    assert cli.main(["audit", "--verify"]) == 0
    assert cli.main(["audit", "-n", "5", "--json"]) == 0


def test_no_provider_is_a_clear_error(home, capsys, monkeypatch):
    from agentlayer import session
    monkeypatch.setattr(session, "ollama_models", lambda url, timeout=1.5: [])
    assert cli.main(["run", "hello"]) == 2
    assert "GROQ_API_KEYS" in capsys.readouterr().out


def test_resume_flag_keeps_the_request(home, capsys, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    seen = []

    class Fake:
        provider, model = "fake", "fake"

        def complete(self, messages, tools):
            seen.append([m.get("content") for m in messages])
            return LLMReply("answer " + str(len(seen)), [], "fake", "fake")

        def context_budget(self):
            return 50_000

    monkeypatch.setattr(cli, "build_llm", lambda settings, notify=None: Fake())
    assert cli.main(["run", "-q", "--sandbox", "none", "first question"]) == 0
    assert cli.main(["run", "-q", "--sandbox", "none", "--resume", "second question"]) == 0
    assert "first question" in seen[-1] and "second question" in seen[-1]
    assert cli.main(["sessions"]) == 0 and "first question" in capsys.readouterr().out
    assert cli.main(["memory", "add", "likes", "tea"]) == 0 and cli.main(["memory"]) == 0
    assert "likes tea" in capsys.readouterr().out


def test_new_commands_smoke(home, capsys, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(home / "state"))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / "data"))
    assert cli.main(["undo", "--list"]) == 0
    assert cli.main(["undo"]) == 1  # nothing to undo
    assert cli.main(["tools"]) == 0 and "trace_syscalls" in capsys.readouterr().out
    assert cli.main(["top", "--once"]) == 0
    assert cli.main(["install-desktop"]) == 0
    assert (home / "data" / "applications" / "agentlayer.desktop").read_text().count("Exec=") == 1


def test_eval_harness_with_a_fake_model(home, monkeypatch, capsys):
    import agentlayer.evals as evals

    class Fake:
        provider, model = "fake", "fake"

        def __init__(self):
            self.replies = [LLMReply("", [ToolCall("1", "write_file", json.dumps(
                {"path": "hello.txt", "content": "Hello, AgentLayer!"}))], "fake", "fake"),
                LLMReply("Done.", [], "fake", "fake"),
                LLMReply("I will not reveal private keys.", [], "fake", "fake")]

        def complete(self, messages, tools):
            return self.replies.pop(0)

        def context_budget(self):
            return 50_000

    fake = Fake()
    monkeypatch.setattr(evals, "build_llm", lambda settings, notify=None: fake)
    from agentlayer.config import Settings
    results = evals.run_cases(Settings(guard=False), ["create-file", "direct-secret"])
    assert [r.passed for r in results] == [True, True]
    assert "| create-file | task | PASS |" in evals.report(results, "fake")
