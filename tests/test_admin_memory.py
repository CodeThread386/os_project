"""Root actions through a fake sudo, long-term memory, and resumable sessions."""

import json
import textwrap

import pytest

from agentlayer.agent import Agent
from agentlayer.broker import Verdict
from agentlayer.memory import MemoryStore, SessionStore
from agentlayer.tools import REGISTRY, admin, prepare_args
from agentlayer.ui import ConsoleEvents

from .conftest import make_env
from .test_agent import Recorder, ScriptedLLM, answer, call, tool_results
from .test_phase5 import GUI_ADMIN

PASSWORD = "correct horse battery staple"


@pytest.fixture
def fake_sudo(tmp_path, monkeypatch):
    """A stand-in for sudo: no cached credentials, accepts one password on stdin."""
    script = tmp_path / "sudo"
    script.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import os, sys
        args = sys.argv[1:]
        if args[:2] == ["-n", "true"]:
            sys.exit(1)                     # credentials are never cached here
        if args[0] == "-S":
            if sys.stdin.readline().rstrip("\\n") != {PASSWORD!r}:
                print("Sorry, try again.", file=sys.stderr)
                print("sudo: 1 incorrect password attempt", file=sys.stderr)
                sys.exit(1)
            args = args[3:]                 # drop -S -p ''
        os.execv(args[0], args)
    """))
    script.chmod(0o755)
    monkeypatch.setattr(admin, "SUDO", str(script))
    return script


def admin_env(tmp_path, password):
    env = make_env(tmp_path / "env", extra=GUI_ADMIN)
    env.ctx.ask_secret = lambda prompt: password
    return env


def run_root(env, command):
    t = REGISTRY["run_as_root"]
    args = prepare_args(t, {"command": command})
    d = env.broker.authorize(t, args, env.ctx)
    assert d.verdict is Verdict.CONFIRM and d.privileged
    return t.run(args, env.ctx, d)


def test_root_command_with_password(tmp_path, fake_sudo):
    out = run_root(admin_env(tmp_path, PASSWORD), "echo installed")
    assert out.ok and out.data["output_tail"] == "installed\n"


def test_wrong_password(tmp_path, fake_sudo):
    out = run_root(admin_env(tmp_path, "guess"), "echo nope")
    assert not out.ok and "did not accept the password" in out.data["error"]


def test_no_terminal_no_password(tmp_path, fake_sudo):
    env = make_env(tmp_path / "env", extra=GUI_ADMIN)  # ask_secret is None
    out = run_root(env, "echo nope")
    assert not out.ok and "no terminal" in out.data["error"]


def test_password_never_reaches_model_or_audit(tmp_path, fake_sudo):
    env = make_env(tmp_path / "env", extra=GUI_ADMIN)
    events = Recorder(approve=True)
    events.secret = lambda prompt: PASSWORD
    llm = ScriptedLLM(call("run_as_root", command="echo done"), answer("ok"))
    Agent(llm, env.broker, env.ctx, env.audit, events).ask("run it")
    assert "done" in tool_results(llm)[0]["output_tail"]
    everything = json.dumps(llm.calls) + env.policy.audit_path.read_text()
    assert PASSWORD not in everything


def test_root_actions_are_never_auto_approved(tmp_path):
    env = make_env(tmp_path, extra=GUI_ADMIN)
    t = REGISTRY["run_as_root"]
    args = prepare_args(t, {"command": "id"})
    d = env.broker.authorize(t, args, env.ctx)
    from rich.console import Console

    events = ConsoleEvents(Console(file=open("/dev/null", "w")), interactive=False, auto_approve=True)
    assert events.confirm(t, args, d, None) is False  # --yes does not cover root


def test_always_approval_is_per_request_and_not_for_root(tmp_path, fake_sudo):
    env = make_env(tmp_path / "env", confirm=True, extra=GUI_ADMIN)
    for name in ("a", "b", "c"):
        (env.ws / name).write_text(name)

    class Always(Recorder):
        def confirm(self, tool, args, decision, preview):
            self.confirms.append(tool.name)
            return "always"

    events = Always()
    events.secret = lambda prompt: PASSWORD
    llm = ScriptedLLM(call("delete_path", path="a"), call("delete_path", path="b"),
                      call("run_as_root", command="echo 1"), call("run_as_root", command="echo 2"),
                      answer("done"), call("delete_path", path="c"), answer("done"))
    agent = Agent(llm, env.broker, env.ctx, env.audit, events)
    agent.ask("clean up")
    assert events.confirms == ["delete_path", "run_as_root", "run_as_root"]  # second delete auto, root asked twice
    agent.ask("one more")
    assert events.confirms[-1] == "delete_path"  # the approval did not outlive the request
    assert not any((env.ws / n).exists() for n in "abc")


# ----------------------------------------------------------------- memory


def test_memory_store(tmp_path):
    store = MemoryStore(tmp_path / "m.json", max_items=3)
    a = store.add("Project lives in ~/Documents/os-lab")
    assert store.add("project LIVES in ~/documents/os-lab").id == a.id  # duplicate ignored
    store.add("Prefers concise answers")
    store.add("Uses zsh")
    store.add("Teammates: Mrinal and Vaaman")
    assert len(store.all()) == 3 and store.all()[0].text == "Prefers concise answers"
    assert store.search("zsh")[0].text == "Uses zsh"
    assert "Teammates" in store.prompt_block().splitlines()[0]
    assert store.forget(store.search("zsh")[0].id) and len(store.all()) == 2
    assert store.clear() == 2 and store.all() == []
    assert (tmp_path / "m.json").stat().st_mode & 0o777 == 0o600


def test_memory_tool_and_prompt(tmp_path):
    env = make_env(tmp_path / "env", extra=GUI_ADMIN)
    env.ctx.memory = MemoryStore(tmp_path / "m.json")
    llm = ScriptedLLM(call("memory", action="remember", text="The user's name is Sarthak"), answer("noted"))
    Agent(llm, env.broker, env.ctx, env.audit).ask("remember my name is Sarthak")
    later = ScriptedLLM(answer("hi"))
    Agent(later, env.broker, env.ctx, env.audit).ask("who am I?")
    assert "The user's name is Sarthak" in later.calls[0]["messages"][0]["content"]
    recall = ScriptedLLM(call("memory", action="recall", text="name"), answer("Sarthak"))
    Agent(recall, env.broker, env.ctx, env.audit).ask("what's my name")
    assert "Sarthak" in tool_results(recall)[0]["notes"]


# --------------------------------------------------------------- sessions


def test_sessions_save_and_resume(tmp_path):
    env = make_env(tmp_path / "env")
    sessions = SessionStore(tmp_path / "sessions")
    first = Agent(ScriptedLLM(answer("Paris")), env.broker, env.ctx, env.audit, sessions=sessions)
    first.ask("What is the capital of France?")
    saved = sessions.list()
    assert saved[0].title == "What is the capital of France?" and saved[0].turns == 1

    llm = ScriptedLLM(answer("About 2 million people"))
    second = Agent(llm, env.broker, env.ctx, env.audit, sessions=sessions)
    sid, messages = sessions.load()
    assert second.resume(sid, messages) == 1
    second.ask("And its population?")
    sent = [m["content"] for m in llm.calls[0]["messages"]]
    assert "What is the capital of France?" in sent and "Paris" in sent
    assert sessions.list()[0].turns == 2 and len(sessions.list()) == 1  # same conversation continued


def test_session_ids_are_validated(tmp_path):
    with pytest.raises(ValueError):
        SessionStore(tmp_path).path("../../etc/passwd")


def test_claimed_memory_write_is_caught(tmp_path):
    """The model says it saved a note without calling the tool: the agent makes it actually do so."""
    env = make_env(tmp_path / "env", extra=GUI_ADMIN)
    env.ctx.memory = MemoryStore(tmp_path / "m.json")
    llm = ScriptedLLM(answer("I stored that."),  # a false claim
                      call("memory", action="remember", text="Partners: Mrinal and Vaaman"),
                      answer("Saved."))
    reply = Agent(llm, env.broker, env.ctx, env.audit).ask("Remember that my partners are Mrinal and Vaaman")
    assert reply == "Saved." and env.ctx.memory.all()[0].text == "Partners: Mrinal and Vaaman"
    assert "[AgentLayer check]" in llm.calls[1]["messages"][-1]["content"]


def test_no_memory_nudge_for_ordinary_requests(tmp_path):
    env = make_env(tmp_path / "env", extra=GUI_ADMIN)
    env.ctx.memory = MemoryStore(tmp_path / "m.json")
    llm = ScriptedLLM(answer("42"))
    assert Agent(llm, env.broker, env.ctx, env.audit).ask("what is 6 x 7?") == "42"
    assert len(llm.calls) == 1
