import json

import pytest

from agentlayer.agent import Agent, AgentEvents
from agentlayer.llm import LLMReply, ToolCall


class ScriptedLLM:
    """Plays back a fixed list of replies and records what it was sent."""

    provider, model = "fake", "scripted"

    def __init__(self, *replies, budget=100_000):
        self.replies = list(replies)
        self.calls = []
        self.budget = budget

    def complete(self, messages, tools):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": [t["function"]["name"] for t in tools]})
        return self.replies.pop(0)

    def context_budget(self):
        return self.budget


def call(name, **args):
    return LLMReply("", [ToolCall(f"c_{name}_{len(args)}", name, json.dumps(args))], "scripted", "fake")


def answer(text):
    return LLMReply(text, [], "scripted", "fake")


class Recorder(AgentEvents):
    def __init__(self, approve=False):
        self.approve = approve
        self.decisions = []
        self.confirms = []

    def decision(self, name, args, decision):
        self.decisions.append((name, decision.verdict.value))

    def confirm(self, tool, args, decision, preview):
        self.confirms.append((tool.name, preview))
        return self.approve


def tool_results(llm, index=-1):
    return [json.loads(m["content"].split("…[truncated")[0]) for m in llm.calls[index]["messages"] if m["role"] == "tool"]


def test_request_plan_execute_observe(env):
    llm = ScriptedLLM(call("write_file", path="hello.txt", content="hi\n"), call("read_file", path="hello.txt"),
                      answer("Wrote and read hello.txt"))
    agent = Agent(llm, env.broker, env.ctx, env.audit)
    assert agent.ask("make a file") == "Wrote and read hello.txt"
    assert (env.ws / "hello.txt").read_text() == "hi\n"
    assert tool_results(llm)[-1]["content"] == "hi\n"  # the result was fed back to the model
    events = [e["event"] for e in env.audit.read(env.policy.audit_path)]
    assert events == ["request", "action", "action", "answer"]


def test_denial_is_reported_to_model_and_audited(env):
    llm = ScriptedLLM(call("read_file", path=str(env.root / "home/.ssh/id_rsa")), answer("I was not allowed"))
    events = Recorder()
    Agent(llm, env.broker, env.ctx, env.audit, events).ask("read my key")
    assert "Permission denied" in tool_results(llm)[0]["error"]
    assert events.decisions == [("read_file", "deny")]
    record = env.audit.read(env.policy.audit_path)[1]
    assert record["verdict"] == "deny" and "deny rule" in record["reason"]


def test_declined_confirmation_blocks_action(confirm_env):
    env = confirm_env
    (env.ws / "precious.txt").write_text("keep me")
    llm = ScriptedLLM(call("delete_path", path="precious.txt"), answer("ok"))
    events = Recorder(approve=False)
    Agent(llm, env.broker, env.ctx, env.audit, events).ask("delete it")
    assert (env.ws / "precious.txt").exists()
    assert events.confirms[0][0] == "delete_path" and "permanently deleted" in events.confirms[0][1].lower()
    assert "declined" in tool_results(llm)[0]["error"]
    assert env.audit.read(env.policy.audit_path)[1]["confirmed"] is False


def test_approved_confirmation_runs_action(confirm_env):
    env = confirm_env
    (env.ws / "old.txt").write_text("x")
    llm = ScriptedLLM(call("delete_path", path="old.txt"), answer("deleted"))
    Agent(llm, env.broker, env.ctx, env.audit, Recorder(approve=True)).ask("delete it")
    assert not (env.ws / "old.txt").exists()


def test_unknown_tool_and_bad_json(env):
    bad = LLMReply("", [ToolCall("x1", "format_disk", "{}"), ToolCall("x2", "read_file", "{not json")], "s", "fake")
    llm = ScriptedLLM(bad, answer("done"))
    Agent(llm, env.broker, env.ctx, env.audit).ask("do stuff")
    results = tool_results(llm)
    assert "unknown tool" in results[0]["error"] and "not valid JSON" in results[1]["error"]


def test_os_errors_become_observations(env):
    llm = ScriptedLLM(call("read_file", path="missing.txt"), answer("it does not exist"))
    Agent(llm, env.broker, env.ctx, env.audit).ask("read missing.txt")
    assert "ENOENT" in tool_results(llm)[0]["error"]


def test_step_limit(env):
    llm = ScriptedLLM(*[call("system_info") for _ in range(10)])
    agent = Agent(llm, env.broker, env.ctx, env.audit, max_steps=3)
    assert "stopped after 3 steps" in agent.ask("loop forever")
    assert len(llm.calls) == 3


def test_only_granted_tools_are_exposed(tmp_path):
    from agentlayer.broker import CapabilityBroker
    from agentlayer.policy import load_policy
    from agentlayer.sandbox import Sandbox
    from agentlayer.tools import ToolContext
    from agentlayer.audit import AuditLog

    policy = load_policy("phase1", workspace=str(tmp_path / "ws"))
    broker = CapabilityBroker(policy)
    llm = ScriptedLLM(answer("hi"))
    Agent(llm, broker, ToolContext(policy, broker, Sandbox("none", "t")), AuditLog(tmp_path / "a.jsonl")).ask("hi")
    exposed = set(llm.calls[0]["tools"])
    assert "read_file" in exposed and "write_file" not in exposed and "run_command" not in exposed


def test_interrupt_rolls_back_transcript(env):
    class Boom(ScriptedLLM):
        def complete(self, messages, tools):
            raise KeyboardInterrupt

    agent = Agent(Boom(), env.broker, env.ctx, env.audit)
    before = list(agent.messages)
    with pytest.raises(KeyboardInterrupt):
        agent.ask("anything")
    assert agent.messages == before


def test_conversation_memory_and_compaction(env):
    (env.ws / "big.txt").write_text("y" * 3500)
    replies = []
    for _ in range(6):
        replies += [call("read_file", path="big.txt"), answer("read it")]
    llm = ScriptedLLM(*replies, budget=4000)
    agent = Agent(llm, env.broker, env.ctx, env.audit)
    for _ in range(6):
        agent.ask("read big.txt again")
    last_prompt = llm.calls[-1]["messages"]
    assert last_prompt[0]["role"] == "system"
    assert sum(len(json.dumps(m)) for m in last_prompt) < 4000 * 4.5  # stayed near the budget
    assert last_prompt[-2]["role"] == "assistant" or any(m["role"] == "user" for m in last_prompt)


def test_system_prompt_describes_policy(env):
    agent = Agent(ScriptedLLM(), env.broker, env.ctx, env.audit)
    prompt = agent.messages[0]["content"]
    assert str(env.ws) in prompt and "fs.write" in prompt and "never accessible" in prompt


def test_current_turn_results_are_shrunk_to_fit(env):
    for i in range(4):
        (env.ws / f"f{i}.txt").write_text(str(i) * 3900)
    llm = ScriptedLLM(*[call("read_file", path=f"f{i}.txt") for i in range(4)], answer("done"), budget=3200)
    Agent(llm, env.broker, env.ctx, env.audit).ask("read all four files")
    final = llm.calls[-1]
    size = sum(len(json.dumps(m)) for m in final["messages"])
    assert size < 3200 * 4.2 and any("shortened" in m["content"] for m in final["messages"] if m["role"] == "tool")


def test_context_too_large_is_retried_with_less_context(env):
    from agentlayer.llm import ContextTooLarge

    class TooBig(ScriptedLLM):
        def complete(self, messages, tools):
            self.calls.append(len(json.dumps(messages)))
            if len(self.calls) == 1:
                raise ContextTooLarge("Request too large ... Limit 7000, Requested 7946")
            return answer("ok")

    (env.ws / "big.txt").write_text("z" * 3900)
    agent = Agent(TooBig(budget=6000), env.broker, env.ctx, env.audit)
    agent.messages.append({"role": "user", "content": "earlier"})
    agent.messages.append({"role": "assistant", "content": "x" * 9000})
    assert agent.ask("hi") == "ok"
    assert agent.llm.calls[1] < agent.llm.calls[0]
