"""The memory tool: long-term notes that survive between sessions."""

from __future__ import annotations

from agentlayer.broker import Requirement
from agentlayer.tools.base import Tool, ToolArgError, ToolContext, ToolOutput, param, register


def _requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    action = args["action"]
    if action == "remember" and not (args.get("text") or "").strip():
        raise ToolArgError("give the fact to remember in 'text'")
    if action == "forget" and not args.get("id"):
        raise ToolArgError("give the id of the note to forget (from recall)")
    return [Requirement("memory", op="read" if action == "recall" else "write")]


def _memory(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    store = ctx.memory
    if store is None:
        return ToolOutput.fail("long-term memory is not available in this session")
    action = args["action"]
    if action == "remember":
        fact = store.add(args["text"], ctx.session)
        return ToolOutput(True, {"saved": fact.text, "id": fact.id}, f"remembered: {fact.text[:70]}")
    if action == "forget":
        fact = store.forget(args["id"])
        if fact is None:
            return ToolOutput.fail(f"no note with id {args['id']}")
        return ToolOutput(True, {"forgotten": fact.text}, f"forgot: {fact.text[:70]}")
    facts = store.search(args.get("text") or "", limit=25)
    text = "\n".join(f"[{f.id}] ({f.when}) {f.text}" for f in facts) or "(nothing remembered yet)"
    return ToolOutput(True, {"notes": text, "count": len(facts)}, f"{len(facts)} note(s) recalled", text)


register(Tool(
    name="memory",
    description="Long-term memory across sessions: 'remember' a lasting fact or preference the user tells you, "
                "'recall' notes (optionally matching text), or 'forget' one by id.",
    rights=("memory",),
    params={
        "action": param("string", "remember, recall or forget", enum=["remember", "recall", "forget"]),
        "text": param("string", "The fact to remember, or words to search for"),
        "id": param("string", "Note id to forget"),
    },
    required=("action",),
    requires=_requires,
    run=_memory,
))
