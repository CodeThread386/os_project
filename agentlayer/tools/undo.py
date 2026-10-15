"""The undo tool: list and reverse the agent's recent file changes."""

from __future__ import annotations

from agentlayer.broker import Requirement
from agentlayer.journal import UndoError
from agentlayer.tools.base import Tool, ToolArgError, ToolContext, ToolOutput, param, register


def _entry(args: dict, ctx: ToolContext):
    if ctx.journal is None:
        raise ToolArgError("undo is not available in this session")
    try:
        return ctx.journal.find(args.get("id"))
    except UndoError as exc:
        raise ToolArgError(str(exc)) from exc


def _requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    if args["action"] == "list":
        return [Requirement("fs.read", path=ctx.workspace, lexical=ctx.workspace, op="list")]
    entry = _entry(args, ctx)
    return [Requirement(right, path=path, lexical=path, op="undo", overwrite=overwrite)
            for right, path, overwrite in ctx.journal.inverse_targets(entry)]


def _run(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    if args["action"] == "list":
        rows = ctx.journal.entries(15) if ctx.journal else []
        text = "\n".join(f"[{e.id}] {e.when} {'(undone) ' if e.undone else ''}{e.summary}"
                         for e in rows) or "(no changes recorded)"
        return ToolOutput(True, {"changes": text}, f"{len(rows)} recent change(s)", text)
    entry = _entry(args, ctx)
    try:
        message = ctx.journal.undo(entry.id)
    except UndoError as exc:
        return ToolOutput.fail(str(exc))
    return ToolOutput(True, {"undone": entry.id, "result": message}, message)


register(Tool(
    name="undo",
    description="Undo a recent file change made by you (overwrite, append, create, delete, move, copy, mkdir); "
                "action=list shows them. Deleted items are in the Trash and come back with undo.",
    rights=("fs.write",),
    params={
        "action": param("string", "list or undo", enum=["list", "undo"], default="undo"),
        "id": param("string", "Change id from list (default: the most recent)"),
    },
    requires=_requires,
    run=_run,
    preview=lambda a, c: f"undo: {_entry(a, c).summary}",
))
