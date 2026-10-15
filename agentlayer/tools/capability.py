"""The request_capability tool: just-in-time least privilege.

When the policy is too narrow for a task, the agent may ask for one specific,
expiring capability ("fs.write on ~/Projects/site/** for 30 minutes") and say
why. The user approves it live, every time; deny rules and the broker's
invariants still apply on top of it, root cannot be requested, and the user
can list and revoke grants at any moment (/grants, /revoke).
"""

from __future__ import annotations

from agentlayer.broker import GRANTABLE, Requirement
from agentlayer.tools.base import Tool, ToolArgError, ToolContext, ToolOutput, param, register


def _requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    if not args["reason"].strip():
        raise ToolArgError("explain why the capability is needed")
    return [Requirement("cap.request", op=args["right"], target=args["target"], minutes=args.get("minutes"),
                        reason=args["reason"])]


def _run(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    try:
        grant = ctx.broker.plan_grant(args["right"], args["target"], args.get("minutes"), args["reason"])
    except ValueError as exc:
        return ToolOutput.fail(str(exc))
    ctx.broker.add_grant(grant)
    data = {"granted": grant.right, "covers": grant.target, "id": grant.id, "lasts": grant.remaining(),
            "note": "deny rules and confirmations still apply; retry the action now"}
    return ToolOutput(True, data, f"temporary {grant.right} on {grant.target} granted ({grant.remaining()})")


register(Tool(
    name="request_capability",
    description="Ask the user for a temporary capability when the broker denied an action only because no grant "
                "covers it (never for deny rules). Be specific: one folder, host, program or app.",
    rights=("cap.request",),
    params={
        "right": param("string", "The right needed", enum=list(GRANTABLE)),
        "target": param("string", "Exactly what it should cover: a folder or path glob, a host, a program, an app id"),
        "reason": param("string", "Why the task needs it (shown to the user)"),
        "minutes": param("integer", "How long it should last (omit for the rest of this session)", minimum=1,
                         maximum=240),
    },
    required=("right", "target", "reason"),
    requires=_requires,
    run=_run,
    preview=lambda a, c: f"grant {a['right']} on {a['target']} "
                         f"for {str(a['minutes']) + ' minutes' if a.get('minutes') else 'the rest of this session'}\n"
                         f"reason given by the agent: {a['reason']}",
))
