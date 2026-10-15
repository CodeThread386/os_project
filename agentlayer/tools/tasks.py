"""Tools for unattended automation: schedule_task and tasks."""

from __future__ import annotations

from agentlayer.broker import Requirement
from agentlayer.scheduler import ScheduleError, Scheduler
from agentlayer.tools.base import Tool, ToolArgError, ToolContext, ToolOutput, param, register


def _scheduler() -> Scheduler:
    return Scheduler()


def _schedule_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    try:
        _scheduler().plan(args["name"], args["when"])
    except ScheduleError as exc:
        raise ToolArgError(str(exc)) from exc
    return [Requirement("task.schedule", op="create", target=args["name"])]


def _schedule_preview(args: dict, ctx: ToolContext) -> str:
    kind, _spec, next_run = _scheduler().plan(args["name"], args["when"])
    repeat = "once" if kind == "once" else f"repeatedly ({args['when']})"
    return (f"task '{args['name']}' runs {repeat}, next at {next_run}, under policy {ctx.policy.name}:\n"
            f"  \"{args['request']}\"\n"
            "It runs unattended: anything that needs a confirmation is declined automatically, and nothing "
            "runs as root. Results go to a desktop notification and the task log.")


def _schedule(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    grant = decision.grant("task.schedule")
    try:
        task = _scheduler().create(args["name"], args["request"], args["when"], ctx.policy.name,
                                   grant.max_tasks if grant else 20)
    except ScheduleError as exc:
        return ToolOutput.fail(str(exc))
    data = {"task": task.name, "next_run": task.next_run, "repeats": task.kind == "calendar",
            "systemd_units": f"agentlayer-task-{task.name}.timer/.service (user)"}
    return ToolOutput(True, data, f"scheduled '{task.name}', next run {task.next_run}")


register(Tool(
    name="schedule_task",
    description="Schedule a request to run later, unattended, on a systemd user timer: 'in 20 minutes', "
                "'every day at 9:00', 'every monday at 18:30', 'every 30 minutes', or an OnCalendar expression.",
    rights=("task.schedule",),
    params={
        "name": param("string", "Short id: lowercase letters, digits, dashes"),
        "request": param("string", "The instruction to carry out at that time, in plain words as the user would say "
                                   "it, e.g. 'Send me a desktop notification saying the backup is due'"),
        "when": param("string", "When to run"),
    },
    required=("name", "request", "when"),
    requires=_schedule_requires,
    run=_schedule,
    preview=_schedule_preview,
))


def _tasks_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    if args["action"] != "list" and not args.get("name"):
        raise ToolArgError(f"{args['action']} needs a task name")
    return [Requirement("task.schedule", op=args["action"], target=args.get("name"))]


def _tasks(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    s = _scheduler()
    action, name = args["action"], args.get("name")
    try:
        if action == "list":
            tasks = s.list()
            lines = [f"{t.name}: next {t.next_run or '?'}; runs {t.runs}"
                     f"{'; last: ' + t.last_status if t.last_status else ''}\n    \"{t.request[:120]}\"" for t in tasks]
            text = "\n".join(lines) or "(no scheduled tasks)"
            return ToolOutput(True, {"tasks": text, "count": len(tasks)}, f"{len(tasks)} scheduled task(s)", text)
        if action == "cancel":
            task = s.cancel(name)
            return ToolOutput(True, {"cancelled": task.name}, f"cancelled task {task.name}")
        if action == "run_now":
            s.run_now(name)
            return ToolOutput(True, {"started": name, "note": "running in the background; check its log"},
                              f"started task {name}")
        text = s.log(name)
        return ToolOutput(True, {"log": text[-3000:]}, f"log of task {name}", text)
    except ScheduleError as exc:
        return ToolOutput.fail(str(exc))


register(Tool(
    name="tasks",
    description="Manage scheduled tasks: list them, cancel one, run one now, or read its log.",
    rights=("task.schedule",),
    params={
        "action": param("string", "What to do", enum=["list", "cancel", "run_now", "log"], default="list"),
        "name": param("string", "Task name"),
    },
    requires=_tasks_requires,
    run=_tasks,
))
