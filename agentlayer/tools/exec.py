"""The run_command tool: allow-listed programs, no shell, optional sandbox."""

from __future__ import annotations

import glob
import os
import pwd
import shutil

from agentlayer import safeio
from agentlayer.broker import Requirement
from agentlayer.executor import CommandSyntaxError, run_pipeline, split_pipeline
from agentlayer.tools.base import Clip, Tool, ToolArgError, ToolContext, ToolOutput, fmt_bytes, param, register

SAFE_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin"


def _cwd(args: dict, ctx: ToolContext) -> str:
    cwd = ctx.real(args["cwd"])
    if not os.path.isdir(cwd):
        raise ToolArgError(f"working directory {ctx.show(cwd)} does not exist")
    return cwd


def parse(args: dict, ctx: ToolContext) -> tuple[str, list[list[str]]]:
    """Tokenise and glob-expand the command exactly as it will be executed."""
    cwd = _cwd(args, ctx)
    try:
        stages = split_pipeline(args["command"])
    except CommandSyntaxError as exc:
        raise ToolArgError(str(exc)) from exc
    argvs = []
    for stage in stages:
        argv: list[str] = []
        for word, globbable in stage:
            if globbable and any(ch in word for ch in "*?[") and len(argv) > 0:
                matches = sorted(glob.glob(word, root_dir=cwd))
                argv.extend(matches or [word])
            else:
                argv.append(word)
        argvs.append(argv)
    return cwd, argvs


def _requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    cwd, argvs = parse(args, ctx)
    return [Requirement("exec", program=argv[0], argv=argv, cwd=cwd, network=args["network"]) for argv in argvs]


def _sandbox_view(ctx: ToolContext, grant) -> tuple[list[tuple[str, bool]], list[str], list[str]]:
    """Extra directories to mount, denied paths to mask, and notes for the model."""
    mounts: list[tuple[str, bool]] = []
    masks: list[str] = []
    notes: list[str] = []
    roots = [(ctx.workspace, True)] + [(d, ctx.broker.writable(d)) for d in grant.extra_dirs if os.path.isdir(d)]
    for root, writable in roots:
        denied = ctx.broker.denied_under(root)
        if denied is None:
            notes.append(f"{ctx.show(root)} is too large to vet for secret files, so it is not mounted")
            continue
        masks += denied
        if root != ctx.workspace:
            mounts.append((root, writable))
    return mounts, masks, notes


def _child_env(ctx: ToolContext) -> dict[str, str]:
    """A minimal environment: secrets in the agent's own environment never leak."""
    try:
        user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        user = str(os.getuid())
    return {
        "PATH": SAFE_PATH,
        "HOME": ctx.workspace,
        "USER": user,
        "LOGNAME": user,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TERM": "dumb",
        "TMPDIR": "/tmp",
        "PAGER": "cat",
        "GIT_PAGER": "cat",
        "NO_COLOR": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _run(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    grant = decision.grant("exec")
    # Execute exactly what the broker approved: no second parse, no second glob expansion.
    argvs = [check.requirement.argv for check in decision.checks]
    cwd = decision.approved().cwd
    network = bool(args.get("network")) and grant.network != "never"
    mounts, masks, notes = _sandbox_view(ctx, grant) if ctx.sandbox.active else ([], [], [])
    stages = []
    for argv in argvs:
        if "/" in argv[0]:
            program = os.path.realpath(os.path.join(cwd, argv[0]))  # vetted by the broker
        else:
            program = shutil.which(argv[0], path=SAFE_PATH)
        if program is None:
            return ToolOutput.fail(f"'{argv[0]}' is allowed but not installed on this system")
        stages.append(ctx.sandbox.wrap(program, argv, cwd=cwd, workspace=ctx.workspace, network=network,
                                       masks=masks, mounts=mounts))
    timeout = min(args.get("timeout") or grant.timeout, grant.max_timeout)
    cwd_fd = safeio.open_dir(cwd)
    try:
        result = run_pipeline(stages, cwd=cwd_fd, env=_child_env(ctx), timeout=timeout,
                              max_output=grant.max_output, limits=grant.limits, on_output=ctx.stream)
    finally:
        os.close(cwd_fd)

    budget = max(1000, ctx.policy.max_tool_output - 600)
    stdout = Clip.of(result.stdout, budget)
    stderr = Clip.of(result.stderr, max(400, budget // 4))
    data: dict = {
        "command": args["command"],
        "exit_code": result.exit_code,
        "status": result.status,
        "stdout": stdout.text,
    }
    if result.stderr:
        data["stderr"] = stderr.text
    if result.truncated or stdout.truncated:
        data["output_truncated"] = f"stdout was {fmt_bytes(result.stdout_bytes)}; only the start is shown"
    data["resources"] = (f"{result.duration:.2f}s wall, {result.user_cpu:.2f}s user + {result.system_cpu:.2f}s system CPU, "
                         f"peak RSS {fmt_bytes(result.max_rss_kb * 1024)}, "
                         f"{result.voluntary_switches}+{result.involuntary_switches} context switches")
    data["sandbox"] = ctx.sandbox.mode + (" + internet" if network else "")
    if notes:
        data["notes"] = notes
    if len(result.stages) > 1:
        data["pipeline"] = [f"{' '.join(s.argv)[:60]}: {s.describe()}" for s in result.stages]
    lines = result.stdout.count("\n") + (1 if result.stdout and not result.stdout.endswith("\n") else 0)
    summary = f"{result.status} · {result.duration:.2f}s · {lines} line(s) of output"
    display = result.stdout + (("\n" if result.stdout else "") + result.stderr if result.stderr else "")
    return ToolOutput(result.exit_code == 0, data, summary, display.rstrip("\n") or None)


register(Tool(
    name="run_command",
    description="Run an allow-listed program without a shell. Programs built in the workspace run as ./name. "
                "Pipes and globs work; redirection, ;, &&, $VARS do not (use write_file to create files). "
                "No internet unless network=true.",
    rights=("exec",),
    params={
        "command": param("string", "Command line, e.g. 'grep -n TODO *.py | head -20'"),
        "cwd": param("string", "Working directory (workspace by default)", default="."),
        "timeout": param("integer", "Time limit in SECONDS (default 30)", minimum=1, maximum=600),
        "network": param("boolean", "Give the command internet access (git clone, pip, curl)", default=False),
    },
    required=("command",),
    requires=_requires,
    run=_run,
    preview=lambda a, c: f"$ {a['command']}\n(in {c.show(_cwd(a, c))}, sandbox: {c.sandbox.mode}"
                         f"{', WITH INTERNET ACCESS' if a.get('network') else ''})",
))
