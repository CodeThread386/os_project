"""Process tools: the kernel's process table, signals and scheduling priority.

Listing and inspection read /proc (the kernel's view of each process
control block). Control goes through kill(2) and setpriority(2), so the
kernel still applies its own permission checks after the broker's.
"""

from __future__ import annotations

import os
import signal
import time

from agentlayer import procfs
from agentlayer.broker import Requirement
from agentlayer.tools.base import (
    Tool, ToolArgError, ToolContext, ToolOutput, fmt_bytes, fmt_duration, fmt_time, param, register,
)
from agentlayer.policy import SIGNAL_NAMES


def _sample(pids: list[int]) -> dict[int, procfs.ProcStat]:
    out = {}
    for pid in pids:
        st = procfs.read_stat(pid)
        if st is not None:
            out[pid] = st
    return out


def _list_processes(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    grant = decision.grant("proc.read")
    own_only = grant is not None and grant.scope == "own"
    me = os.getuid()
    mem_total = procfs.meminfo().get("MemTotal", 1)

    pids = procfs.list_pids()
    first = _sample(pids)
    interval = 0.4
    t0 = time.monotonic()
    time.sleep(interval)  # CPU% is a rate: sample twice, like top(1)
    second = _sample(pids)
    elapsed = time.monotonic() - t0

    name_filter = (args.get("name") or "").lower()
    rows = []
    states: dict[str, int] = {}
    for pid, st in second.items():
        uid = procfs.process_owner(pid)
        if own_only and uid != me:
            continue
        states[st.state_name] = states.get(st.state_name, 0) + 1
        cmd = procfs.read_cmdline(pid)
        command = " ".join(cmd) if cmd else f"[{st.name}]"
        if name_filter and name_filter not in st.name.lower() and name_filter not in command.lower():
            continue
        if args.get("user") and procfs.username(uid if uid is not None else -1) != args["user"]:
            continue
        prev = first.get(pid)
        cpu = (st.cpu_ticks - prev.cpu_ticks) / procfs.CLK_TCK / elapsed * 100 if prev else 0.0
        rows.append({
            "pid": pid, "ppid": st.ppid, "user": procfs.username(uid) if uid is not None else "?",
            "state": st.state, "cpu": cpu, "mem": st.rss_bytes / mem_total * 100, "rss": st.rss_bytes,
            "threads": st.num_threads, "nice": st.nice, "name": st.name, "command": command,
        })

    key = {"cpu": lambda r: -r["cpu"], "memory": lambda r: -r["rss"], "pid": lambda r: r["pid"],
           "name": lambda r: r["name"].lower()}[args["sort_by"]]
    rows.sort(key=key)
    shown = rows[: args["limit"]]
    header = f"{'PID':>7} {'PPID':>7} {'USER':<10} S {'%CPU':>5} {'%MEM':>5} {'RSS':>9} {'THR':>4} {'NI':>3}  COMMAND"
    lines = [header] + [
        f"{r['pid']:>7} {r['ppid']:>7} {r['user'][:10]:<10} {r['state']} {r['cpu']:5.1f} {r['mem']:5.1f} "
        f"{fmt_bytes(r['rss']):>9} {r['threads']:>4} {r['nice']:>3}  {r['command'][:70]}"
        for r in shown
    ]
    table = "\n".join(lines)
    scope = "your processes" if own_only else "all processes"
    data = {"scope": scope, "matching": len(rows), "states": states, "sorted_by": args["sort_by"], "table": table}
    return ToolOutput(True, data, f"{len(rows)} matching processes ({scope}), showing {len(shown)}", table)


register(Tool(
    name="list_processes",
    description="List running processes from /proc with CPU%, memory, state and command (like ps/top).",
    rights=("proc.read",),
    params={
        "sort_by": param("string", "Sort order", enum=["cpu", "memory", "pid", "name"], default="cpu"),
        "limit": param("integer", "Rows to return", default=15, minimum=1, maximum=60),
        "name": param("string", "Only processes whose name or command line contains this text"),
        "user": param("string", "Only processes owned by this user name"),
    },
    requires=lambda a, c: [Requirement("proc.read")],
    run=_list_processes,
))


def _affinity(pid: int) -> str:
    try:
        cpus = sorted(os.sched_getaffinity(pid))
    except OSError:
        return "unavailable"
    ranges, start = [], None
    for i, cpu in enumerate(cpus):
        if start is None:
            start = cpu
        if i + 1 == len(cpus) or cpus[i + 1] != cpu + 1:
            ranges.append(f"{start}-{cpu}" if cpu != start else str(cpu))
            start = None
    return ",".join(ranges)


def _readlink(path: str) -> str:
    try:
        return os.readlink(path)
    except OSError as exc:
        return f"unavailable ({exc.strerror.lower()})"


def _process_info(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    pid = args["pid"]
    st = procfs.read_stat(pid)
    if st is None:
        return ToolOutput.fail(f"PID {pid} no longer exists")
    status = procfs.read_status(pid)
    cmd = procfs.read_cmdline(pid)
    uid = procfs.real_uid(pid)
    started = procfs.boot_time() + st.starttime / procfs.CLK_TCK
    try:
        fds: int | str = len(os.listdir(f"/proc/{pid}/fd"))
    except OSError as exc:
        fds = f"unavailable ({exc.strerror.lower()})"
    parent = procfs.read_stat(st.ppid)
    children = []
    for other in procfs.list_pids():
        o = procfs.read_stat(other)
        if o is not None and o.ppid == pid:
            children.append(f"{other} ({o.name})")
    pcb = {
        "identification": {
            "pid": pid, "parent": f"{st.ppid} ({parent.name})" if parent else st.ppid,
            "process_group": st.pgrp, "session": st.session,
            "user": f"{procfs.username(uid)} (uid {uid})" if uid is not None else "?",
            "name": st.name, "command_line": " ".join(cmd) or f"[{st.name}] (kernel thread)",
            "executable": _readlink(f"/proc/{pid}/exe"),
        },
        "state": f"{st.state} ({st.state_name})",
        "scheduling": {
            "policy": procfs.SCHED_POLICIES.get(st.policy, str(st.policy)),
            "nice": st.nice, "kernel_priority": st.priority, "realtime_priority": st.rt_priority,
            "last_ran_on_cpu": st.processor, "allowed_cpus": _affinity(pid),
            "voluntary_context_switches": status.get("voluntary_ctxt_switches"),
            "involuntary_context_switches": status.get("nonvoluntary_ctxt_switches"),
        },
        "memory": {
            "virtual": fmt_bytes(st.vsize), "resident": fmt_bytes(st.rss_bytes),
            "peak_resident": status.get("VmHWM", "?"), "swapped": status.get("VmSwap", "0 kB"),
            "minor_faults": st.minflt, "major_faults": st.majflt,
        },
        "accounting": {
            "user_cpu_seconds": round(st.utime / procfs.CLK_TCK, 2),
            "system_cpu_seconds": round(st.stime / procfs.CLK_TCK, 2),
            "started": fmt_time(started), "running_for": fmt_duration(time.time() - started),
        },
        "resources": {"threads": st.num_threads, "open_file_descriptors": fds,
                      "working_directory": _readlink(f"/proc/{pid}/cwd")},
        "children": children[:20] or "none",
    }
    summary = f"PID {pid} {st.name}: {st.state_name}, {fmt_bytes(st.rss_bytes)} resident, nice {st.nice}"
    return ToolOutput(True, pcb, summary)


register(Tool(
    name="process_info",
    description="Show one process's control block from /proc: identity, state, scheduling, memory, CPU accounting, children.",
    rights=("proc.read",),
    params={"pid": param("integer", "Process ID", minimum=1)},
    required=("pid",),
    requires=lambda a, c: [Requirement("proc.read", pid=a["pid"])],
    run=_process_info,
))


def _signal_name(raw: str) -> str:
    text = str(raw).strip().upper()
    if text.isdigit():
        try:
            text = signal.Signals(int(text)).name
        except ValueError as exc:
            raise ToolArgError(f"unknown signal number {raw}") from exc
    if not text.startswith("SIG"):
        text = "SIG" + text
    if text not in SIGNAL_NAMES:
        raise ToolArgError(f"unsupported signal {raw}; use one of {', '.join(SIGNAL_NAMES)}")
    return text


def _signal_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    return [Requirement("proc.signal", pid=args["pid"], signal=_signal_name(args["signal"]))]


def _describe_process(pid: int) -> str:
    st = procfs.read_stat(pid)
    if st is None:
        return f"PID {pid} (not running)"
    cmd = " ".join(procfs.read_cmdline(pid)) or st.name
    uid = procfs.real_uid(pid)
    started = procfs.boot_time() + st.starttime / procfs.CLK_TCK
    return (f"PID {pid}  {cmd[:100]}\nuser {procfs.username(uid) if uid is not None else '?'} · state {st.state_name} · "
            f"{fmt_bytes(st.rss_bytes)} resident · running for {fmt_duration(time.time() - started)}")


def _signal(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    pid, name = args["pid"], _signal_name(args["signal"])
    before = procfs.read_stat(pid)
    os.kill(pid, signal.Signals[name])
    outcome = "signal delivered"
    if name in ("SIGTERM", "SIGKILL", "SIGINT", "SIGHUP", "SIGQUIT"):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            st = procfs.read_stat(pid)
            if st is None:
                outcome = "process terminated and was reaped"
                break
            if st.state == "Z":
                outcome = "process terminated (zombie: it exited and is waiting for its parent to call wait())"
                break
            time.sleep(0.05)
        else:
            outcome = "process is still running (it may be handling or ignoring the signal)"
    elif name in ("SIGSTOP", "SIGCONT"):
        time.sleep(0.1)
        st = procfs.read_stat(pid)
        outcome = f"process is now {st.state_name}" if st else "process is gone"
    data = {"pid": pid, "name": before.name if before else "?", "signal": name, "outcome": outcome}
    return ToolOutput(True, data, f"sent {name} to PID {pid}: {outcome}")


register(Tool(
    name="signal_process",
    description="Send a signal to a process with kill(2): SIGTERM asks it to exit, SIGKILL forces it, SIGSTOP/SIGCONT pause and resume.",
    rights=("proc.signal",),
    params={
        "pid": param("integer", "Process ID", minimum=1),
        "signal": param("string", "Signal name", default="SIGTERM"),
    },
    required=("pid",),
    requires=_signal_requires,
    run=_signal,
    preview=lambda a, c: _describe_process(a["pid"]),
    destructive=True,
))


def _renice(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    pid, nice = args["pid"], args["nice"]
    old = os.getpriority(os.PRIO_PROCESS, pid)
    try:
        os.setpriority(os.PRIO_PROCESS, pid, nice)
    except PermissionError:
        return ToolOutput.fail(
            f"the kernel refused to change nice from {old} to {nice}: lowering a nice value (raising priority) "
            "needs CAP_SYS_NICE; unprivileged users can only make processes nicer", pid=pid, nice=old)
    new = os.getpriority(os.PRIO_PROCESS, pid)
    st = procfs.read_stat(pid)
    data = {"pid": pid, "name": st.name if st else "?", "old_nice": old, "new_nice": new,
            "policy": procfs.SCHED_POLICIES.get(st.policy if st else 0, "?")}
    return ToolOutput(True, data, f"PID {pid} nice {old} -> {new}")


register(Tool(
    name="set_priority",
    description="Change a process's nice value with setpriority(2) (-20 highest priority … 19 lowest). "
                "Without root you can only increase it.",
    rights=("proc.priority",),
    params={
        "pid": param("integer", "Process ID", minimum=1),
        "nice": param("integer", "New nice value", minimum=-20, maximum=19),
    },
    required=("pid", "nice"),
    requires=lambda a, c: [Requirement("proc.priority", pid=a["pid"])],
    run=_renice,
    preview=lambda a, c: _describe_process(a["pid"]) + f"\nnew nice value: {a['nice']}",
))
