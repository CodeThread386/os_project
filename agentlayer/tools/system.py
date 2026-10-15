"""System overview from uname(2), statvfs(2) and the kernel counters in /proc."""

from __future__ import annotations

import os
import socket

from agentlayer import procfs
from agentlayer.broker import Requirement
from agentlayer.tools.base import Tool, ToolContext, ToolOutput, fmt_bytes, fmt_duration, register


def os_name() -> str:
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return "Linux"


def _disk(path: str) -> str:
    vfs = os.statvfs(path)
    total = vfs.f_blocks * vfs.f_frsize
    used = total - vfs.f_bfree * vfs.f_frsize
    return f"{fmt_bytes(used)} used of {fmt_bytes(total)} ({used / total * 100 if total else 0:.0f}%)"


def _system_info(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    uts = os.uname()
    mem = procfs.meminfo()
    k = procfs.kernel_stats()
    load = os.getloadavg()
    total, avail = mem.get("MemTotal", 0), mem.get("MemAvailable", 0)
    swap_total, swap_free = mem.get("SwapTotal", 0), mem.get("SwapFree", 0)
    data = {
        "host": socket.gethostname(),
        "os": os_name(),
        "kernel": f"{uts.sysname} {uts.release} ({uts.machine})",
        "uptime": fmt_duration(procfs.uptime_seconds()),
        "cpu": f"{procfs.cpu_model()}, {os.cpu_count()} logical CPUs",
        "load_average_1_5_15min": [round(x, 2) for x in load],
        "memory": f"{fmt_bytes(total - avail)} used of {fmt_bytes(total)} ({(total - avail) / total * 100 if total else 0:.0f}%), "
                  f"{fmt_bytes(avail)} available",
        "swap": f"{fmt_bytes(swap_total - swap_free)} used of {fmt_bytes(swap_total)}",
        "processes": {
            "total": len(procfs.list_pids()),
            "running_now": k.get("procs_running"),
            "blocked_on_io": k.get("procs_blocked"),
            "created_since_boot": k.get("processes"),
            "context_switches_since_boot": k.get("ctxt"),
        },
        "disk_root": _disk("/"),
        "disk_workspace": _disk(ctx.workspace),
        "agent_user": f"{procfs.username(os.getuid())} (uid {os.getuid()})",
    }
    summary = f"{data['os']}, {data['kernel']}, up {data['uptime']}, load {load[0]:.2f}"
    return ToolOutput(True, data, summary)


register(Tool(
    name="system_info",
    description="Overview of this machine: OS, kernel, uptime, CPU, load, memory, process counters, disk usage.",
    rights=("sys.read",),
    params={},
    requires=lambda a, c: [Requirement("sys.read")],
    run=_system_info,
))

