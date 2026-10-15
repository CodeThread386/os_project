"""Read-only view of the kernel's process table through /proc.

Every process has a process control block (``task_struct``) inside the
kernel. Linux exposes a text view of it under ``/proc/<pid>/``; this module
parses those files directly, without third-party libraries, so the agent's
process tools show exactly what the kernel reports.
"""

from __future__ import annotations

import os
import pwd
from dataclasses import dataclass
from functools import lru_cache

CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")

STATE_NAMES = {
    "R": "running",
    "S": "sleeping",
    "D": "uninterruptible sleep (I/O)",
    "Z": "zombie",
    "T": "stopped",
    "t": "stopped by debugger",
    "X": "dead",
    "I": "idle kernel thread",
    "P": "parked",
    "W": "paging",
    "K": "wakekill",
}

SCHED_POLICIES = {
    0: "SCHED_OTHER (CFS/EEVDF, time-sharing)",
    1: "SCHED_FIFO (real-time)",
    2: "SCHED_RR (real-time round-robin)",
    3: "SCHED_BATCH",
    5: "SCHED_IDLE",
    6: "SCHED_DEADLINE",
}


@dataclass
class ProcStat:
    """Fields of /proc/<pid>/stat (see proc(5))."""

    pid: int
    name: str
    state: str
    ppid: int
    pgrp: int
    session: int
    tty_nr: int
    minflt: int
    majflt: int
    utime: int  # clock ticks spent in user mode
    stime: int  # clock ticks spent in kernel mode
    priority: int
    nice: int
    num_threads: int
    starttime: int  # clock ticks after boot
    vsize: int  # bytes
    rss_pages: int
    processor: int
    rt_priority: int
    policy: int

    @property
    def cpu_ticks(self) -> int:
        return self.utime + self.stime

    @property
    def rss_bytes(self) -> int:
        return self.rss_pages * PAGE_SIZE

    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(self.state, self.state)


def read_stat(pid: int) -> ProcStat | None:
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read().decode(errors="replace")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    # The command name is in parentheses and may itself contain spaces or ')'.
    left, right = data.index("("), data.rindex(")")
    f = data[right + 2 :].split()
    try:
        return ProcStat(
            pid=int(data[:left]),
            name=data[left + 1 : right],
            state=f[0],
            ppid=int(f[1]),
            pgrp=int(f[2]),
            session=int(f[3]),
            tty_nr=int(f[4]),
            minflt=int(f[7]),
            majflt=int(f[9]),
            utime=int(f[11]),
            stime=int(f[12]),
            priority=int(f[15]),
            nice=int(f[16]),
            num_threads=int(f[17]),
            starttime=int(f[19]),
            vsize=int(f[20]),
            rss_pages=int(f[21]),
            processor=int(f[36]) if len(f) > 36 else -1,
            rt_priority=int(f[37]) if len(f) > 37 else 0,
            policy=int(f[38]) if len(f) > 38 else 0,
        )
    except (IndexError, ValueError):
        return None


def read_status(pid: int) -> dict[str, str]:
    """Key/value pairs from /proc/<pid>/status (empty if the process is gone)."""
    out: dict[str, str] = {}
    try:
        with open(f"/proc/{pid}/status", "rb") as f:
            for raw in f:
                key, _, value = raw.decode(errors="replace").partition(":")
                out[key.strip()] = value.strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return out


def read_cmdline(pid: int) -> list[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def real_uid(pid: int) -> int | None:
    uid = read_status(pid).get("Uid")
    return int(uid.split()[0]) if uid else None


@lru_cache(maxsize=512)
def username(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def list_pids() -> list[int]:
    return sorted(int(name) for name in os.listdir("/proc") if name.isdigit())


def process_owner(pid: int) -> int | None:
    """Effective UID of a process (the owner of its /proc directory)."""
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except (FileNotFoundError, ProcessLookupError):
        return None


def lineage(pid: int) -> list[int]:
    """The process and all of its ancestors, nearest first (stops before PID 1)."""
    chain: list[int] = []
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        chain.append(pid)
        stat = read_stat(pid)
        if stat is None:
            break
        pid = stat.ppid
    return chain


def kernel_stats() -> dict[str, int]:
    """Counters from /proc/stat: boot time, forks, context switches, run queue."""
    out: dict[str, int] = {}
    with open("/proc/stat") as f:
        for line in f:
            key, _, rest = line.partition(" ")
            if key in ("btime", "processes", "ctxt", "procs_running", "procs_blocked"):
                out[key] = int(rest.split()[0])
    return out


def boot_time() -> float:
    return float(kernel_stats()["btime"])


def meminfo() -> dict[str, int]:
    """/proc/meminfo in bytes."""
    out: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            parts = rest.split()
            if parts:
                out[key] = int(parts[0]) * (1024 if len(parts) > 1 else 1)
    return out


def uptime_seconds() -> float:
    with open("/proc/uptime") as f:
        return float(f.read().split()[0])


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return os.uname().machine
