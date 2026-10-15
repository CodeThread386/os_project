"""Live system metrics from /proc and /sys, sampled the way top(1) does it.

Rates (CPU %, disk and network throughput, per-process CPU) need two
samples: the kernel exposes monotonically increasing counters, and a rate is
the difference between two readings divided by the time between them.
"""

from __future__ import annotations

import os
import time

from agentlayer import procfs


def _cpu_times() -> dict[str, tuple[int, int]]:
    """{'cpu': (busy, total), 'cpu0': ...} in clock ticks from /proc/stat."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                break
            name, *values = line.split()
            v = [int(x) for x in values[:8]]
            idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
            total = sum(v)
            out[name] = (total - idle, total)
    return out


def _net_bytes() -> tuple[int, int]:
    rx = tx = 0
    with open("/proc/net/dev") as f:
        for line in f.readlines()[2:]:
            name, data = line.split(":", 1)
            if name.strip() == "lo":
                continue
            fields = data.split()
            rx += int(fields[0])
            tx += int(fields[8])
    return rx, tx


def _disk_bytes() -> tuple[int, int]:
    read = written = 0
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            name = p[2]
            if name.startswith(("loop", "ram", "zram", "dm-", "sr")) or (name[-1].isdigit() and not name.startswith("nvme")):
                continue  # partitions and virtual devices would double count
            if name.startswith("nvme") and "p" in name[4:]:
                continue
            read += int(p[5]) * 512
            written += int(p[9]) * 512
    return read, written


def _temperature() -> float | None:
    """CPU package temperature if the platform names it, otherwise the hottest thermal zone."""
    preferred, others = [], []
    base = "/sys/class/thermal"
    for zone in os.listdir(base) if os.path.isdir(base) else []:
        try:
            with open(f"{base}/{zone}/type") as f:
                kind = f.read().strip()
            with open(f"{base}/{zone}/temp") as f:
                temp = int(f.read().strip()) / 1000
        except (OSError, ValueError):
            continue
        (preferred if kind in ("x86_pkg_temp", "coretemp", "k10temp", "cpu-thermal") else others).append(temp)
    pool = preferred or others
    return max(pool) if pool else None


def _battery() -> dict | None:
    base = "/sys/class/power_supply"
    for dev in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        try:
            with open(f"{base}/{dev}/type") as f:
                if f.read().strip() != "Battery":
                    continue
            with open(f"{base}/{dev}/capacity") as f:
                level = int(f.read().strip())
            with open(f"{base}/{dev}/status") as f:
                status = f.read().strip()
            return {"level": level, "status": status}
        except (OSError, ValueError):
            continue
    return None


class Monitor:
    def __init__(self) -> None:
        self._prev: dict | None = None
        self._prev_procs: dict[int, int] = {}

    def sample(self, top: int = 8) -> dict:
        now = time.monotonic()
        cpu = _cpu_times()
        net = _net_bytes()
        disk = _disk_bytes()
        procs = {}
        states: dict[str, int] = {}
        try:  # the kernel counts runnable and I/O-blocked threads (a main thread often sleeps while others run)
            kernel = {k: int(v) for k, v in (line.split()[:2] for line in open("/proc/stat") if line.startswith("procs_"))}
        except OSError:
            kernel = {}
        for pid in procfs.list_pids():
            st = procfs.read_stat(pid)
            if st is not None:
                procs[pid] = st
                states[st.state] = states.get(st.state, 0) + 1
        mem = procfs.meminfo()
        out: dict = {
            "time": time.time(),
            "load": [round(x, 2) for x in os.getloadavg()],
            "uptime": procfs.uptime_seconds(),
            "memory": {"total": mem.get("MemTotal", 0), "used": mem.get("MemTotal", 0) - mem.get("MemAvailable", 0),
                       "cached": mem.get("Cached", 0) + mem.get("Buffers", 0),
                       "swap_total": mem.get("SwapTotal", 0), "swap_used": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)},
            "processes": {"total": len(procs), "running": kernel.get("procs_running", states.get("R", 0)),
                          "sleeping": states.get("S", 0) + states.get("I", 0),
                          "blocked": kernel.get("procs_blocked", states.get("D", 0)), "zombie": states.get("Z", 0), "stopped": states.get("T", 0)},
            "temperature": _temperature(),
            "battery": _battery(),
            "cpu": {"total": 0.0, "cores": []}, "net": {"rx": 0.0, "tx": 0.0}, "disk": {"read": 0.0, "write": 0.0},
            "top_cpu": [], "top_memory": [],
        }
        prev = self._prev
        if prev is not None:
            dt = max(1e-3, now - prev["now"])
            for name, (busy, total) in cpu.items():
                pbusy, ptotal = prev["cpu"].get(name, (busy, total))
                pct = 100.0 * (busy - pbusy) / max(1, total - ptotal)
                if name == "cpu":
                    out["cpu"]["total"] = round(pct, 1)
                else:
                    out["cpu"]["cores"].append(round(pct, 1))
            out["net"] = {"rx": max(0.0, (net[0] - prev["net"][0]) / dt), "tx": max(0.0, (net[1] - prev["net"][1]) / dt)}
            out["disk"] = {"read": max(0.0, (disk[0] - prev["disk"][0]) / dt),
                           "write": max(0.0, (disk[1] - prev["disk"][1]) / dt)}
            rates = []
            for pid, st in procs.items():
                before = self._prev_procs.get(pid)
                if before is not None:
                    rates.append((100.0 * (st.cpu_ticks - before) / procfs.CLK_TCK / dt, pid, st))
            rates.sort(key=lambda r: -r[0])
            out["top_cpu"] = [{"pid": pid, "name": st.name, "cpu": round(rate, 1), "rss": st.rss_bytes,
                               "state": st.state} for rate, pid, st in rates[:top]]
        by_mem = sorted(procs.items(), key=lambda kv: -kv[1].rss_bytes)[:top]
        out["top_memory"] = [{"pid": pid, "name": st.name, "rss": st.rss_bytes, "state": st.state} for pid, st in by_mem]
        self._prev = {"now": now, "cpu": cpu, "net": net, "disk": disk}
        self._prev_procs = {pid: st.cpu_ticks for pid, st in procs.items()}
        return out
