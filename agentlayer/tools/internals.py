"""OS-internals tools: look inside processes, the kernel, the hardware and the services.

Everything here is read-only and comes straight from the kernel's own
interfaces: procfs (/proc), sysfs (/sys), inotify, ptrace (through strace),
and the read-only queries of systemd.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess

from agentlayer import inotify, procfs
from agentlayer.broker import Requirement
from agentlayer.executor import run_pipeline
from agentlayer.tools.base import Clip, Tool, ToolArgError, ToolContext, ToolOutput, fmt_bytes, param, register

PROCESS_TOPICS = ["memory_map", "open_files", "limits", "namespaces", "cgroup", "threads", "scheduler", "io"]
SYSTEM_TOPICS = ["interrupts", "vmstat", "meminfo", "pressure", "buddyinfo", "mounts", "modules", "swaps", "cpus",
                 "load", "locks", "filesystems"]


def _read(path: str, limit: int = 2_000_000) -> str:
    with open(path, "rb") as f:
        return f.read(limit).decode(errors="replace")


def _table(rows: list[list], header: list[str]) -> str:
    cols = [header] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cols) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in cols)


def _kb(text: str) -> int:
    parts = text.split()
    return int(parts[0]) * 1024 if parts and parts[0].isdigit() else 0


# --------------------------------------------------------- per-process views


def _memory_map(pid: int) -> tuple[dict, str]:
    exe = os.path.realpath(f"/proc/{pid}/exe") if os.access(f"/proc/{pid}/exe", os.F_OK) else ""
    regions, cats = [], {}
    for line in _read(f"/proc/{pid}/maps").splitlines():
        parts = line.split(maxsplit=5)
        start, end = (int(x, 16) for x in parts[0].split("-"))
        perms, name = parts[1], parts[5] if len(parts) > 5 else ""
        if not name:
            cat = "anonymous memory (malloc arenas, thread stacks)"
        elif name.startswith("["):
            cat = {"[heap]": "heap (brk)", "[stack]": "main thread stack", "[vdso]": "vDSO (kernel code for fast syscalls)",
                   "[vvar]": "vvar (kernel data for vDSO)", "[vsyscall]": "legacy vsyscall page"}.get(name, name)
        elif name == exe:
            cat = "program image"
        elif ".so" in os.path.basename(name):
            cat = "shared libraries"
        else:
            cat = "memory-mapped files"
        c = cats.setdefault(cat, {"regions": 0, "size": 0, "executable": 0, "writable": 0})
        c["regions"] += 1
        c["size"] += end - start
        c["executable"] += "x" in perms
        c["writable"] += "w" in perms
        regions.append((parts[0], perms, fmt_bytes(end - start), name[-60:]))
    rollup = {}
    try:
        for line in _read(f"/proc/{pid}/smaps_rollup").splitlines()[1:]:
            key, _, value = line.partition(":")
            if key in ("Rss", "Pss", "Shared_Clean", "Shared_Dirty", "Private_Clean", "Private_Dirty", "Swap", "Anonymous"):
                rollup[key] = fmt_bytes(_kb(value))
    except OSError:
        pass
    summary = {cat: f"{v['regions']} regions, {fmt_bytes(v['size'])} virtual" for cat, v in
               sorted(cats.items(), key=lambda kv: -kv[1]["size"])}
    data = {"layout": summary, "resident": rollup, "regions": len(regions),
            "note": "virtual size is address space reserved; resident (Rss/Pss) is RAM actually used"}
    shown = _table([list(r) for r in regions[:45]], ["address range", "perms", "size", "mapping"])
    return data, shown


_FD_FLAGS = {0o1: "O_WRONLY", 0o2: "O_RDWR", 0o2000: "O_APPEND", 0o4000: "O_NONBLOCK", 0o2000000: "O_CLOEXEC",
             0o40000: "O_DIRECT"}


def _open_files(pid: int) -> tuple[dict, str]:
    rows, kinds = [], {}
    fds = sorted(os.listdir(f"/proc/{pid}/fd"), key=int)
    for fd in fds:
        try:
            target = os.readlink(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        kind = ("socket" if target.startswith("socket:") else "pipe" if target.startswith("pipe:") else
                "kernel object" if target.startswith("anon_inode:") else "device" if target.startswith("/dev/") else "file")
        kinds[kind] = kinds.get(kind, 0) + 1
        flags = ""
        try:
            info = dict(line.split(":\t", 1) for line in _read(f"/proc/{pid}/fdinfo/{fd}").splitlines() if ":\t" in line)
            raw = int(info.get("flags", "0").strip(), 8)
            flags = "|".join(n for bit, n in _FD_FLAGS.items() if raw & bit) or "O_RDONLY"
        except (OSError, ValueError):
            pass
        rows.append((fd, kind, flags, target[-70:]))
    data = {"descriptors": len(fds), "by_kind": kinds,
            "note": "0, 1, 2 are stdin, stdout, stderr; sockets and pipes show their kernel inode numbers"}
    return data, _table([list(r) for r in rows[:60]], ["fd", "kind", "flags", "points to"])


def _limits(pid: int) -> tuple[dict, str]:
    rows = []
    for line in _read(f"/proc/{pid}/limits").splitlines()[1:]:
        m = re.match(r"(.+?)\s{2,}(\S+)\s+(\S+)\s*(\S*)", line)
        if m:
            rows.append(list(m.groups()))
    return {"limits": {r[0]: f"soft {r[1]}, hard {r[2]} {r[3]}".strip() for r in rows}}, \
        _table(rows, ["resource", "soft", "hard", "units"])


def _namespaces(pid: int) -> tuple[dict, str]:
    rows, data = [], {}
    for ns in sorted(os.listdir(f"/proc/{pid}/ns")):
        try:
            mine = os.readlink(f"/proc/{pid}/ns/{ns}")
        except OSError:
            continue
        try:
            ours = os.readlink(f"/proc/self/ns/{ns}")
        except OSError:
            ours = ""
        same = "shared with AgentLayer" if mine == ours else "SEPARATE (isolated)"
        data[ns] = f"{mine.split(':', 1)[1]} {same}"
        rows.append((ns, mine.split(":", 1)[1], same))
    note = ("a process in different namespaces sees its own mounts, PIDs, network, hostname, IPC or users; "
            "containers and the AgentLayer sandbox are built from these")
    return {"namespaces": data, "note": note}, _table([list(r) for r in rows], ["namespace", "id", "relation"])


def _cgroup(pid: int) -> tuple[dict, str]:
    path = _read(f"/proc/{pid}/cgroup").strip().splitlines()[-1].split("::", 1)[-1]
    base = f"/sys/fs/cgroup{path}"
    stats = {"cgroup": path}
    for name in ("memory.current", "memory.max", "memory.peak", "pids.current", "pids.max", "cpu.weight"):
        try:
            value = _read(f"{base}/{name}").strip()
            stats[name] = fmt_bytes(int(value)) if value.isdigit() and name.startswith("memory") else value
        except OSError:
            continue
    try:
        cpu = dict(line.split() for line in _read(f"{base}/cpu.stat").splitlines())
        stats["cpu_time"] = f"{int(cpu.get('usage_usec', 0)) / 1e6:.1f}s (user {int(cpu.get('user_usec', 0)) / 1e6:.1f}s)"
    except (OSError, ValueError):
        pass
    return stats, "\n".join(f"{k}: {v}" for k, v in stats.items())


def _threads(pid: int) -> tuple[dict, str]:
    rows = []
    for tid in sorted(os.listdir(f"/proc/{pid}/task"), key=int):
        try:
            raw = _read(f"/proc/{pid}/task/{tid}/stat")
            name = raw[raw.index("(") + 1: raw.rindex(")")]
            f = raw[raw.rindex(")") + 2:].split()
            rows.append((tid, name, f"{f[0]} ({procfs.STATE_NAMES.get(f[0], '?')})",
                         f"{(int(f[11]) + int(f[12])) / procfs.CLK_TCK:.2f}s", f[36] if len(f) > 36 else "?"))
        except (OSError, ValueError, IndexError):
            continue
    return {"threads": len(rows), "note": "threads share the address space; each is scheduled separately"}, \
        _table([list(r) for r in rows[:50]], ["tid", "name", "state", "cpu time", "last cpu"])


def _scheduler(pid: int) -> tuple[dict, str]:
    stat = procfs.read_stat(pid)
    data = {"policy": procfs.SCHED_POLICIES.get(stat.policy, str(stat.policy)) if stat else "?",
            "nice": stat.nice if stat else "?"}
    try:
        run_ns, wait_ns, slices = (int(x) for x in _read(f"/proc/{pid}/schedstat").split()[:3])
        data.update(cpu_time=f"{run_ns / 1e9:.2f}s", time_waiting_for_cpu=f"{wait_ns / 1e9:.2f}s",
                    times_scheduled=slices)
    except (OSError, ValueError):
        pass
    try:
        for line in _read(f"/proc/{pid}/sched").splitlines():
            m = re.match(r"(nr_switches|nr_voluntary_switches|nr_involuntary_switches|se\.load\.weight|prio)\s*:\s*(\S+)", line)
            if m:
                data[m.group(1)] = m.group(2)
    except OSError:
        pass
    return data, "\n".join(f"{k}: {v}" for k, v in data.items())


def _io(pid: int) -> tuple[dict, str]:
    raw = dict(line.split(": ") for line in _read(f"/proc/{pid}/io").splitlines())
    data = {"bytes_read_by_syscalls": fmt_bytes(int(raw["rchar"])), "bytes_written_by_syscalls": fmt_bytes(int(raw["wchar"])),
            "read_syscalls": raw["syscr"], "write_syscalls": raw["syscw"],
            "bytes_read_from_storage": fmt_bytes(int(raw["read_bytes"])),
            "bytes_written_to_storage": fmt_bytes(int(raw["write_bytes"])),
            "note": "the difference between syscall bytes and storage bytes is what the page cache absorbed"}
    return data, "\n".join(f"{k}: {v}" for k, v in data.items())


# ------------------------------------------------------------ system views


def _interrupts() -> tuple[dict, str]:
    lines = _read("/proc/interrupts").splitlines()
    cpus = len(lines[0].split())
    rows = []
    for line in lines[1:]:
        irq, _, rest = line.partition(":")
        parts = rest.split()
        counts = [int(p) for p in parts[:cpus] if p.isdigit()]
        rows.append((irq.strip(), sum(counts), " ".join(parts[len(counts):])[:50]))
    rows.sort(key=lambda r: -r[1])
    return {"cpus": cpus, "busiest": {r[0]: f"{r[1]:,} ({r[2]})" for r in rows[:10]},
            "note": "hardware and inter-processor interrupts handled since boot"}, \
        _table([[r[0], f"{r[1]:,}", r[2]] for r in rows[:20]], ["irq", "total", "device / type"])


_VMSTAT = {
    "pgfault": "page faults (all)", "pgmajfault": "major page faults (had to read from disk)",
    "pgpgin": "KiB paged in from disk", "pgpgout": "KiB paged out to disk", "pswpin": "pages swapped in",
    "pswpout": "pages swapped out", "pgscan_kswapd": "pages scanned by kswapd (memory reclaim)",
    "pgsteal_kswapd": "pages reclaimed by kswapd", "oom_kill": "processes killed by the OOM killer",
    "thp_fault_alloc": "transparent huge pages allocated", "workingset_refault_file": "evicted pages needed again",
}


def _vmstat() -> tuple[dict, str]:
    raw = dict(line.split() for line in _read("/proc/vmstat").splitlines())
    data = {f"{k} ({v})": f"{int(raw[k]):,}" for k, v in _VMSTAT.items() if k in raw}
    return data, "\n".join(f"{k}: {v}" for k, v in data.items())


def _meminfo() -> tuple[dict, str]:
    info = procfs.meminfo()
    keys = ["MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapCached", "Active", "Inactive",
            "Dirty", "Writeback", "AnonPages", "Mapped", "Shmem", "Slab", "SReclaimable", "KernelStack",
            "PageTables", "CommitLimit", "Committed_AS", "SwapTotal", "SwapFree", "HugePages_Total", "Hugepagesize"]
    data = {k: fmt_bytes(info[k]) if k != "HugePages_Total" else info[k] for k in keys if k in info}
    data["note"] = "Cached/Buffers are reclaimable page cache; MemAvailable is what new programs can really get"
    return data, "\n".join(f"{k}: {v}" for k, v in data.items())


def _pressure() -> tuple[dict, str]:
    data = {}
    for res in ("cpu", "memory", "io"):
        try:
            for line in _read(f"/proc/pressure/{res}").splitlines():
                kind, *fields = line.split()
                vals = dict(f.split("=") for f in fields)
                data[f"{res} {kind}"] = f"{vals['avg10']}% (10s) {vals['avg60']}% (1m) {vals['avg300']}% (5m)"
        except OSError:
            continue
    data["note"] = "PSI: share of time tasks were stalled waiting for this resource ('some' = at least one task)"
    return data, "\n".join(f"{k}: {v}" for k, v in data.items())


def _buddyinfo() -> tuple[dict, str]:
    rows, data = [], {}
    for line in _read("/proc/buddyinfo").splitlines():
        parts = line.split()
        zone, counts = f"node {parts[1].rstrip(',')} {parts[3]}", [int(x) for x in parts[4:]]
        free = sum(c * (2 ** order) * 4096 for order, c in enumerate(counts))
        data[zone] = f"{fmt_bytes(free)} free; blocks per order 0..{len(counts) - 1}: {' '.join(map(str, counts))}"
        rows.append([zone] + counts)
    note = ("the buddy allocator keeps free memory in blocks of 2^order pages (4 KiB .. 4 MiB); many high-order "
            "blocks mean low fragmentation")
    return {"zones": data, "note": note}, _table(rows, ["zone"] + [f"o{i}" for i in range(len(rows[0]) - 1)])


def _mounts() -> tuple[dict, str]:
    rows, types = [], {}
    for line in _read("/proc/self/mountinfo").splitlines():
        left, _, right = line.partition(" - ")
        mount_point, options = left.split()[4], left.split()[5]
        fstype, source = right.split()[0], right.split()[1]
        types[fstype] = types.get(fstype, 0) + 1
        if fstype in ("proc", "sysfs", "cgroup2", "devpts", "mqueue", "securityfs", "pstore", "bpf", "tracefs",
                      "debugfs", "configfs", "fusectl", "hugetlbfs", "autofs", "efivarfs", "binfmt_misc", "nsfs"):
            continue
        rows.append((mount_point[:40], fstype, source[:30], options[:30]))
    return {"filesystem_types": types, "real_mounts": len(rows)}, \
        _table([list(r) for r in rows[:40]], ["mount point", "type", "source", "options"])


def _modules() -> tuple[dict, str]:
    rows = []
    for line in _read("/proc/modules").splitlines():
        parts = line.split()
        rows.append((parts[0], int(parts[1]), int(parts[2])))
    rows.sort(key=lambda r: -r[1])
    return {"loaded_modules": len(rows), "note": "kernel modules are drivers and features loaded at run time"}, \
        _table([[r[0], fmt_bytes(r[1]), r[2]] for r in rows[:25]], ["module", "size", "used by"])


def _swaps() -> tuple[dict, str]:
    text = _read("/proc/swaps")
    return {"swap_devices": max(0, len(text.splitlines()) - 1)}, text


def _cpus() -> tuple[dict, str]:
    rows = []
    base = "/sys/devices/system/cpu"
    for name in sorted((n for n in os.listdir(base) if re.fullmatch(r"cpu\d+", n)), key=lambda n: int(n[3:])):
        def rd(rel: str) -> str:
            try:
                return _read(f"{base}/{name}/{rel}").strip()
            except OSError:
                return "?"
        cur = rd("cpufreq/scaling_cur_freq")
        rows.append((name, f"{int(cur) // 1000} MHz" if cur.isdigit() else cur, rd("cpufreq/scaling_governor"),
                     rd("topology/core_id"), rd("topology/physical_package_id")))
    data = {"online": _read(f"{base}/online").strip(), "logical_cpus": len(rows),
            "governor": rows[0][2] if rows else "?",
            "note": "logical CPUs sharing a core_id are SMT (hyper-thread) siblings"}
    return data, _table([list(r) for r in rows], ["cpu", "frequency", "governor", "core", "package"])


def _load() -> tuple[dict, str]:
    one, five, fifteen, running, last = _read("/proc/loadavg").split()
    run, total = running.split("/")
    k = procfs.kernel_stats()
    data = {"load_1_5_15": f"{one} {five} {fifteen}", "runnable_threads_now": run, "threads_total": total,
            "last_pid_assigned": last, "blocked_on_io": k.get("procs_blocked"),
            "context_switches_since_boot": f"{k.get('ctxt', 0):,}", "forks_since_boot": f"{k.get('processes', 0):,}",
            "note": f"load is the average number of runnable or uninterruptible tasks; this machine has {os.cpu_count()} CPUs"}
    return data, "\n".join(f"{k}: {v}" for k, v in data.items())


def _locks() -> tuple[dict, str]:
    rows, kinds = [], {}
    for line in _read("/proc/locks").splitlines():
        parts = line.split()
        kinds[parts[1]] = kinds.get(parts[1], 0) + 1
        pid = parts[4] if parts[4].isdigit() else parts[5]
        name = procfs.read_stat(int(pid)).name if pid.isdigit() and procfs.read_stat(int(pid)) else "?"
        rows.append((parts[1], parts[2], parts[3], f"{pid} ({name})"))
    return {"file_locks": len(rows), "by_kind": kinds,
            "note": "FLOCK = flock(2), POSIX = fcntl(2) record locks, OFDLCK = open-file-description locks"}, \
        _table([list(r) for r in rows[:30]], ["kind", "mode", "type", "holder"])


def _filesystems() -> tuple[dict, str]:
    lines = _read("/proc/filesystems").splitlines()
    virtual = [line.split()[-1] for line in lines if line.startswith("nodev")]
    block = [line.split()[-1] for line in lines if not line.startswith("nodev")]
    return {"disk_filesystems": block, "virtual_filesystems": len(virtual)}, \
        "disk-backed: " + ", ".join(block) + "\nvirtual: " + ", ".join(virtual)


_PROCESS_VIEWS = {"memory_map": _memory_map, "open_files": _open_files, "limits": _limits, "namespaces": _namespaces,
                  "cgroup": _cgroup, "threads": _threads, "scheduler": _scheduler, "io": _io}
_SYSTEM_VIEWS = {"interrupts": _interrupts, "vmstat": _vmstat, "meminfo": _meminfo, "pressure": _pressure,
                 "buddyinfo": _buddyinfo, "mounts": _mounts, "modules": _modules, "swaps": _swaps, "cpus": _cpus,
                 "load": _load, "locks": _locks, "filesystems": _filesystems}


def _internals_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    topic = args["topic"]
    if topic in _PROCESS_VIEWS:
        if not args.get("pid"):
            raise ToolArgError(f"topic '{topic}' needs a pid")
        return [Requirement("proc.read", pid=args["pid"], op=topic)]
    return [Requirement("sys.read", op=topic)]


def _internals(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    topic = args["topic"]
    try:
        if topic in _PROCESS_VIEWS:
            data, shown = _PROCESS_VIEWS[topic](args["pid"])
            label = f"{topic} of PID {args['pid']}"
        else:
            data, shown = _SYSTEM_VIEWS[topic]()
            label = topic
    except PermissionError:
        return ToolOutput.fail(f"the kernel does not let you read {topic} of that process (it belongs to another "
                               "user or is protected; ptrace access mode is required)")
    except FileNotFoundError:
        return ToolOutput.fail("that process no longer exists" if topic in _PROCESS_VIEWS else f"{topic} is not available")
    data["table"] = Clip.of(shown, 3500).text
    return ToolOutput(True, data, f"{label}: {len(shown.splitlines())} line(s)", shown)


register(Tool(
    name="os_internals",
    description="Look inside the OS. Per process (needs pid): " + ", ".join(PROCESS_TOPICS) +
                ". System-wide: " + ", ".join(SYSTEM_TOPICS) + ".",
    rights=("proc.read", "sys.read"),
    params={
        "topic": param("string", "What to inspect", enum=PROCESS_TOPICS + SYSTEM_TOPICS),
        "pid": param("integer", "Process ID (for per-process topics)", minimum=1),
    },
    required=("topic",),
    requires=_internals_requires,
    run=_internals,
))


# --------------------------------------------------------------- sockets

_TCP_STATES = {"01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV", "04": "FIN_WAIT1", "05": "FIN_WAIT2",
               "06": "TIME_WAIT", "07": "CLOSE", "08": "CLOSE_WAIT", "09": "LAST_ACK", "0A": "LISTEN", "0B": "CLOSING"}


def _addr(hexaddr: str) -> str:
    ip_hex, port_hex = hexaddr.split(":")
    raw = bytes.fromhex(ip_hex)
    if len(raw) == 4:
        ip = socket.inet_ntop(socket.AF_INET, raw[::-1])
    else:
        ip = socket.inet_ntop(socket.AF_INET6, b"".join(raw[i:i + 4][::-1] for i in range(0, 16, 4)))
        ip = f"[{ip}]"
    return f"{ip}:{int(port_hex, 16)}"


def _socket_owners() -> dict[str, tuple[int, str]]:
    owners = {}
    for pid in procfs.list_pids():
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    target = os.readlink(f"/proc/{pid}/fd/{fd}")
                except OSError:
                    continue
                if target.startswith("socket:["):
                    stat = procfs.read_stat(pid)
                    owners[target[8:-1]] = (pid, stat.name if stat else "?")
        except OSError:
            continue  # other users' file descriptors are not visible to us
    return owners


def _connections(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    owners = _socket_owners()
    rows, counts = [], {}
    for proto in ("tcp", "tcp6", "udp", "udp6"):
        if args["protocol"] != "all" and not proto.startswith(args["protocol"]):
            continue
        try:
            lines = _read(f"/proc/net/{proto}").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            local, remote, st, uid, inode = f[1], f[2], f[3], int(f[7]), f[9]
            if proto.startswith("tcp"):
                state = _TCP_STATES.get(st, st)
            else:  # UDP has no connection states: bound-but-unconnected sockets are the "listeners"
                state = "UNCONN" if set(remote.replace(":", "")) == {"0"} else "CONNECTED"
            want = args["state"]
            if want == "listening" and state not in ("LISTEN", "UNCONN"):
                continue
            if want == "established" and state not in ("ESTABLISHED", "CONNECTED"):
                continue
            counts[state] = counts.get(state, 0) + 1
            owner = owners.get(inode)
            rows.append((proto, _addr(local), _addr(remote) if state not in ("LISTEN", "UNCONN") else "*", state,
                         procfs.username(uid),
                         f"{owner[0]} ({owner[1]})" if owner else "?"))
    rows.sort(key=lambda r: (r[3] not in ("LISTEN", "UNCONN"), r[0], r[1]))
    table = _table([list(r) for r in rows[:60]], ["proto", "local", "remote", "state", "user", "process"])
    data = {"sockets": len(rows), "by_state": counts, "table": Clip.of(table, 3500).text,
            "note": "read from /proc/net; '?' processes belong to other users (their fd tables are private)"}
    return ToolOutput(True, data, f"{len(rows)} socket(s) ({args['state']})", table)


register(Tool(
    name="network_connections",
    description="Open network sockets from /proc/net: listening ports and connections with their owning process.",
    rights=("sys.read",),
    params={
        "state": param("string", "Which sockets", enum=["listening", "established", "all"], default="listening"),
        "protocol": param("string", "tcp, udp or all", enum=["tcp", "udp", "all"], default="all"),
    },
    requires=lambda a, c: [Requirement("sys.read", op="sockets")],
    run=_connections,
))


# --------------------------------------------------------------- hardware


def _run(argv: list[str], timeout: float = 10) -> str:
    if not shutil.which(argv[0]):
        return ""
    try:
        result = subprocess.run(argv, capture_output=True, timeout=timeout, env={"PATH": "/usr/bin:/bin:/usr/sbin",
                                                                                 "LANG": "C.UTF-8"})
        return result.stdout.decode(errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _sysread(path: str) -> str:
    try:
        return _read(path).strip()
    except OSError:
        return ""


def _battery() -> dict:
    out = {}
    for dev in sorted(os.listdir("/sys/class/power_supply")) if os.path.isdir("/sys/class/power_supply") else []:
        base = f"/sys/class/power_supply/{dev}"
        kind = _sysread(f"{base}/type")
        if kind == "Mains":
            out["charger"] = "plugged in" if _sysread(f"{base}/online") == "1" else "unplugged"
        elif kind == "Battery":
            now = _sysread(f"{base}/energy_now") or _sysread(f"{base}/charge_now")
            full = _sysread(f"{base}/energy_full") or _sysread(f"{base}/charge_full")
            design = _sysread(f"{base}/energy_full_design") or _sysread(f"{base}/charge_full_design")
            power = _sysread(f"{base}/power_now")
            info = {"level": f"{_sysread(base + '/capacity')}%", "status": _sysread(f"{base}/status"),
                    "model": f"{_sysread(base + '/manufacturer')} {_sysread(base + '/model_name')}".strip(),
                    "cycles": _sysread(f"{base}/cycle_count")}
            if full.isdigit() and design.isdigit() and int(design):
                info["health"] = f"{int(full) / int(design) * 100:.0f}% of design capacity"
            if power.isdigit() and int(power):
                info["power_draw"] = f"{int(power) / 1e6:.1f} W"
                if info["status"] == "Discharging" and now.isdigit():
                    hours = int(now) / int(power)
                    info["time_left"] = f"{int(hours)}h {int(hours % 1 * 60)}m"
            out[dev] = info
    return out or {"battery": "none found"}


def _thermal() -> dict:
    out = {}
    base = "/sys/class/thermal"
    for zone in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        if zone.startswith("thermal_zone"):
            temp = _sysread(f"{base}/{zone}/temp")
            if temp.lstrip("-").isdigit():
                out[_sysread(f"{base}/{zone}/type") or zone] = f"{int(temp) / 1000:.1f} °C"
    raw = _run(["sensors", "-j"])
    if raw:
        try:
            for chip, features in json.loads(raw).items():
                for feature, values in features.items():
                    if isinstance(values, dict):
                        for key, value in values.items():
                            if key.endswith("_input") and key.startswith(("temp", "fan")):
                                unit = "°C" if key.startswith("temp") else "RPM"
                                out[f"{chip.split('-')[0]} {feature}"] = f"{value:.0f} {unit}"
        except (ValueError, AttributeError):
            pass
    gpu = _run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]).strip()
    if gpu:
        out["NVIDIA GPU"] = f"{gpu} °C"
    return out


def _disks() -> dict:
    raw = _run(["lsblk", "-J", "-b", "-o", "NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,ROTA,TRAN"])
    out = {}
    try:
        devices = json.loads(raw)["blockdevices"]
    except (ValueError, KeyError):
        return {"disks": "lsblk unavailable"}

    def walk(dev, depth=0):
        mounts = [m for m in (dev.get("mountpoints") or []) if m]
        line = f"{dev['type']} {fmt_bytes(int(dev.get('size') or 0))}"
        if dev.get("model"):
            line += f" {dev['model'].strip()}"
        if dev["type"] == "disk":
            line += " (HDD)" if dev.get("rota") else " (SSD/NVMe)"
            sched = _sysread(f"/sys/block/{dev['name']}/queue/scheduler")
            if sched:
                line += f", I/O scheduler {sched}"
        if dev.get("fstype"):
            line += f", {dev['fstype']}"
        for m in mounts:
            try:
                vfs = os.statvfs(m)
                used = (vfs.f_blocks - vfs.f_bfree) * vfs.f_frsize
                line += f", {m} {used / max(1, vfs.f_blocks * vfs.f_frsize) * 100:.0f}% used"
            except OSError:
                line += f", {m}"
        out["  " * depth + dev["name"]] = line
        for child in dev.get("children", []):
            walk(child, depth + 1)

    for dev in devices:
        if dev["type"] != "loop":
            walk(dev)
    return out


def _network() -> dict:
    out = {}
    addrs = {}
    try:
        for iface in json.loads(_run(["ip", "-j", "addr"]) or "[]"):
            addrs[iface["ifname"]] = [f"{a['local']}/{a['prefixlen']}" for a in iface.get("addr_info", [])]
    except (ValueError, KeyError):
        pass
    for iface in sorted(os.listdir("/sys/class/net")):
        base = f"/sys/class/net/{iface}"
        rx, tx = _sysread(f"{base}/statistics/rx_bytes"), _sysread(f"{base}/statistics/tx_bytes")
        speed = _sysread(f"{base}/speed")
        line = f"{_sysread(base + '/operstate')}, MAC {_sysread(base + '/address')}"
        if speed.isdigit() and int(speed) > 0:
            line += f", {speed} Mb/s"
        if rx.isdigit():
            line += f", received {fmt_bytes(int(rx))}, sent {fmt_bytes(int(tx))}"
        if addrs.get(iface):
            line += f", {' '.join(addrs[iface])}"
        out[iface] = line
    return out


def _usb() -> dict:
    lines = [line.split("ID ", 1)[-1] for line in _run(["lsusb"]).splitlines()]
    return {f"device {i + 1}": line for i, line in enumerate(lines)} or {"usb": "lsusb unavailable"}


def _gpu() -> dict:
    out = {}
    raw = _run(["nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used,"
                "memory.total,power.draw", "--format=csv,noheader"]).strip()
    for i, line in enumerate(raw.splitlines()):
        name, driver, temp, util, used, total, power = [p.strip() for p in line.split(",")]
        out[f"gpu{i}"] = f"{name}, driver {driver}, {temp} °C, {util} busy, VRAM {used} / {total}, {power}"
    for card in sorted(os.listdir("/sys/class/drm")) if os.path.isdir("/sys/class/drm") else []:
        if re.fullmatch(r"card\d+", card):
            vendor = {"0x10de": "NVIDIA", "0x8086": "Intel", "0x1002": "AMD"}.get(
                _sysread(f"/sys/class/drm/{card}/device/vendor"), "unknown vendor")
            uevent = dict(line.split("=", 1) for line in _sysread(f"/sys/class/drm/{card}/device/uevent").splitlines()
                          if "=" in line)
            out[card] = f"{vendor}, kernel driver {uevent.get('DRIVER', '?')}"
    return out or {"gpu": "none found"}


def _cpu() -> dict:
    caches = {}
    base = "/sys/devices/system/cpu/cpu0/cache"
    for idx in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        level, kind, size = (_sysread(f"{base}/{idx}/{n}") for n in ("level", "type", "size"))
        caches[f"L{level} {kind.lower()}"] = size
    flags = ""
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("flags"):
            flags = line
            break
    max_freq = _sysread("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    return {"model": procfs.cpu_model(), "logical_cpus": os.cpu_count(),
            "max_frequency": f"{int(max_freq) // 1000} MHz" if max_freq.isdigit() else "?",
            "caches": caches, "virtualization": "VT-x" if " vmx" in flags else "AMD-V" if " svm" in flags else "none",
            "governor": _sysread("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")}


_HARDWARE = {"battery": _battery, "thermal": _thermal, "disks": _disks, "network": _network, "usb": _usb,
             "gpu": _gpu, "cpu": _cpu}


def _hardware(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    topics = list(_HARDWARE) if args["topic"] == "summary" else [args["topic"]]
    data = {}
    for topic in topics:
        if args["topic"] == "summary" and topic in ("usb",):
            continue
        data[topic] = _HARDWARE[topic]()
    shown = json.dumps(data, indent=1, ensure_ascii=False)
    return ToolOutput(True, data, f"hardware: {', '.join(data)}", shown)


register(Tool(
    name="hardware_info",
    description="Hardware from sysfs and system tools: battery (health, time left), thermal (temperatures, fans), "
                "disks (SSD/HDD, I/O scheduler, usage), network interfaces, usb, gpu, cpu, or a summary.",
    rights=("sys.read",),
    params={"topic": param("string", "What to show", enum=["summary", *_HARDWARE], default="summary")},
    requires=lambda a, c: [Requirement("sys.read", op="hardware")],
    run=_hardware,
))


# --------------------------------------------------------------- services

_UNIT = re.compile(r"[A-Za-z0-9@._:\\-]{1,128}")


def _systemctl(args: dict) -> list[str]:
    return ["systemctl"] + (["--user"] if args["scope"] == "user" else []) + ["--no-pager", "--plain"]


def _services_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    if args.get("name") and not _UNIT.fullmatch(args["name"]):
        raise ToolArgError("invalid unit name")
    if args["action"] in ("status",) and not args.get("name"):
        raise ToolArgError("status needs a service name")
    right = "sys.logs" if args["action"] == "logs" else "sys.read"
    return [Requirement(right, op=f"services {args['action']}", target=args.get("name"))]


def _services(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    action, name = args["action"], args.get("name")
    if action == "failed":
        text = _run(_systemctl(args) + ["list-units", "--failed", "--no-legend"]).strip()
        rows = [line.split()[0] for line in text.splitlines() if line.strip()]
        data = {"failed_units": rows or "none"}
        return ToolOutput(True, data, f"{len(rows)} failed unit(s)", text or "no failed units")
    if action == "list":
        text = _run(_systemctl(args) + ["list-units", "--type=service", "--state=running", "--no-legend"]).strip()
        lines = text.splitlines()
        names = []
        for line in lines:
            parts = line.split(maxsplit=4)  # unit load active sub description
            names.append(f"{parts[0]}  {parts[4] if len(parts) > 4 else ''}".rstrip())
        data = {"running_services": len(lines), "list": Clip.of("\n".join(names), 3000).text}
        return ToolOutput(True, data, f"{len(lines)} running service(s)", data["list"])
    if action == "timers":
        text = _run(_systemctl(args) + ["list-timers", "--no-legend"]).strip()
        return ToolOutput(True, {"timers": Clip.of(text, 3000).text}, f"{len(text.splitlines())} timer(s)", text)
    if action == "status":
        props = "Id,Description,LoadState,ActiveState,SubState,UnitFileState,MainPID,ExecMainStartTimestamp,MemoryCurrent,CPUUsageNSec,NRestarts,Result"
        text = _run(_systemctl(args) + ["show", name, "-p", props]).strip()
        data = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
        if data.get("LoadState") == "not-found":
            return ToolOutput.fail(f"no unit named {name} ({args['scope']} scope)")
        if data.get("MemoryCurrent", "").isdigit():
            data["MemoryCurrent"] = fmt_bytes(int(data["MemoryCurrent"]))
        if data.get("CPUUsageNSec", "").isdigit():
            data["CPUUsageNSec"] = f"{int(data['CPUUsageNSec']) / 1e9:.1f}s"
        return ToolOutput(True, data, f"{name}: {data.get('ActiveState', '?')} ({data.get('SubState', '?')})",
                          "\n".join(f"{k}: {v}" for k, v in data.items()))
    # logs
    argv = ["journalctl"] + (["--user"] if args["scope"] == "user" else []) + \
        ["--no-pager", "-o", "short-iso", "-n", str(args["lines"])]
    if name:
        argv += ["-u", name]
    else:
        argv += ["-p", "warning"]
    if args.get("since"):
        if not re.fullmatch(r"[0-9A-Za-z :\-]{1,40}", args["since"]):
            return ToolOutput.fail("invalid 'since' value (examples: '1 hour ago', 'today', '2026-10-13 09:00')")
        argv += ["--since", args["since"]]
    text = _run(argv, timeout=20).strip()
    clip = Clip.of(text, 3500)
    return ToolOutput(True, {"log": clip.text, "lines": len(text.splitlines())},
                      f"{len(text.splitlines())} log line(s){' for ' + name if name else ' (warnings and errors)'}", text)


register(Tool(
    name="services",
    description="systemd, read-only: failed units, running services, timers, one service's status, or journal logs "
                "(of a service, or recent warnings/errors when no name is given).",
    rights=("sys.read",),
    params={
        "action": param("string", "What to show", enum=["failed", "list", "timers", "status", "logs"], default="failed"),
        "name": param("string", "Unit name, e.g. 'NetworkManager' or 'nginx.service'"),
        "scope": param("string", "system or user units", enum=["system", "user"], default="system"),
        "lines": param("integer", "Log lines", default=40, minimum=5, maximum=200),
        "since": param("string", "Logs since, e.g. '1 hour ago' or 'today'"),
    },
    requires=_services_requires,
    run=_services,
))


# ---------------------------------------------------------- syscall tracing

SYSCALL_HELP = {
    "openat": "open a file (relative to a directory fd)", "read": "read from a file descriptor",
    "write": "write to a file descriptor", "close": "release a file descriptor", "mmap": "map memory or a file",
    "munmap": "unmap memory", "mprotect": "change memory protection", "brk": "grow the heap",
    "fstat": "file metadata from an fd", "newfstatat": "file metadata by path", "statx": "extended file metadata",
    "execve": "replace the program image", "clone": "create a process or thread", "clone3": "create a process or thread",
    "wait4": "wait for a child to exit", "exit_group": "end the process", "rt_sigaction": "install a signal handler",
    "rt_sigprocmask": "block/unblock signals", "ioctl": "device-specific control", "getdents64": "read directory entries",
    "pread64": "read at an offset", "lseek": "move the file offset", "access": "check permissions",
    "faccessat2": "check permissions", "arch_prctl": "set thread-local storage", "set_tid_address": "thread id bookkeeping",
    "futex": "fast user-space locking", "prlimit64": "get/set resource limits", "getrandom": "random bytes from the kernel",
    "socket": "create a network endpoint", "connect": "connect a socket", "sendto": "send on a socket",
    "recvfrom": "receive from a socket", "pipe2": "create a pipe", "dup2": "duplicate an fd", "dup3": "duplicate an fd",
    "fcntl": "fd flags and locks", "poll": "wait for fd events", "select": "wait for fd events",
    "epoll_wait": "wait for fd events", "readlink": "read a symlink", "getcwd": "current directory",
    "chdir": "change directory", "unlink": "delete a name", "unlinkat": "delete a name", "rename": "rename a file",
    "mkdir": "create a directory", "set_robust_list": "thread cleanup list", "rseq": "restartable sequences",
    "sysinfo": "system statistics", "uname": "kernel name and version", "getuid": "real user id",
    "geteuid": "effective user id", "getpid": "process id", "kill": "send a signal", "nanosleep": "sleep",
    "clock_nanosleep": "sleep", "madvise": "memory usage hints", "fadvise64": "file access hints",
}


def _trace_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    from agentlayer.tools.exec import parse

    if not shutil.which("strace"):
        raise ToolArgError("strace is not installed (sudo apt install strace)")
    if args.get("filter") and not re.fullmatch(r"[a-z0-9_,%!]+", args["filter"]):
        raise ToolArgError("filter must look like 'openat,read' or '%file,%process'")
    cwd, argvs = parse({"command": args["command"], "cwd": args["cwd"]}, ctx)
    if len(argvs) != 1:
        raise ToolArgError("trace one program at a time (no pipes)")
    return [Requirement("exec", program=argvs[0][0], argv=argvs[0], cwd=cwd)]


def _trace(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    from agentlayer.tools.exec import SAFE_PATH, _child_env

    grant = decision.grant("exec")
    argv, cwd = decision.approved().argv, decision.approved().cwd
    program = os.path.realpath(os.path.join(cwd, argv[0])) if "/" in argv[0] else shutil.which(argv[0], path=SAFE_PATH)
    strace = shutil.which("strace", path=SAFE_PATH)
    if program is None or strace is None:
        return ToolOutput.fail(f"'{argv[0]}' or strace is not installed")
    if args["mode"] == "summary":
        flags = ["-f", "-c", "-S", "calls"]
    else:
        flags = ["-f", "-tt", "-s", "48", "-e", f"trace={args.get('filter') or '%process,%file,%network,%signal'}"]
    if args.get("filter") and args["mode"] == "summary":
        flags += ["-e", f"trace={args['filter']}"]
    exec_path, exec_argv = ctx.sandbox.wrap(strace, [strace, *flags, "--", program, *argv[1:]], cwd=cwd,
                                            workspace=ctx.workspace, network=False)
    result = run_pipeline([(exec_path, exec_argv)], cwd=cwd, env=_child_env(ctx), timeout=min(60, grant.timeout),
                          max_output=200_000, limits=grant.limits)
    err = result.stderr
    if args["mode"] == "summary":
        lines = err.splitlines()
        start = next((i for i, l in enumerate(lines) if l.startswith("% time")), None)
        if start is None:
            return ToolOutput.fail("strace produced no summary", stderr=Clip.of(err, 1500).text)
        rows = []
        for line in lines[start + 2:]:
            if line.startswith("------"):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[-1] == "total":
                break
            calls = int(parts[3])
            errors = int(parts[4]) if len(parts) == 6 else 0
            name = parts[-1]
            rows.append((name, calls, errors, SYSCALL_HELP.get(name, "")))
        table = _table([[n, c, e or "", h] for n, c, e, h in rows[:30]], ["syscall", "calls", "errors", "what it does"])
        total_calls = sum(r[1] for r in rows)
        data = {"program": " ".join(argv), "exit_code": result.exit_code, "distinct_syscalls": len(rows),
                "total_calls": total_calls, "table": table, "program_output": Clip.of(result.stdout, 800).text,
                "note": "one execve is the traced program itself being started; failed calls (errors) are often "
                        "harmless probes, e.g. looking for optional files"}
        return ToolOutput(True, data, f"{' '.join(argv)[:40]}: {total_calls} syscalls of {len(rows)} kinds", table)
    trace = "\n".join(err.splitlines()[:80])
    data = {"program": " ".join(argv), "exit_code": result.exit_code, "trace_lines": len(err.splitlines()),
            "trace": Clip.of(trace, 3500).text}
    return ToolOutput(True, data, f"traced {' '.join(argv)[:40]}: {len(err.splitlines())} lines", trace)


register(Tool(
    name="trace_syscalls",
    description="Run an allow-listed program under strace inside the sandbox and show the system calls it makes: "
                "a summary table (counts, errors, meaning) or the first lines of the raw trace.",
    rights=("exec",),
    params={
        "command": param("string", "Program and arguments, e.g. 'ls -la' or 'python3 script.py'"),
        "mode": param("string", "summary or log", enum=["summary", "log"], default="summary"),
        "filter": param("string", "Optional syscall filter, e.g. 'openat,read' or '%file'"),
        "cwd": param("string", "Working directory", default="."),
    },
    required=("command",),
    requires=_trace_requires,
    run=_trace,
))


# ------------------------------------------------------------ watch a path


def _watch(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    path = decision.approved().path
    events = inotify.watch(path, args["seconds"], recursive=args["recursive"], until_first=args["mode"] == "first")
    lines = [e.describe() for e in events]
    shown = "\n".join(lines) or "(nothing happened)"
    data = {"watched": ctx.show(path), "events": len(events), "log": Clip.of(shown, 3000).text,
            "note": "events come from the kernel via inotify; no polling"}
    summary = f"{len(events)} event(s) in {ctx.show(path)}" if events else f"no changes in {ctx.show(path)} for {args['seconds']}s"
    return ToolOutput(True, data, summary, shown)


register(Tool(
    name="watch_path",
    description="Wait for changes in a folder or file (inotify): returns created/modified/deleted/moved events, "
                "e.g. to wait until a download finishes.",
    rights=("fs.read",),
    params={
        "path": param("string", "Folder or file to watch"),
        "seconds": param("integer", "How long to wait at most", default=60, minimum=1, maximum=600),
        "recursive": param("boolean", "Also watch subfolders", default=False),
        "mode": param("string", "first: return after the first burst of events; all: collect until the time is up",
                      enum=["first", "all"], default="first"),
    },
    required=("path",),
    requires=lambda a, c: [Requirement("fs.read", path=c.real(a["path"]), lexical=c.lexical(a["path"]), op="watch")],
    run=_watch,
))

