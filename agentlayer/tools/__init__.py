"""The OS-interface layer: every action the agent can take, as a declared tool.

Tools are grouped. The ``core`` group is always offered to the model; the
others are loaded on demand (by keywords in the request, or by the model
calling ``enable_tools``), which keeps every request small enough for the
per-request token limits of hosted models.
"""

from agentlayer.tools import (
    admin, capability, exec, fs, gui, internals, memory, net, proc, system, tasks, undo, vision,
)
from agentlayer.tools.base import REGISTRY, Tool, ToolArgError, ToolContext, ToolOutput, prepare_args

# Importing a tool module registers its tools.
MODULES = (admin, capability, exec, fs, gui, internals, memory, net, proc, system, tasks, undo, vision)

GROUPS: dict[str, str] = {
    "core": "everyday file, process, command, web and memory tools (always loaded)",
    "files": "file metadata, copies, folders, disk usage, watching folders for changes, describing images",
    "processes": "process details, signals, priorities, OS internals of a process, syscall tracing",
    "system": "hardware (battery, temperatures, disks, GPU, USB, network), open ports, services and logs",
    "desktop": "launch apps, open files/URLs, manage windows, screenshots, look at the screen, clipboard, notifications",
    "admin": "install/remove software and run admin programs as root (always confirmed)",
    "automation": "schedule recurring or one-off tasks with systemd timers",
}

# name: (group, mutating, untrusted output, what it does underneath)
_W = lambda a: a.get("action", "list") not in ("list", "read", "recall", "search", "log", "status")  # noqa: E731
TOOL_META: dict[str, tuple] = {
    "system_info": ("core", False, False, "uname, statvfs; reads /proc/meminfo, /proc/stat, /proc/loadavg"),
    "list_directory": ("core", False, False, "openat walk (O_NOFOLLOW), getdents64, fstatat"),
    "read_file": ("core", False, True, "openat walk (O_NOFOLLOW), fstat, lseek, read, close"),
    "search_files": ("core", False, True, "fwalk: openat, getdents64, fstatat; read per matching file"),
    "write_file": ("core", True, False, "openat walk, mkdirat, openat(O_CREAT|O_TRUNC|O_NOFOLLOW), write, fsync"),
    "move_path": ("core", True, False, "openat walk to both parents, renameat"),
    "delete_path": ("core", True, False, "renameat into the freedesktop Trash (undoable)"),
    "list_processes": ("core", False, False, "reads /proc/<pid>/stat twice (CPU% is a rate), /proc/<pid>/cmdline"),
    "run_command": ("core", True, True, "pipe, fork, setsid, dup2, setrlimit, fchdir, execve (bwrap: clone into new namespaces), poll, wait4"),
    "fetch_url": ("core", False, True, "getaddrinfo, socket, connect, TLS over send/recv"),
    "memory": ("core", _W, False, "flock, write, fsync, rename (atomic replace)"),
    "undo": ("core", _W, False, "renameat out of the Trash, copy_file_range, ftruncate"),
    "request_capability": ("core", True, False, "none: adds an expiring entry to the broker's capability list"),
    "file_info": ("files", False, False, "openat(O_PATH), fstat, lstat, readlink, access"),
    "copy_path": ("files", True, False, "openat walk, copy_file_range, fsync"),
    "make_directory": ("files", True, False, "openat walk, mkdirat"),
    "disk_usage": ("files", False, False, "statvfs, getdents64, lstat"),
    "watch_path": ("files", False, True, "inotify_init1, inotify_add_watch, select, read (kernel file events)"),
    "describe_image": ("files", False, True, "openat walk, read; execve(ffmpeg) to downscale; HTTPS to a vision model"),
    "process_info": ("processes", False, False, "reads /proc/<pid>/stat,status,cmdline; readlink exe,cwd; sched_getaffinity"),
    "signal_process": ("processes", True, False, "kill(2)"),
    "set_priority": ("processes", True, False, "getpriority, setpriority(PRIO_PROCESS)"),
    "os_internals": ("processes", False, False, "reads /proc/<pid>/maps,smaps_rollup,fd,limits,ns,cgroup,task,sched and /proc/{interrupts,vmstat,pressure}"),
    "trace_syscalls": ("processes", True, True, "strace: ptrace(PTRACE_SEIZE) on a sandboxed child, wait4"),
    "hardware_info": ("system", False, False, "reads /sys/class/{power_supply,thermal,net,drm}, /sys/block; lsblk, ip, lsusb, nvidia-smi"),
    "network_connections": ("system", False, False, "reads /proc/net/{tcp,tcp6,udp,udp6}; readlink /proc/<pid>/fd/* to map socket inodes"),
    "services": ("system", False, True, "execve(systemctl, journalctl) read-only queries"),
    "open_app": ("desktop", True, False, "fork + setsid -f, execve(gtk-launch)"),
    "open_path": ("desktop", True, False, "fork + setsid -f, execve(xdg-open)"),
    "windows": ("desktop", lambda a: a.get("action", "list") != "list", True, "D-Bus over a UNIX socket to KWin's scripting interface"),
    "screenshot": ("desktop", True, False, "D-Bus screenshot via Spectacle, copy_file_range into the workspace"),
    "look_at_screen": ("desktop", False, True, "screenshot, execve(ffmpeg) to downscale, HTTPS to a vision model"),
    "clipboard": ("desktop", lambda a: a.get("action") == "write", True, "execve(wl-paste / wl-copy) with pipes"),
    "notify": ("desktop", True, False, "D-Bus call org.freedesktop.Notifications.Notify"),
    "packages": ("admin", _W, False, "fork, execve(sudo → setuid root → apt-get/snap/flatpak), password through a pipe"),
    "run_as_root": ("admin", True, False, "fork, execve(sudo → setuid root), password through a pipe"),
    "schedule_task": ("automation", True, False, "writes systemd user units, execve(systemctl --user enable --now)"),
    "tasks": ("automation", _W, False, "execve(systemctl --user show/disable), reads task logs"),
}

for _name, (_group, _mutating, _untrusted, _syscalls) in TOOL_META.items():
    if _name in REGISTRY:
        _t = REGISTRY[_name]
        _t.group, _t.mutating, _t.untrusted, _t.syscalls = _group, _mutating, _untrusted, _syscalls

ORDER = [name for name in TOOL_META if name in REGISTRY]


def all_tools() -> list[Tool]:
    return [REGISTRY[name] for name in ORDER]


def exposed_tools(policy) -> list[Tool]:
    """Least privilege at the interface too: the model only sees tools it holds rights for."""
    return [t for t in all_tools() if all(policy.has(r) for r in t.rights)]


ENABLE_TOOLS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "enable_tools",
        "description": "Load extra tool groups for this conversation (see the list in your instructions).",
        "parameters": {
            "type": "object",
            "properties": {"groups": {"type": "array", "items": {"type": "string"}, "description": "Group names"}},
            "required": ["groups"],
        },
    },
}

__all__ = ["ENABLE_TOOLS_SCHEMA", "GROUPS", "REGISTRY", "Tool", "ToolArgError", "ToolContext", "ToolOutput",
           "all_tools", "exposed_tools", "prepare_args"]
