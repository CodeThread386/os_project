"""inotify(7) through ctypes: kernel notifications of file-system events.

Instead of polling a directory, the process asks the kernel for an inotify
file descriptor, registers watches on paths, and then simply blocks in
select(2) until the kernel has events to deliver. Each event read(2) from
the descriptor is a ``struct inotify_event`` (wd, mask, cookie, len, name).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import select
import struct
import time
from dataclasses import dataclass

IN_ACCESS = 0x1
IN_MODIFY = 0x2
IN_ATTRIB = 0x4
IN_CLOSE_WRITE = 0x8
IN_CLOSE_NOWRITE = 0x10
IN_OPEN = 0x20
IN_MOVED_FROM = 0x40
IN_MOVED_TO = 0x80
IN_CREATE = 0x100
IN_DELETE = 0x200
IN_DELETE_SELF = 0x400
IN_MOVE_SELF = 0x800
IN_UNMOUNT = 0x2000
IN_Q_OVERFLOW = 0x4000
IN_IGNORED = 0x8000
IN_DONT_FOLLOW = 0x02000000
IN_ISDIR = 0x40000000

DEFAULT_MASK = (IN_MODIFY | IN_ATTRIB | IN_CLOSE_WRITE | IN_MOVED_FROM | IN_MOVED_TO | IN_CREATE | IN_DELETE
                | IN_DELETE_SELF | IN_MOVE_SELF)
_NAMES = [
    (IN_CREATE, "created"), (IN_DELETE, "deleted"), (IN_MODIFY, "modified"), (IN_CLOSE_WRITE, "finished writing"),
    (IN_MOVED_FROM, "moved away"), (IN_MOVED_TO, "moved in"), (IN_ATTRIB, "metadata changed"),
    (IN_DELETE_SELF, "watched path deleted"), (IN_MOVE_SELF, "watched path moved"), (IN_Q_OVERFLOW, "event queue overflowed"),
]
_EVENT = struct.Struct("iIII")
_libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)


@dataclass
class Event:
    after: float  # seconds after the watch started
    directory: str
    name: str
    what: list[str]
    is_dir: bool

    def describe(self) -> str:
        target = os.path.join(self.directory, self.name) if self.name else self.directory
        return f"+{self.after:5.1f}s  {', '.join(self.what)}: {target}{'/' if self.is_dir else ''}"


def _check(result: int) -> int:
    if result < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return result


def watch(path: str, seconds: float, *, recursive: bool = False, settle: float = 1.0, until_first: bool = True,
          max_dirs: int = 512, max_events: int = 200) -> list[Event]:
    """Block until events arrive under ``path`` (or ``seconds`` pass)."""
    fd = _check(_libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC))
    watches: dict[int, str] = {}
    try:
        targets = [path]
        if recursive and os.path.isdir(path):
            for dirpath, dirnames, _ in os.walk(path, followlinks=False):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                targets += [os.path.join(dirpath, d) for d in dirnames]
                if len(targets) >= max_dirs:
                    break
        for target in targets[:max_dirs]:
            wd = _check(_libc.inotify_add_watch(fd, os.fsencode(target), DEFAULT_MASK | IN_DONT_FOLLOW))
            watches[wd] = target
        start = time.monotonic()
        deadline = start + seconds
        first: float | None = None
        events: list[Event] = []
        while len(events) < max_events:
            now = time.monotonic()
            limit = deadline - now
            if first is not None and until_first:
                limit = min(limit, first + settle - now)
            if limit <= 0:
                break
            ready, _, _ = select.select([fd], [], [], limit)
            if not ready:
                continue
            data = os.read(fd, 64 * 1024)
            offset = 0
            while offset < len(data):
                wd, mask, _cookie, length = _EVENT.unpack_from(data, offset)
                raw = data[offset + _EVENT.size: offset + _EVENT.size + length]
                offset += _EVENT.size + length
                if mask & IN_IGNORED:
                    continue
                what = [label for bit, label in _NAMES if mask & bit]
                if what:
                    events.append(Event(time.monotonic() - start, watches.get(wd, "?"), os.fsdecode(raw.rstrip(b"\0")),
                                        what, bool(mask & IN_ISDIR)))
            if events and first is None:
                first = time.monotonic()
        return events
    finally:
        os.close(fd)
