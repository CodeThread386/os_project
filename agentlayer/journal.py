"""Undo journal and the desktop Trash: a safety net for an agent that edits files.

Every change a file tool makes is journaled before or right after it
happens, so it can be reversed later:

* overwrite  -> the old contents are copied aside first (copy_file_range);
* append     -> the old length is remembered (undo is one ftruncate);
* create     -> undo moves the new file to the Trash;
* delete     -> the file is *moved* into the freedesktop.org Trash
                (~/.local/share/Trash), where Dolphin and other file
                managers can restore it too;
* move/copy/mkdir -> recorded, reversed with rename/trash/rmdir.

Undo refuses when the file changed since (size or modification time differ),
so it never clobbers newer work. The journal and its backups live in
AgentLayer's state directory, which the agent's file tools cannot touch.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import shutil
import stat
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator
from urllib.parse import quote, unquote

from agentlayer import safeio
from agentlayer.config import state_dir

MAX_BACKUP_BYTES = 64 * 1024 * 1024
MAX_ENTRIES = 300


class UndoError(Exception):
    pass


def _data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or "~/.local/share").expanduser()


class Trash:
    """The freedesktop.org Trash specification, home trash only."""

    def __init__(self, root: Path | None = None) -> None:
        root = Path(root) if root else _data_home() / "Trash"
        (root / "files").mkdir(parents=True, exist_ok=True, mode=0o700)
        (root / "info").mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root = Path(os.path.realpath(root))

    def put(self, path: str) -> tuple[str, str]:
        """Move ``path`` into the Trash; returns (trashed file, .trashinfo file)."""
        base = os.path.basename(path.rstrip("/")) or "unnamed"
        stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        for n in range(1, 10_000):
            name = base if n == 1 else f"{base}.{n}"
            info = self.root / "info" / f"{name}.trashinfo"
            try:  # O_EXCL reserves the name atomically, as the spec requires
                fd = os.open(info, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            with os.fdopen(fd, "w") as f:
                f.write(f"[Trash Info]\nPath={quote(path)}\nDeletionDate={stamp}\n")
            target = str(self.root / "files" / name)
            if os.path.lexists(target):
                os.unlink(info)
                continue
            try:
                safeio.rename(path, target)
            except OSError as exc:
                if exc.errno != errno.EXDEV:  # another filesystem -> copy, then delete
                    os.unlink(info)
                    raise
                shutil.move(path, target)
            return target, str(info)
        raise UndoError("could not find a free name in the Trash")

    def restore(self, trashed: str, info: str, original: str) -> None:
        if os.path.lexists(original):
            raise UndoError(f"{original} exists again; move it away first")
        if not os.path.lexists(trashed):
            raise UndoError("the item is no longer in the Trash (it was emptied or restored)")
        safeio.make_dirs(os.path.dirname(original))
        safeio.rename(trashed, original)
        try:
            os.unlink(info)
        except FileNotFoundError:
            pass

    def original_path(self, info: str) -> str | None:
        try:
            for line in Path(info).read_text().splitlines():
                if line.startswith("Path="):
                    return unquote(line[5:])
        except OSError:
            pass
        return None


@dataclass
class Entry:
    id: str
    op: str  # write | append | create | trash | move | copy | mkdir
    path: str
    time: float
    summary: str
    dest: str = ""
    backup: str = ""
    trash_file: str = ""
    trash_info: str = ""
    old_size: int = -1
    after_size: int = -1
    after_mtime: float = -1.0
    undone: bool = False
    note: str = ""

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.time).strftime("%m-%d %H:%M:%S")


def _fingerprint(path: str) -> tuple[int, float]:
    st = os.lstat(path)
    return st.st_size, st.st_mtime


class UndoJournal:
    def __init__(self, root: Path | None = None, trash: Trash | None = None) -> None:
        self.root = Path(root) if root else state_dir() / "undo"
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.file = self.root / "journal.json"
        self._trash = trash

    @property
    def trash_bin(self) -> Trash:
        if self._trash is None:
            self._trash = Trash()
        return self._trash

    # ----------------------------------------------------------- storage

    @contextmanager
    def _locked(self) -> Iterator[list[Entry]]:
        lock = os.open(str(self.file) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            entries = self._read()
            yield entries
            del entries[: max(0, len(entries) - MAX_ENTRIES)]
            live = {e.backup for e in entries if e.backup}
            for blob in self.blobs.iterdir():  # drop backups of pruned entries
                if str(blob) not in live:
                    blob.unlink(missing_ok=True)
            tmp = self.file.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump([asdict(e) for e in entries], f, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.file)
        finally:
            os.close(lock)

    def _read(self) -> list[Entry]:
        try:
            return [Entry(**e) for e in json.loads(self.file.read_text())]
        except (OSError, ValueError, TypeError):
            return []

    def _add(self, entry: Entry) -> Entry:
        with self._locked() as entries:
            entries.append(entry)
        return entry

    def entries(self, limit: int = 20, include_undone: bool = True) -> list[Entry]:
        found = [e for e in self._read() if include_undone or not e.undone]
        return found[-limit:][::-1]

    # ----------------------------------------------------------- recording

    def prepare_write(self, path: str, mode: str) -> Entry:
        """Call before writing ``path``; returns a pending entry for :meth:`commit`."""
        entry = Entry(uuid.uuid4().hex[:6], "create", path, time.time(), "")
        if not os.path.lexists(path):
            entry.summary = f"created {path}"
            return entry
        size = os.lstat(path).st_size
        if mode == "append":
            entry.op, entry.old_size, entry.summary = "append", size, f"appended to {path}"
            return entry
        entry.op, entry.summary = "write", f"overwrote {path}"
        if size > MAX_BACKUP_BYTES:
            entry.note = "too large to back up: this overwrite cannot be undone"
            return entry
        blob = str(self.blobs / entry.id)
        safeio.copy_file(path, blob)
        entry.backup = blob
        return entry

    def commit(self, entry: Entry) -> Entry:
        if os.path.lexists(entry.path):
            entry.after_size, entry.after_mtime = _fingerprint(entry.path)
        return self._add(entry)

    def record(self, op: str, path: str, dest: str = "", summary: str = "") -> Entry:
        entry = Entry(uuid.uuid4().hex[:6], op, path, time.time(), summary or f"{op} {path}", dest=dest)
        probe = dest if op in ("move", "copy") else path
        if os.path.lexists(probe) and not os.path.isdir(probe):
            entry.after_size, entry.after_mtime = _fingerprint(probe)
        return self._add(entry)

    def trash(self, path: str) -> Entry:
        trashed, info = self.trash_bin.put(path)
        entry = Entry(uuid.uuid4().hex[:6], "trash", path, time.time(), f"moved {path} to the Trash",
                      trash_file=trashed, trash_info=info)
        return self._add(entry)

    # ---------------------------------------------------------------- undo

    def find(self, entry_id: str | None) -> Entry:
        candidates = [e for e in self._read() if not e.undone]
        if entry_id:
            candidates = [e for e in candidates if e.id == entry_id]
        if not candidates:
            raise UndoError("nothing to undo" if not entry_id else f"no undoable change with id {entry_id}")
        return candidates[-1]

    def _unchanged(self, entry: Entry, path: str) -> None:
        if entry.after_size < 0 or not os.path.lexists(path) or os.path.isdir(path):
            return
        if _fingerprint(path) != (entry.after_size, entry.after_mtime):
            raise UndoError(f"{path} was changed after this action; undoing it would lose those changes")

    def undo(self, entry_id: str | None = None) -> str:
        entry = self.find(entry_id)
        op, path = entry.op, entry.path
        if op == "write":
            if not entry.backup:
                raise UndoError(entry.note or "no backup was kept for this overwrite")
            self._unchanged(entry, path)
            safeio.copy_file(entry.backup, path)
            message = f"restored the previous contents of {path}"
        elif op == "append":
            self._unchanged(entry, path)
            fd = safeio.open_file(path, os.O_WRONLY)
            try:
                os.ftruncate(fd, entry.old_size)
                os.fsync(fd)
            finally:
                os.close(fd)
            message = f"removed the appended text from {path}"
        elif op in ("create", "copy"):
            target = entry.dest if op == "copy" else path
            if not os.path.lexists(target):
                raise UndoError(f"{target} no longer exists")
            if op == "create":
                self._unchanged(entry, target)
            self.trash_bin.put(target)
            message = f"moved {target} to the Trash"
        elif op == "trash":
            self.trash_bin.restore(entry.trash_file, entry.trash_info, path)
            message = f"restored {path} from the Trash"
        elif op == "move":
            if os.path.lexists(path):
                raise UndoError(f"{path} exists again; cannot move the item back")
            if not os.path.lexists(entry.dest):
                raise UndoError(f"{entry.dest} no longer exists")
            safeio.rename(entry.dest, path)
            message = f"moved {entry.dest} back to {path}"
        elif op == "mkdir":
            if os.path.isdir(path) and not stat.S_ISLNK(os.lstat(path).st_mode):
                try:
                    safeio.rmdir(path)
                except OSError as exc:
                    raise UndoError(f"{path} is not empty any more") from exc
            message = f"removed the folder {path}"
        else:
            raise UndoError(f"cannot undo '{op}'")
        with self._locked() as entries:
            for e in entries:
                if e.id == entry.id:
                    e.undone = True
        return message

    def inverse_targets(self, entry: Entry) -> list[tuple[str, str, bool]]:
        """(right, path, overwrite) the broker must allow before the agent may undo ``entry``."""
        if entry.op in ("write", "append"):
            return [("fs.write", entry.path, True)]
        if entry.op == "create":
            return [("fs.delete", entry.path, False)]
        if entry.op == "copy":
            return [("fs.delete", entry.dest, False)]
        if entry.op == "trash":
            return [("fs.write", entry.path, False)]
        if entry.op == "move":
            return [("fs.write", entry.dest, False), ("fs.write", entry.path, False)]
        return [("fs.delete", entry.path, False)]
