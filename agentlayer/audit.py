"""Append-only, hash-chained audit log.

Every broker decision (allowed, denied, confirmed or declined) is appended
as one JSON line. Like a filesystem journal, each record is flushed and
``fsync``-ed before the agent moves on. Each record also stores the SHA-256
of the previous record, so editing or deleting any earlier line breaks the
chain, and ``agentlayer audit --verify`` reports where.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def _canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(prev: str, record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(prev.encode() + _canonical(body)).hexdigest()


def _last_line(fd: int) -> bytes:
    size = os.lseek(fd, 0, os.SEEK_END)
    if size == 0:
        return b""
    block = 4096
    pos = size
    data = b""
    while pos > 0:
        step = min(block, pos)
        pos -= step
        data = os.pread(fd, step, pos) + data
        lines = data.rstrip(b"\n").split(b"\n")
        if len(lines) > 1 or pos == 0:
            return lines[-1]
    return data


@dataclass
class VerifyResult:
    ok: bool
    entries: int
    message: str
    broken_at: int | None = None


class AuditLog:
    def __init__(self, path: Path, session: str | None = None) -> None:
        self.path = Path(path)
        self.session = session or uuid.uuid4().hex[:12]
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    def record(self, event: str, **fields) -> dict:
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC
        fd = os.open(self.path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # serialise concurrent sessions
            last = _last_line(fd)
            prev, seq = GENESIS, 0
            if last:
                try:
                    tail = json.loads(last)
                    prev, seq = tail.get("hash", GENESIS), int(tail.get("seq", 0))
                except (ValueError, TypeError):
                    prev = hashlib.sha256(last).hexdigest()
            entry = {
                "seq": seq + 1,
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "session": self.session,
                "event": event,
                **fields,
                "prev": prev,
            }
            entry["hash"] = _digest(prev, entry)
            os.write(fd, _canonical(entry) + b"\n")
            os.fsync(fd)
        finally:
            os.close(fd)  # also releases the lock
        return entry

    @staticmethod
    def read(path: Path) -> list[dict]:
        entries: list[dict] = []
        try:
            with open(path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except ValueError:
                            entries.append({"event": "corrupt", "raw": line[:200].decode(errors="replace")})
        except FileNotFoundError:
            pass
        return entries

    @staticmethod
    def tail(path: Path, n: int = 20) -> list[dict]:
        return AuditLog.read(path)[-n:]

    @staticmethod
    def verify(path: Path) -> VerifyResult:
        if not Path(path).exists():
            return VerifyResult(True, 0, "no audit log yet")
        prev = GENESIS
        count = 0
        with open(path, "rb") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    return VerifyResult(False, count, f"line {lineno} is not valid JSON", lineno)
                if entry.get("prev") != prev:
                    return VerifyResult(False, count, f"line {lineno} (seq {entry.get('seq')}): previous-hash link is broken "
                                                      "(a record before it was altered, removed or inserted)", lineno)
                if _digest(prev, entry) != entry.get("hash"):
                    return VerifyResult(False, count, f"line {lineno} (seq {entry.get('seq')}): contents do not match "
                                                      "their hash (the record was modified)", lineno)
                prev = entry["hash"]
                count += 1
        return VerifyResult(True, count, f"{count} records, hash chain intact")

