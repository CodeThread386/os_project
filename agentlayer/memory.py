"""Persistence across sessions: long-term notes and resumable conversations.

Long-term memory is a small list of facts the agent was asked to keep ("my
project lives in ~/Documents/os-lab"). The most recent ones are placed in the
system prompt of every new session; the rest can be searched. Conversations
are saved after every turn so a session can be resumed later.

Both live in the state directory with owner-only permissions and belong to
the broker's trusted computing base: the agent changes them only through the
memory tool, never through file tools. Memory can only inform the model; it
can never grant a capability.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from agentlayer.config import state_dir

MAX_FACT_CHARS = 500


def _write_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@dataclass
class Fact:
    id: str
    text: str
    created: float
    session: str = ""

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.created).strftime("%Y-%m-%d")


class MemoryStore:
    def __init__(self, path: Path | None = None, max_items: int = 200) -> None:
        self.path = Path(path) if path else state_dir() / "memory.json"
        self.max_items = max_items

    @contextmanager
    def _locked(self) -> Iterator[list[Fact]]:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock = os.open(str(self.path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            facts = self._read()
            yield facts
            _write_atomic(self.path, [asdict(f) for f in facts])
        finally:
            os.close(lock)

    def _read(self) -> list[Fact]:
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        return [Fact(**item) for item in raw if isinstance(item, dict) and "text" in item]

    def all(self) -> list[Fact]:
        return self._read()

    def add(self, text: str, session: str = "") -> Fact:
        text = " ".join(text.split())[:MAX_FACT_CHARS]
        if not text:
            raise ValueError("nothing to remember")
        with self._locked() as facts:
            for fact in facts:
                if fact.text.lower() == text.lower():
                    return fact
            fact = Fact(uuid.uuid4().hex[:6], text, time.time(), session)
            facts.append(fact)
            del facts[: max(0, len(facts) - self.max_items)]  # oldest go first
        return fact

    def forget(self, fact_id: str) -> Fact | None:
        with self._locked() as facts:
            for i, fact in enumerate(facts):
                if fact.id == fact_id:
                    return facts.pop(i)
        return None

    def clear(self) -> int:
        with self._locked() as facts:
            count = len(facts)
            facts.clear()
        return count

    def search(self, query: str, limit: int = 20) -> list[Fact]:
        facts = self._read()
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
        if not words:
            return facts[-limit:][::-1]
        scored = [(sum(w in f.text.lower() for w in words), f) for f in facts]
        return [f for score, f in sorted(scored, key=lambda x: (-x[0], -x[1].created)) if score][:limit]

    def prompt_block(self, max_chars: int = 1500) -> str:
        """The most recent facts, newest first, sized for the system prompt."""
        lines: list[str] = []
        used = 0
        for fact in reversed(self._read()):
            line = f"- [{fact.id}] {fact.text}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)


@dataclass
class SessionInfo:
    id: str
    started: float
    updated: float
    title: str
    turns: int
    policy: str


class SessionStore:
    """Saved conversations (the transcript minus the system prompt)."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else state_dir() / "sessions"

    def path(self, session_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{6,32}", session_id):
            raise ValueError(f"invalid session id {session_id!r}")
        return self.root / f"{session_id}.json"

    def save(self, session_id: str, messages: list[dict], *, policy: str, started: float) -> None:
        conversation = [m for m in messages if m.get("role") != "system"]
        users = [m["content"] for m in conversation if m.get("role") == "user"]
        if not users:
            return
        _write_atomic(self.path(session_id), {
            "id": session_id, "started": started, "updated": time.time(), "policy": policy,
            "title": " ".join(users[0].split())[:80], "turns": len(users), "messages": conversation,
        })

    def list(self, limit: int = 20) -> list[SessionInfo]:
        out = []
        for file in self.root.glob("*.json"):
            try:
                data = json.loads(file.read_text())
                out.append(SessionInfo(data["id"], data["started"], data["updated"], data["title"],
                                       data["turns"], data.get("policy", "")))
            except (OSError, ValueError, KeyError):
                continue
        return sorted(out, key=lambda s: -s.updated)[:limit]

    def load(self, session_id: str | None = None) -> tuple[str, list[dict]] | None:
        """Messages of the given session, or of the most recent one."""
        if session_id is None:
            recent = self.list(1)
            if not recent:
                return None
            session_id = recent[0].id
        try:
            data = json.loads(self.path(session_id).read_text())
        except (OSError, ValueError):
            return None
        return data["id"], data["messages"]
