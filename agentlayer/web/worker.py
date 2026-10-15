"""The agent worker process behind the web dashboard.

The web server forks this worker *before* it starts any threads, then talks
to it over a UNIX socketpair with one JSON object per line. The worker is
single-threaded and is the only process that runs tools, so fork/exec for
commands stays safe, and the network-facing web process never touches the
operating system on the agent's behalf (privilege separation).
"""

from __future__ import annotations

import json
import os
import select
import socket
import time
import uuid
from contextlib import contextmanager
from typing import Iterator

from agentlayer.agent import AgentEvents
from agentlayer.config import Settings
from agentlayer.llm import LLMError
from agentlayer.memory import SessionStore
from agentlayer.session import build_llm, build_runtime, describe_llm, make_agent, session_banner_facts


class Stop(Exception):
    """The user pressed Stop."""


class Channel:
    """Line-oriented JSON messages over a socket, with a non-blocking poll."""

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buffer = b""
        self.backlog: list[dict] = []

    def send(self, message: dict) -> None:
        self.sock.sendall(json.dumps(message, ensure_ascii=False, default=str).encode() + b"\n")

    def _fill(self, timeout: float | None) -> bool:
        ready, _, _ = select.select([self.sock], [], [], timeout)
        if not ready:
            return False
        data = self.sock.recv(65536)
        if not data:
            raise EOFError
        self.buffer += data
        return True

    def _pop(self) -> dict | None:
        if b"\n" not in self.buffer:
            return None
        line, self.buffer = self.buffer.split(b"\n", 1)
        try:
            return json.loads(line)
        except ValueError:
            return None

    def receive(self, timeout: float | None = None) -> dict | None:
        """Next message (from the backlog first); None if none arrived within ``timeout``."""
        if self.backlog:
            return self.backlog.pop(0)
        while True:
            message = self._pop()
            if message is not None:
                return message
            if not self._fill(timeout):
                return None


class WorkerEvents(AgentEvents):
    def __init__(self, channel: Channel) -> None:
        self.ch = channel
        self.stop_requested = False

    def _check_stop(self) -> None:
        while True:
            message = self.ch.receive(timeout=0)
            if message is None:
                break
            if message.get("cmd") == "stop":
                self.stop_requested = True
            else:
                self.ch.backlog.append(message)
        if self.stop_requested:
            self.stop_requested = False
            raise Stop()

    @contextmanager
    def planning(self) -> Iterator[None]:
        self._check_stop()
        self.ch.send({"ev": "thinking", "on": True})
        try:
            yield
        finally:
            self.ch.send({"ev": "thinking", "on": False})

    def tool_call(self, name, args):
        self._check_stop()
        self.ch.send({"ev": "tool_call", "name": name, "args": args})

    def decision(self, name, args, decision):
        self.ch.send({"ev": "decision", "name": name, "verdict": decision.verdict.value, "reason": decision.reason,
                      "rights": decision.rights})

    def _wait(self, rid: str):
        while True:
            message = self.ch.receive(timeout=None)
            if message is None:
                continue
            if message.get("cmd") == "answer" and message.get("request") == rid:
                return message.get("value")
            if message.get("cmd") == "stop":
                return None
            self.ch.backlog.append(message)

    def confirm(self, tool, args, decision, preview):
        rid = uuid.uuid4().hex[:10]
        self.ch.send({"ev": "confirm", "rid": rid, "tool": tool.name, "args": args, "reason": decision.reason,
                      "preview": preview, "root": decision.privileged, "human": decision.requires_human})
        value = self._wait(rid)
        if value == "always" and not decision.requires_human:
            return "always"
        return value == "yes"

    def secret(self, prompt):
        rid = uuid.uuid4().hex[:10]
        self.ch.send({"ev": "secret", "rid": rid, "prompt": prompt})
        value = self._wait(rid)
        return value or None

    def result(self, name, output):
        self.ch.send({"ev": "result", "name": name, "ok": output.ok, "summary": output.summary,
                      "display": (output.display or "")[:12000], "flagged": "security_notice" in output.data})

    def output(self, text):
        self.ch.send({"ev": "output", "text": text[-4000:]})

    def reply(self, reply):
        self.ch.send({"ev": "usage", "model": reply.model, "provider": reply.provider, "key": reply.key,
                      "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens,
                      "notes": reply.notes})

    def notice(self, message):
        self.ch.send({"ev": "notice", "text": message})

    def warning(self, message):
        self.ch.send({"ev": "warning", "text": message})


def _status(rt, llm, agent) -> dict:
    return {
        "policy": rt.policy.name, "description": rt.policy.description, "model": describe_llm(llm),
        "sandbox": f"{rt.sandbox.mode} — {rt.sandbox.reason}", "workspace": str(rt.policy.workspace),
        "facts": session_banner_facts(rt, agent), "dry_run": agent.dry_run, "session": agent.session_id,
        "groups": sorted({t.group for t in agent.available}), "enabled": sorted(agent.enabled),
        "tools": len(agent.available), "pid": os.getpid(),
        "grants": [{"id": g.id, "right": g.right, "target": g.target, "lasts": g.remaining(), "reason": g.reason}
                   for g in rt.broker.active_grants()],
    }


def worker_main(sock: socket.socket, settings: Settings, llm_factory=None) -> None:
    channel = Channel(sock)
    events = WorkerEvents(channel)
    try:
        rt = build_runtime(settings)
        llm = (llm_factory or build_llm)(settings, notify=events.notice)
        agent = make_agent(rt, llm, events, settings, sessions=SessionStore())
    except Exception as exc:  # report start-up problems to the browser instead of dying silently
        channel.send({"ev": "fatal", "text": f"{type(exc).__name__}: {exc}"})
        return
    rt.audit.record("session_start", frontend="web", policy=rt.policy.name, provider=llm.provider, model=llm.model,
                    sandbox=rt.sandbox.mode, pid=os.getpid())
    channel.send({"ev": "ready", "status": _status(rt, llm, agent)})
    try:
        while True:
            try:
                message = channel.receive(timeout=None)
            except EOFError:
                break
            if message is None:
                continue
            cmd = message.get("cmd")
            if cmd == "ask":
                text = str(message.get("text", "")).strip()
                if not text:
                    continue
                channel.send({"ev": "user", "text": text})
                channel.send({"ev": "busy", "on": True})
                started = time.monotonic()
                try:
                    answer = agent.ask(text)
                    channel.send({"ev": "answer", "text": answer, "dry_run": agent.dry_run,
                                  "seconds": round(time.monotonic() - started, 1)})
                except Stop:
                    channel.send({"ev": "notice", "text": "stopped — the request was abandoned"})
                except LLMError as exc:
                    channel.send({"ev": "error", "text": f"model error: {exc}"})
                except EOFError:
                    break
                except Exception as exc:  # keep serving after unexpected errors
                    channel.send({"ev": "error", "text": f"{type(exc).__name__}: {exc}"})
                finally:
                    channel.send({"ev": "busy", "on": False})
                    channel.send({"ev": "status", "status": _status(rt, llm, agent)})
            elif cmd == "reset":
                agent.reset()
                channel.send({"ev": "cleared"})
            elif cmd == "resume":
                loaded = agent.sessions.load(message.get("session") or None)
                if loaded:
                    turns = agent.resume(*loaded)
                    history = [{"role": m["role"], "content": m.get("content")} for m in agent.messages[1:]
                               if m["role"] in ("user", "assistant") and m.get("content")]
                    channel.send({"ev": "resumed", "session": loaded[0], "turns": turns, "history": history})
                else:
                    channel.send({"ev": "notice", "text": "no saved conversation to resume"})
            elif cmd == "dryrun":
                agent.dry_run = bool(message.get("on"))
                channel.send({"ev": "notice", "text": f"dry-run mode {'ON: changes are simulated' if agent.dry_run else 'off'}"})
            elif cmd == "revoke":
                gone = rt.broker.revoke(message.get("id") or "all")
                if gone:
                    rt.audit.record("grants_revoked", ids=[g.id for g in gone], reason="revoked in the dashboard")
                channel.send({"ev": "notice", "text": f"revoked {len(gone)} temporary capability(ies)"})
            elif cmd == "stop":
                pass  # nothing running
            if cmd in ("reset", "resume", "dryrun", "revoke", "status"):
                channel.send({"ev": "status", "status": _status(rt, llm, agent)})
    finally:
        rt.broker.revoke("all")
        rt.audit.record("session_end", frontend="web", **agent.totals)
