"""Privilege-separated broker over a Unix domain socket.

This module demonstrates a fundamental OS security principle: the LLM
("planner") and the system-call executor ("executor") run as separate
processes communicating over a Unix domain socket.  Even if the planner
process is compromised, it cannot touch the OS directly — it can only send
*requests* that the executor validates against the policy.

Architecture
------------

    ┌─────────────────────┐      Unix domain socket      ┌──────────────────────┐
    │  Planner process    │  ──── JSON request ──────►  │  Executor process    │
    │  (LLM client)       │  ◄─── JSON response ──────  │  (policy enforcer)   │
    │  no direct syscalls │                              │  fork/execve/rlimit  │
    └─────────────────────┘                              └──────────────────────┘

This mirrors how capability-based microkernels (L4, seL4) and real-world
security designs (systemd, Capsicum, pledge) structure privilege separation.

Usage
-----
    # In the executor process (fork before calling this):
    broker = PrivBroker(policy, workspace)
    broker.serve("/tmp/agentlayer-privbroker.sock")

    # In the planner process:
    result = PrivBroker.send("/tmp/agentlayer-privbroker.sock", {
        "action": "fs.read",
        "path": "/home/user/workspace/notes.txt",
    })
"""

from __future__ import annotations

import json
import os
import select
import shutil
import socket
import time
from dataclasses import dataclass
from pathlib import Path


# ─────────────────────────────────────────── protocol ──────────────────────


# All requests and responses are newline-delimited JSON (JSON-lines).
# Request schema:
#   {"action": "<right>", "path"?: "...", "program"?: "...", "url"?: "..."}
# Response schema:
#   {"ok": true|false, "verdict": "allow"|"deny", "reason": "..."}
#
# The executor only understands a small, read-only subset of rights for this
# demo: fs.read, exec (dry-run), net.read.  Writes are always denied so the
# demo stays safe without needing a full broker instance.

ALLOWED_ACTIONS = frozenset(["fs.read", "exec", "net.read", "sys.read"])

_RECV_SIZE = 65536
_TIMEOUT = 5.0   # seconds for send/recv


# ─────────────────────────────────────────── executor ──────────────────────


@dataclass
class PrivBroker:
    """Minimal privilege-separated broker.

    This is intentionally simple — the goal is to make the concept visible,
    not to replace the full CapabilityBroker.
    """

    workspace: str
    allowed_programs: list[str]
    allowed_paths: list[str]   # prefix-allowed read paths (e.g. workspace)
    allowed_hosts: list[str]   # suffix-matched hosts

    # ───────────── executor side ────────────────────────────────────────────

    def _evaluate(self, req: dict) -> dict:
        """Check one request; return a verdict dict."""
        action = req.get("action", "")
        if action not in ALLOWED_ACTIONS:
            return {"ok": False, "verdict": "deny",
                    "reason": f"unknown action {action!r}; allowed: {', '.join(sorted(ALLOWED_ACTIONS))}"}

        if action == "fs.read":
            path = req.get("path", "")
            if not path:
                return {"ok": False, "verdict": "deny", "reason": "fs.read requires 'path'"}
            real = os.path.realpath(path)
            if any(real.startswith(allowed) for allowed in self.allowed_paths):
                return {"ok": True, "verdict": "allow",
                        "reason": f"fs.read granted: {real} is within an allowed path prefix"}
            return {"ok": False, "verdict": "deny",
                    "reason": f"fs.read denied: {real} is outside allowed paths ({', '.join(self.allowed_paths)})"}

        if action == "exec":
            prog = req.get("program", "")
            if not prog:
                return {"ok": False, "verdict": "deny", "reason": "exec requires 'program'"}
            basename = os.path.basename(prog)
            if basename in self.allowed_programs:
                return {"ok": True, "verdict": "allow",
                        "reason": f"exec granted (dry-run): '{basename}' is on the allow-list"}
            return {"ok": False, "verdict": "deny",
                    "reason": f"exec denied: '{basename}' is not on the allow-list "
                               f"({', '.join(self.allowed_programs)})"}

        if action == "net.read":
            url = req.get("url", "")
            if not url:
                return {"ok": False, "verdict": "deny", "reason": "net.read requires 'url'"}
            host = url.lower().removeprefix("https://").removeprefix("http://").split("/")[0]
            if any(host == h or host.endswith("." + h) for h in self.allowed_hosts):
                return {"ok": True, "verdict": "allow",
                        "reason": f"net.read granted: {host} matches the host allow-list"}
            return {"ok": False, "verdict": "deny",
                    "reason": f"net.read denied: {host} is not in the host allow-list "
                               f"({', '.join(self.allowed_hosts)})"}

        if action == "sys.read":
            return {"ok": True, "verdict": "allow", "reason": "sys.read always granted in demo"}

        return {"ok": False, "verdict": "deny", "reason": "internal error"}

    def _handle(self, conn: socket.socket) -> None:
        """Serve one client connection until it closes."""
        buf = b""
        while True:
            try:
                ready, _, _ = select.select([conn], [], [], _TIMEOUT)
            except OSError:
                break
            if not ready:
                break
            try:
                chunk = conn.recv(_RECV_SIZE)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError as exc:
                    resp = {"ok": False, "verdict": "deny", "reason": f"invalid JSON: {exc}"}
                else:
                    resp = self._evaluate(req)
                try:
                    conn.sendall(json.dumps(resp).encode() + b"\n")
                except OSError:
                    return

    def serve(self, sock_path: str, *, max_requests: int = 50) -> None:
        """Listen on ``sock_path`` and handle up to ``max_requests`` connections.

        Call this in the forked executor process.  It exits cleanly when the
        socket is closed or after ``max_requests`` have been answered.
        """
        path = Path(sock_path)
        path.unlink(missing_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(sock_path)
        os.chmod(sock_path, 0o600)   # owner-only: only our own process connects
        srv.listen(4)
        srv.settimeout(_TIMEOUT)
        served = 0
        try:
            while served < max_requests:
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    break
                try:
                    self._handle(conn)
                finally:
                    conn.close()
                served += 1
        finally:
            srv.close()
            path.unlink(missing_ok=True)

    # ───────────── planner side (class method) ───────────────────────────────

    @staticmethod
    def send(sock_path: str, request: dict, *, timeout: float = _TIMEOUT) -> dict:
        """Send one request to the executor and return its verdict.

        This runs in the planner process.  It never touches the OS for the
        requested action — it just asks the executor to do so.
        """
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(timeout)
        try:
            client.connect(sock_path)
            client.sendall(json.dumps(request).encode() + b"\n")
            buf = b""
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("executor did not respond in time")
                ready, _, _ = select.select([client], [], [], remaining)
                if not ready:
                    raise TimeoutError("executor did not respond in time")
                chunk = client.recv(_RECV_SIZE)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    line = buf.split(b"\n")[0]
                    return json.loads(line)
        finally:
            client.close()
        raise ConnectionError("executor closed the connection without a response")


# ─────────────────────────── demo helper ────────────────────────────────────


def make_demo_broker(workspace: str) -> PrivBroker:
    """Return a PrivBroker configured for the typical demo/viva scenario."""
    return PrivBroker(
        workspace=workspace,
        allowed_programs=["ls", "cat", "echo", "wc", "grep", "find", "python3", "git"],
        allowed_paths=[workspace, "/usr", "/etc/os-release", "/proc/version"],
        allowed_hosts=["pypi.org", "python.org", "github.com", "api.github.com"],
    )


DEMO_REQUESTS = [
    {"action": "fs.read",  "path": "{workspace}/notes.txt",
     "label": "read a workspace file (should ALLOW)"},
    {"action": "fs.read",  "path": "/etc/shadow",
     "label": "read /etc/shadow (should DENY)"},
    {"action": "exec",     "program": "ls",
     "label": "run ls (should ALLOW)"},
    {"action": "exec",     "program": "nc",
     "label": "run netcat (should DENY)"},
    {"action": "net.read", "url": "https://pypi.org/pypi/requests/json",
     "label": "fetch from pypi.org (should ALLOW)"},
    {"action": "net.read", "url": "https://malware.example.com/payload",
     "label": "fetch from malware.example.com (should DENY)"},
    {"action": "exec",     "program": "/bin/bash",
     "label": "run bash (should DENY — path not basename)"},
]
