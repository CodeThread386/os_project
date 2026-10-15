"""The web dashboard server (``agentlayer ui``).

Two processes, privilege-separated:

* this process serves the browser (HTTP on 127.0.0.1 only) and never runs
  tools; it reads system metrics, the audit log and the stores directly;
* a worker process, forked before any thread starts, owns the agent and runs
  every tool (see worker.py). They talk over a UNIX socketpair.

Access control: a random token in the first URL becomes a SameSite=Strict
HttpOnly cookie; every API call must carry it, every state-changing call must
also send it in a header (CSRF), and the Host header must be the loopback
address (DNS-rebinding defence). Responses carry a strict Content-Security-
Policy, and the page loads no third-party code.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import secrets
import signal
import socket
import sys
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from agentlayer import __version__
from agentlayer.audit import AuditLog
from agentlayer.config import Settings
from agentlayer.journal import UndoError, UndoJournal
from agentlayer.memory import MemoryStore, SessionStore
from agentlayer.monitor import Monitor
from agentlayer.policy import load_policy
from agentlayer.scheduler import ScheduleError, Scheduler
from agentlayer.tools import all_tools, exposed_tools

STATIC = Path(__file__).parent / "static"
HISTORY = 600
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
       "media-src 'self' blob:; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")



class _Server(ThreadingHTTPServer):
    daemon_threads = False  # server_close() waits for handlers, so shutdown is clean
    block_on_close = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return  # the browser went away mid-response: not an error worth a traceback
        super().handle_error(request, client_address)

class Hub:
    """Fan-out of worker events to browser connections, with a replay buffer."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clients: set[queue.Queue] = set()
        self.history: list[dict] = []
        self.status: dict = {}
        self.busy = False
        self.worker_alive = True
        self.pending: dict | None = None  # an unanswered confirmation or password prompt

    def publish(self, event: dict) -> None:
        event = {"t": time.time(), **event}
        with self.lock:
            if event.get("ev") in ("ready", "status"):
                self.status = event.get("status", {})
            if event.get("ev") == "busy":
                self.busy = bool(event.get("on"))
                if not self.busy:
                    self.pending = None
            if event.get("ev") in ("confirm", "secret"):
                self.pending = event
            if event.get("ev") == "cleared":
                self.history = []
            elif event.get("ev") not in ("thinking", "status"):
                self.history.append(event)
                del self.history[:-HISTORY]
            for q in list(self.clients):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    self.clients.discard(q)

    def subscribe(self) -> tuple[queue.Queue, list[dict]]:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self.lock:
            self.clients.add(q)
            return q, list(self.history)

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            self.clients.discard(q)

    def close(self) -> None:
        """End every open event stream (each handler thread exits promptly)."""
        with self.lock:
            for q in self.clients:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass


class WebApp:
    def __init__(self, settings: Settings, port: int = 0, llm_factory=None) -> None:
        self.settings = settings
        self.token = secrets.token_urlsafe(32)
        self.hub = Hub()
        self.monitor = Monitor()
        self.monitor_lock = threading.Lock()
        self.policy = load_policy(settings.policy, settings.workspace)
        self._voice_llm = None
        self._llm_factory = llm_factory
        # Bind first: if the port is taken we fail before any worker exists.
        self.server = _Server(("127.0.0.1", port), self._handler())
        self.port = self.server.server_address[1]
        self.sock, child = socket.socketpair()
        self.worker_pid = os.fork()  # before any thread exists: the worker forks commands safely
        if self.worker_pid == 0:
            self.sock.close()
            self.server.socket.close()  # the worker never serves HTTP
            from agentlayer.web.worker import worker_main

            code = 0
            try:
                worker_main(child, settings, llm_factory)
            except BaseException:  # noqa: BLE001 - the child must never return into the server
                code = 1
            finally:
                os._exit(code)
        child.close()
        self.send_lock = threading.Lock()
        threading.Thread(target=self._pump, daemon=True, name="worker-events").start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?token={self.token}"

    # -------------------------------------------------------------- worker

    def _pump(self) -> None:
        buffer = b""
        while True:
            try:
                data = self.sock.recv(65536)
            except OSError:
                data = b""
            if not data:
                self.hub.worker_alive = False
                self.hub.publish({"ev": "fatal", "text": "the agent worker process stopped"})
                return
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    self.hub.publish(json.loads(line))
                except ValueError:
                    continue

    def command(self, message: dict) -> None:
        with self.send_lock:
            self.sock.sendall(json.dumps(message).encode() + b"\n")

    def voice_llm(self):
        if self._voice_llm is None:
            from agentlayer.llm import GroqLLM
            from agentlayer.session import QUOTA_STATE

            if not self.settings.groq_keys:
                return None
            self._voice_llm = GroqLLM(self.settings.groq_keys, state_path=QUOTA_STATE)
        return self._voice_llm

    def shutdown(self) -> None:
        self.hub.close()
        try:
            self.sock.close()
        except OSError:
            pass
        self.server.shutdown()
        self.server.server_close()
        try:
            os.kill(self.worker_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(50):
            pid, _ = os.waitpid(self.worker_pid, os.WNOHANG)
            if pid:
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(self.worker_pid, signal.SIGKILL)
                os.waitpid(self.worker_pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass

    # ----------------------------------------------------------------- api

    def api_get(self, path: str, query: dict) -> tuple[int, object]:
        if path == "/api/status":
            return 200, {"version": __version__, "worker": self.hub.status, "busy": self.hub.busy,
                         "alive": self.hub.worker_alive}
        if path == "/api/system":
            with self.monitor_lock:
                return 200, self.monitor.sample()
        if path == "/api/audit":
            n = min(500, int(query.get("n", ["150"])[0]))
            return 200, AuditLog.tail(self.policy.audit_path, n)
        if path == "/api/audit/verify":
            r = AuditLog.verify(self.policy.audit_path)
            return 200, {"ok": r.ok, "entries": r.entries, "message": r.message}
        if path == "/api/matrix":
            from agentlayer.ui import matrix_data

            data = matrix_data(self.policy)
            data["temporary"] = self.hub.status.get("grants", [])
            return 200, data
        if path == "/api/memory":
            return 200, [vars(f) | {"when": f.when} for f in reversed(MemoryStore().all())]
        if path == "/api/sessions":
            return 200, [vars(s) for s in SessionStore().list(30)]
        if path == "/api/undo":
            return 200, [vars(e) | {"when": e.when} for e in UndoJournal().entries(40)]
        if path == "/api/tasks":
            return 200, [vars(t) for t in Scheduler().list()]
        if path == "/api/tools":
            exposed = {t.name for t in exposed_tools(self.policy)}
            return 200, [{"name": t.name, "group": t.group, "rights": t.rights, "available": t.name in exposed,
                          "description": t.description, "syscalls": t.syscalls} for t in all_tools()]
        if path == "/api/quota":
            llm = self.voice_llm()
            if llm is None:
                return 200, {"keys": 0, "groups": [], "rows": []}
            groups = llm.pool.groups()
            return 200, {"keys": len(llm.pool.keys), "groups": [[k.label for k in g] for g in groups.values()],
                         "rows": llm.pool.status(llm.models[:3])}
        return 404, {"error": "not found"}

    def api_post(self, path: str, body: dict, raw: bytes, content_type: str) -> tuple[int, object]:
        if path == "/api/ask":
            text = str(body.get("text", "")).strip()
            if not text:
                return 400, {"error": "empty request"}
            if self.hub.busy:
                return 409, {"error": "the agent is still working on the previous request"}
            self.command({"cmd": "ask", "text": text[:8000]})
            return 202, {"ok": True}
        if path == "/api/answer":
            self.hub.pending = None
            self.command({"cmd": "answer", "request": str(body.get("request", "")), "value": body.get("value")})
            return 200, {"ok": True}
        if path in ("/api/stop", "/api/reset"):
            self.command({"cmd": path.rsplit("/", 1)[1]})
            return 200, {"ok": True}
        if path == "/api/resume":
            self.command({"cmd": "resume", "session": body.get("session")})
            return 200, {"ok": True}
        if path == "/api/dryrun":
            self.command({"cmd": "dryrun", "on": bool(body.get("on"))})
            return 200, {"ok": True}
        if path == "/api/revoke":
            self.command({"cmd": "revoke", "id": body.get("id") or "all"})
            return 200, {"ok": True}
        if path == "/api/memory/add":
            fact = MemoryStore().add(str(body.get("text", "")))
            return 200, vars(fact)
        if path == "/api/memory/forget":
            fact = MemoryStore().forget(str(body.get("id", "")))
            return (200, {"ok": True}) if fact else (404, {"error": "no such note"})
        if path == "/api/undo":
            try:
                message = UndoJournal().undo(body.get("id") or None)
            except UndoError as exc:
                return 409, {"error": str(exc)}
            AuditLog(self.policy.audit_path).record("undo", by="user (dashboard)", result=message)
            return 200, {"ok": True, "message": message}
        if path == "/api/tasks/cancel":
            try:
                Scheduler().cancel(str(body.get("name", "")))
            except ScheduleError as exc:
                return 404, {"error": str(exc)}
            return 200, {"ok": True}
        if path == "/api/voice":
            llm = self.voice_llm()
            if llm is None:
                return 503, {"error": "voice needs Groq keys (Whisper)"}
            if not raw or len(raw) > 25 * 1024 * 1024:
                return 400, {"error": "no audio, or more than 25 MB"}
            ext = "webm" if "webm" in content_type else "ogg" if "ogg" in content_type else "wav"
            try:
                text = llm.transcribe(raw, f"speech.{ext}")
            except Exception as exc:  # report, never crash the server
                return 502, {"error": str(exc)}
            return 200, {"text": text}
        return 404, {"error": "not found"}

    # ------------------------------------------------------------- handler

    def _handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AgentLayer"
            sys_version = ""

            def log_message(self, fmt, *args):  # quiet
                pass

            def _headers(self, status: int, ctype: str, length: int | None = None, extra: dict | None = None):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                if length is not None:
                    self.send_header("Content-Length", str(length))
                self.send_header("Content-Security-Policy", CSP)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()

            def _json(self, status: int, payload) -> None:
                data = json.dumps(payload, ensure_ascii=False, default=str).encode()
                self._headers(status, "application/json; charset=utf-8", len(data))
                self.wfile.write(data)

            def _host_ok(self) -> bool:
                host = (self.headers.get("Host") or "").lower()
                return host in (f"127.0.0.1:{app.port}", f"localhost:{app.port}")

            def _token_ok(self, header_required: bool) -> bool:
                cookie = SimpleCookie(self.headers.get("Cookie") or "")
                in_cookie = cookie.get("agentlayer_token")
                cookie_ok = in_cookie is not None and secrets.compare_digest(in_cookie.value, app.token)
                header = self.headers.get("X-AgentLayer-Token") or ""
                header_ok = bool(header) and secrets.compare_digest(header, app.token)
                return (cookie_ok and header_ok) if header_required else (cookie_ok or header_ok)

            def do_GET(self):  # noqa: N802
                if not self._host_ok():
                    return self._json(403, {"error": "bad host"})
                url = urlsplit(self.path)
                query = parse_qs(url.query)
                if url.path == "/" and "token" in query:
                    if not secrets.compare_digest(query["token"][0], app.token):
                        return self._json(403, {"error": "bad token"})
                    self._headers(303, "text/plain", 0, {
                        "Location": "/",
                        "Set-Cookie": f"agentlayer_token={app.token}; HttpOnly; SameSite=Strict; Path=/"})
                    return None
                if url.path in ("/", "/index.html", "/app.js", "/app.css", "/icon.svg"):
                    if not self._token_ok(False):
                        return self._json(403, {"error": "open the link printed by `agentlayer ui`"})
                    name = "index.html" if url.path in ("/", "/index.html") else url.path.lstrip("/")
                    data = (STATIC / name).read_bytes()
                    if name == "index.html":  # the page needs the token for its POST header
                        data = data.replace(b"__TOKEN__", app.token.encode())
                    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
                    self._headers(200, ctype + ("; charset=utf-8" if ctype.startswith("text") or "javascript" in ctype else ""),
                                  len(data))
                    self.wfile.write(data)
                    return None
                if not self._token_ok(False):
                    return self._json(403, {"error": "unauthorised"})
                if url.path == "/api/events":
                    return self._events()
                try:
                    status, payload = app.api_get(url.path, query)
                except Exception as exc:  # an API failure must not kill the server
                    status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
                return self._json(status, payload)

            def do_POST(self):  # noqa: N802
                if not self._host_ok():
                    return self._json(403, {"error": "bad host"})
                if not self._token_ok(True):
                    return self._json(403, {"error": "unauthorised"})
                length = int(self.headers.get("Content-Length") or 0)
                if length > 30 * 1024 * 1024:
                    return self._json(413, {"error": "too large"})
                raw = self.rfile.read(length) if length else b""
                ctype = self.headers.get("Content-Type") or ""
                body = {}
                if "json" in ctype and raw:
                    try:
                        body = json.loads(raw)
                    except ValueError:
                        return self._json(400, {"error": "bad json"})
                try:
                    status, payload = app.api_post(urlsplit(self.path).path, body, raw, ctype)
                except Exception as exc:
                    status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
                return self._json(status, payload)

            def _events(self):
                q, history = app.hub.subscribe()
                try:
                    self._headers(200, "text/event-stream", None, {"Connection": "keep-alive"})
                    hello = {"ev": "hello", "status": app.hub.status, "busy": app.hub.busy, "history": history,
                             "pending": app.hub.pending}
                    self.wfile.write(f"data: {json.dumps(hello, default=str)}\n\n".encode())
                    self.wfile.flush()
                    while True:
                        try:
                            event = q.get(timeout=15)
                            if event is None:  # the server is shutting down
                                break
                            self.wfile.write(f"data: {json.dumps(event, default=str)}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    app.hub.unsubscribe(q)

        return Handler

    def serve_forever(self) -> None:
        self.server.serve_forever(poll_interval=0.5)

