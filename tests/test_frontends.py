"""Scheduler, MCP server and web dashboard."""

import json
import subprocess
import sys
import threading
import time
from http.client import HTTPConnection
from pathlib import Path

import pytest

from agentlayer.config import Settings
from agentlayer.llm import LLMReply, ToolCall
from agentlayer.scheduler import ScheduleError, Scheduler, interpret

from .conftest import BASE_POLICY

# ----------------------------------------------------------------- scheduler


@pytest.mark.parametrize("text,kind,spec", [
    ("in 20 minutes", "once", "1200"),
    ("in 2 hours", "once", "7200"),
    ("every day at 9:00", "calendar", "*-*-* 09:00:00"),
    ("every monday at 6:30pm", "calendar", "Mon *-*-* 18:30:00"),
    ("weekdays at 8am", "calendar", "Mon..Fri *-*-* 08:00:00"),
    ("every 15 minutes", "calendar", "*:0/15"),
    ("every 2 hours", "calendar", "*-*-* 0/2:00:00"),
    ("daily", "calendar", "daily"),
    ("at 23:15", "calendar", "*-*-* 23:15:00"),
])
def test_interpret(text, kind, spec):
    assert interpret(text) == (kind, spec)


@pytest.mark.parametrize("bad", ["in 10 seconds", "every 1 minutes", "every day at 25:00"])
def test_interpret_rejects(bad):
    with pytest.raises(ScheduleError):
        interpret(bad)


class FakeSystemd:
    def __init__(self):
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        out = ""
        if argv[:2] == ["systemd-analyze", "calendar"]:
            out = "  Original form: x\nNext elapse: Tue 2026-10-14 09:00:00 IST\n"
        if "show" in argv:
            out = "NextElapseUSecRealtime=Tue 2026-10-14 09:00:00 IST\nActiveState=active\n"
        return subprocess.CompletedProcess(argv, 0, out, "")


def test_scheduler_lifecycle(tmp_path):
    fake = FakeSystemd()
    s = Scheduler(tmp_path / "tasks", tmp_path / "units", run=fake)
    task = s.create("disk-report", "tell me the free disk space", "every day at 9:00", "phase5-desktop")
    service = (tmp_path / "units" / "agentlayer-task-disk-report.service").read_text()
    timer = (tmp_path / "units" / "agentlayer-task-disk-report.timer").read_text()
    assert "task-run disk-report" in service and sys.executable in service
    assert "OnCalendar=*-*-* 09:00:00" in timer and "Persistent=true" in timer
    assert ["systemctl", "--user", "enable", "--now", "agentlayer-task-disk-report.timer"] in fake.calls
    assert task.next_run.startswith("Tue")
    assert [t.name for t in s.list()] == ["disk-report"]
    with pytest.raises(ScheduleError, match="already exists"):
        s.create("disk-report", "x", "daily", "p")
    s.record_run("disk-report", "ok: 41 GB free")
    assert s.load("disk-report").runs == 1
    s.cancel("disk-report")
    assert s.list() == [] and not (tmp_path / "units" / "agentlayer-task-disk-report.timer").exists()


def test_one_off_task_uses_onactivesec(tmp_path):
    s = Scheduler(tmp_path / "t", tmp_path / "u", run=FakeSystemd())
    s.create("tea", "remind me that the tea is ready", "in 5 minutes", "p")
    assert "OnActiveSec=300s" in (tmp_path / "u" / "agentlayer-task-tea.timer").read_text()


def test_bad_task_names(tmp_path):
    s = Scheduler(tmp_path / "t", tmp_path / "u", run=FakeSystemd())
    for name in ("Bad Name", "../escape", ""):
        with pytest.raises(ScheduleError):
            s.create(name, "x", "daily", "p")


# ------------------------------------------------------------------------ MCP


@pytest.fixture
def mcp_settings(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("KEY")
    (tmp_path / "shared").mkdir()
    policy = tmp_path / "policy.toml"
    policy.write_text(BASE_POLICY.format(ws=tmp_path / "ws", state=tmp_path / "state", home=home,
                                         shared=tmp_path / "shared", confirm="true", sandbox="none", exec_extra=""))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    return Settings(provider="openai", policy=str(policy), groq_keys=[], guard=True)


def test_mcp_protocol(mcp_settings, tmp_path):
    from agentlayer.mcp import MCPServer

    server = MCPServer(mcp_settings, confirm_mode="deny")
    init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "pytest"}}})
    assert init["result"]["protocolVersion"] == "2025-06-18" and init["result"]["serverInfo"]["name"] == "agentlayer"
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    tools = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"read_file", "write_file", "run_command"} <= names and all("inputSchema" in t for t in tools)
    (tmp_path / "ws" / "hello.txt").write_text("hi from the workspace")

    def call(name, args, mid=3):
        return server.handle({"jsonrpc": "2.0", "id": mid, "method": "tools/call",
                              "params": {"name": name, "arguments": args}})["result"]

    ok = call("read_file", {"path": "hello.txt"})
    assert not ok["isError"] and "hi from the workspace" in ok["content"][0]["text"]
    denied = call("read_file", {"path": str(tmp_path / "home/.ssh/id_rsa")})
    assert denied["isError"] and "Permission denied" in denied["content"][0]["text"]
    declined = call("delete_path", {"path": "hello.txt"})  # needs confirmation; no dialog in tests
    assert declined["isError"] and (tmp_path / "ws" / "hello.txt").exists()
    missing = server.handle({"jsonrpc": "2.0", "id": 9, "method": "nope"})
    assert missing["error"]["code"] == -32601


def test_mcp_over_stdio(mcp_settings, tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"), "XDG_STATE_HOME": str(tmp_path / "xdg"),
           "AGENTLAYER_MCP_CONFIRM": "deny", "AGENTLAYER_PROVIDER": "openai"}
    proc = subprocess.Popen([sys.executable, "-m", "agentlayer", "-p", mcp_settings.policy, "mcp"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                            cwd=str(tmp_path))
    try:
        messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
                    {"jsonrpc": "2.0", "method": "notifications/initialized"},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                     "params": {"name": "system_info", "arguments": {}}}]
        out, err = proc.communicate("\n".join(json.dumps(m) for m in messages) + "\n", timeout=60)
    finally:
        proc.kill()
    replies = [json.loads(line) for line in out.splitlines() if line.strip()]
    assert [r["id"] for r in replies] == [1, 2], err
    assert replies[0]["result"]["protocolVersion"] == "2024-11-05"
    assert "kernel" in replies[1]["result"]["content"][0]["text"]


# ------------------------------------------------------------- web dashboard


class FakeLLM:
    provider, model = "fake", "scripted"

    def __init__(self, settings=None, notify=None):
        self.replies = [LLMReply("", [ToolCall("c1", "system_info", "{}")], "scripted", "fake"),
                        LLMReply("All **good**.", [], "scripted", "fake")]

    def complete(self, messages, tools):
        return self.replies.pop(0) if self.replies else LLMReply("again", [], "scripted", "fake")

    def context_budget(self):
        return 50_000


@pytest.fixture
def web(mcp_settings):
    from agentlayer.web.server import WebApp

    app = WebApp(mcp_settings, port=0, llm_factory=FakeLLM)
    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    yield app
    app.shutdown()
    thread.join(timeout=5)


def request(app, method, path, body=None, cookie=True, header=True, host=None):
    conn = HTTPConnection("127.0.0.1", app.port, timeout=15)
    headers = {"Host": host or f"127.0.0.1:{app.port}"}
    if cookie:
        headers["Cookie"] = f"agentlayer_token={app.token}"
    if header:
        headers["X-AgentLayer-Token"] = app.token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=headers)
    resp = conn.getresponse()
    return resp.status, resp.getheaders(), resp.read()


def test_web_access_control(web):
    assert request(web, "GET", "/", cookie=False, header=False)[0] == 403
    status, headers, _ = request(web, "GET", f"/?token={web.token}", cookie=False, header=False)
    assert status == 303 and "SameSite=Strict" in dict(headers)["Set-Cookie"]
    status, headers, body = request(web, "GET", "/")
    assert status == 200 and b"AgentLayer" in body and "script-src 'self'" in dict(headers)["Content-Security-Policy"]
    assert request(web, "GET", "/api/status", host="evil.example:80")[0] == 403  # DNS rebinding
    assert request(web, "POST", "/api/ask", {"text": "hi"}, header=False)[0] == 403  # CSRF: header required
    assert request(web, "GET", "/api/status", cookie=False, header=False)[0] == 403


def test_web_chat_round_trip(web):
    events = []
    done = threading.Event()
    conns = []

    def listen():
        conn = HTTPConnection("127.0.0.1", web.port, timeout=30)
        conns.append(conn)
        conn.request("GET", "/api/events", headers={"Host": f"127.0.0.1:{web.port}",
                                                     "Cookie": f"agentlayer_token={web.token}"})
        resp = conn.getresponse()
        while not done.is_set():
            try:
                line = resp.fp.readline().decode()
            except (OSError, ValueError, AttributeError):
                return
            if line.startswith("data: "):
                event = json.loads(line[6:])
                events.append(event)
                if event.get("ev") == "answer":
                    done.set()

    listener = threading.Thread(target=listen, daemon=True)
    listener.start()
    deadline = time.time() + 20
    while time.time() < deadline and not web.hub.status:
        time.sleep(0.1)
    assert web.hub.status.get("policy") == "test"
    status, _, _ = request(web, "POST", "/api/ask", {"text": "how is the system?"})
    assert status == 202
    assert done.wait(20), events
    kinds = [e["ev"] for e in events]
    assert "tool_call" in kinds and "decision" in kinds and "result" in kinds
    assert next(e for e in events if e["ev"] == "answer")["text"] == "All **good**."
    listener.join(timeout=5)  # it stops by itself after the answer
    assert not listener.is_alive()


def test_web_data_endpoints(web):
    for path in ("/api/system", "/api/matrix", "/api/audit/verify", "/api/tools", "/api/memory", "/api/undo",
                 "/api/sessions"):
        status, _, body = request(web, "GET", path)
        assert status == 200, (path, body)
    matrix = json.loads(request(web, "GET", "/api/matrix")[2])
    assert matrix["policy"] == "test" and matrix["rows"] and matrix["invariants"]
    status, _, body = request(web, "POST", "/api/memory/add", {"text": "likes dark mode"})
    assert status == 200 and json.loads(body)["text"] == "likes dark mode"


def test_static_assets_are_served(web):
    for path in ("/app.js", "/app.css", "/icon.svg"):
        status, headers, body = request(web, "GET", path)
        assert status == 200 and body, path
    assert Path(__file__).parent.parent.joinpath("agentlayer/web/static/app.js").read_text().count("innerHTML") > 0
