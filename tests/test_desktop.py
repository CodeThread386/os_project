"""Desktop tools against a fake desktop backend (nothing appears on the real screen)."""

from pathlib import Path

import pytest

from agentlayer import desktop
from agentlayer.broker import Verdict
from agentlayer.tools import REGISTRY, prepare_args

from .conftest import make_env
from .test_phase5 import GUI_ADMIN

APPS = [
    desktop.AppEntry("org.example.editor", "Editor", "Text Editor", "code;text;", "editor %F", "/x/editor.desktop"),
    desktop.AppEntry("firefox", "Firefox", "Web Browser", "internet;", "firefox %u", "/x/firefox.desktop"),
    desktop.AppEntry("org.kde.dolphin", "Dolphin", "File Manager", "files;", "dolphin %u", "/x/dolphin.desktop"),
]
WINDOWS = [
    {"id": "{11111111-aaaa}", "title": "notes.txt — Editor", "app": "editor", "pid": 999991, "minimized": False,
     "active": True, "desktop_file": "org.example.editor"},
    {"id": "{22222222-bbbb}", "title": "Mozilla Firefox", "app": "firefox", "pid": 999992, "minimized": False,
     "active": False, "desktop_file": "firefox"},
    {"id": "{33333333-cccc}", "title": "Firefox Settings", "app": "firefox", "pid": 999992, "minimized": True,
     "active": False, "desktop_file": "firefox"},
]


@pytest.fixture
def fake(monkeypatch):
    log = []
    monkeypatch.setattr(desktop, "list_apps", lambda: APPS)
    monkeypatch.setattr(desktop, "launch_app", lambda app, targets=(): log.append(("launch", app.id, list(targets))))
    monkeypatch.setattr(desktop, "open_with_default", lambda target: log.append(("open", target)))
    monkeypatch.setattr(desktop, "notify", lambda title, body: log.append(("notify", title)) or 1)
    monkeypatch.setattr(desktop, "list_windows", lambda: WINDOWS)
    monkeypatch.setattr(desktop, "window_action", lambda wid, action: log.append((action, wid)) or {})
    monkeypatch.setattr(desktop, "clipboard_read", lambda limit=4000: "copied text")
    monkeypatch.setattr(desktop, "clipboard_write", lambda text: log.append(("copy", text)))

    def shot(output: Path, mode="screen"):
        output.write_bytes(b"\x89PNG fake " + mode.encode())

    monkeypatch.setattr(desktop, "screenshot", shot)
    return log


def run(env, tool, expect=Verdict.ALLOW, **args):
    t = REGISTRY[tool]
    prepared = prepare_args(t, args)
    d = env.broker.authorize(t, prepared, env.ctx)
    assert d.verdict is expect, d.reason
    return t.run(prepared, env.ctx, d)


def test_find_apps_ranking():
    assert desktop.find_apps("firefox", APPS)[0].id == "firefox"
    assert desktop.find_apps("file manager", APPS)[0].id == "org.kde.dolphin"
    assert desktop.find_apps("dolph", APPS)[0].id == "org.kde.dolphin"
    assert desktop.find_apps("nothing-like-this", APPS) == []


def test_session_env_strips_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", "gsk_x")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    monkeypatch.setenv("MY_PASSWORD", "p")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    env = desktop.session_env()
    assert "WAYLAND_DISPLAY" in env and not {"GROQ_API_KEYS", "GITHUB_TOKEN", "MY_PASSWORD"} & set(env)


def test_match_windows():
    assert [w["id"] for w in desktop.match_windows("editor", WINDOWS)] == ["{11111111-aaaa}"]
    assert len(desktop.match_windows("firefox", WINDOWS)) == 2
    assert desktop.match_windows("22222222", WINDOWS)[0]["title"] == "Mozilla Firefox"


def test_open_app(tmp_path, fake):
    env = make_env(tmp_path, extra=GUI_ADMIN)
    out = run(env, "open_app", name="editor")
    assert out.ok and fake[-1] == ("launch", "org.example.editor", [])
    run(env, "open_app", expect=Verdict.CONFIRM, name="firefox")  # not on this policy's allow-list
    t = REGISTRY["open_app"]
    d = env.broker.authorize(t, prepare_args(t, {"name": "photoshop"}), env.ctx)
    assert d.verdict is Verdict.DENY and "no installed application" in d.reason


def test_open_path_rules(tmp_path, fake):
    env = make_env(tmp_path, extra=GUI_ADMIN)
    (env.ws / "report.pdf").write_text("pdf")
    script = env.ws / "run.sh"
    script.write_text("#!/bin/sh\nrm -rf ~\n")
    script.chmod(0o755)
    assert run(env, "open_path", target="report.pdf").ok and fake[-1][0] == "open"
    t = REGISTRY["open_path"]
    d = env.broker.authorize(t, prepare_args(t, {"target": "run.sh"}), env.ctx)
    assert d.verdict is Verdict.DENY and "might run it" in d.reason
    run(env, "open_path", target="run.sh", app="editor")  # opening it in an editor is fine
    assert fake[-1] == ("launch", "org.example.editor", [str(script)])
    d = env.broker.authorize(t, prepare_args(t, {"target": str(env.root / "home/.ssh/id_rsa")}), env.ctx)
    assert d.verdict is Verdict.DENY
    d = env.broker.authorize(t, prepare_args(t, {"target": "http://127.0.0.1:8080/admin"}), env.ctx)
    assert d.verdict is Verdict.DENY and "private or local" in d.reason


def test_windows_tool(tmp_path, fake):
    env = make_env(tmp_path, extra=GUI_ADMIN)
    out = run(env, "windows", action="list")
    assert out.data["count"] == 3 and "Mozilla Firefox" in out.data["windows"]
    run(env, "windows", action="focus", window="editor")
    assert fake[-1] == ("focus", "{11111111-aaaa}")
    t = REGISTRY["windows"]
    d = env.broker.authorize(t, prepare_args(t, {"action": "minimize", "window": "firefox"}), env.ctx)
    assert d.verdict is Verdict.DENY and "2 windows match" in d.reason
    run(env, "windows", expect=Verdict.CONFIRM, action="close", window="Mozilla")


def test_screenshot_saved_into_workspace(tmp_path, fake):
    env = make_env(tmp_path, extra=GUI_ADMIN)
    out = run(env, "screenshot", expect=Verdict.CONFIRM, mode="window", path="shots/one.png")
    assert out.ok and (env.ws / "shots/one.png").read_bytes().endswith(b"window")
    t = REGISTRY["screenshot"]
    d = env.broker.authorize(t, prepare_args(t, {"path": str(env.root / "elsewhere.png")}), env.ctx)
    assert d.verdict is Verdict.DENY


def test_clipboard_and_notify(tmp_path, fake):
    env = make_env(tmp_path, extra=GUI_ADMIN)
    assert run(env, "clipboard", expect=Verdict.CONFIRM, action="read").data["text"] == "copied text"
    run(env, "clipboard", action="write", text="hello")
    assert fake[-1] == ("copy", "hello")
    run(env, "notify", title="Build finished", message="all tests passed")
    assert fake[-1] == ("notify", "Build finished")


def test_desktop_errors_become_tool_failures(tmp_path, monkeypatch):
    env = make_env(tmp_path, extra=GUI_ADMIN)

    def broken():
        raise desktop.DesktopError("no graphical session")

    monkeypatch.setattr(desktop, "list_windows", broken)
    out = run(env, "windows", action="list")
    assert not out.ok and "no graphical session" in out.data["error"]


@pytest.mark.skipif(desktop.graphical_session() is not None or not desktop.kwin_available(),
                    reason="needs a live KDE Plasma session")
def test_live_kwin_window_list():
    windows = desktop.list_windows()
    assert isinstance(windows, list) and all({"id", "title", "pid"} <= set(w) for w in windows)
