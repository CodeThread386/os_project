"""Desktop integration: applications, windows, notifications, clipboard, screenshots.

Applications are found through freedesktop ``.desktop`` entries and started
detached with ``gtk-launch``/``xdg-open``. Window management uses KWin's
scripting interface on the session D-Bus (KDE Plasma, Wayland or X11): a tiny
JavaScript program is loaded into the compositor, acts on its window list,
and reports back by calling a method on a temporary D-Bus name that
AgentLayer owns for the duration of the call. Notifications use
``org.freedesktop.Notifications``; the clipboard uses wl-clipboard (Wayland)
or xclip (X11); screenshots use Spectacle (KDE), grim or gnome-screenshot.

Simulated keyboard and mouse input is deliberately not offered: Wayland
compositors do not let ordinary clients inject input into other windows.

GUI programs are started with a copy of the user's environment from which
secrets (API keys, tokens, passwords) have been removed.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

_SECRET_ENV = re.compile(r"(API_?KEY|TOKEN|SECRET|PASSW|CREDENTIAL|^GROQ|^AGENTLAYER_|^OPENAI|^ANTHROPIC|^AWS_)", re.I)


class DesktopError(Exception):
    pass


def session_env() -> dict[str, str]:
    """The user's environment for GUI apps, minus anything that looks like a secret."""
    return {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}


def graphical_session() -> str | None:
    """Why the desktop is unusable, or None if it is usable."""
    if not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
        return "no graphical session (WAYLAND_DISPLAY/DISPLAY are not set)"
    if not os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return "no session D-Bus (DBUS_SESSION_BUS_ADDRESS is not set)"
    return None


def _require_session() -> None:
    problem = graphical_session()
    if problem:
        raise DesktopError(problem)


def _detached(argv: list[str]) -> None:
    """Start a GUI program in its own session, fully detached from the agent.

    ``setsid -f`` forks once more and exits at once, so the program is
    re-parented to the session's subreaper and never becomes our zombie.
    """
    _require_session()
    setsid = shutil.which("setsid") or "/usr/bin/setsid"
    # No pipes: the launched app would inherit them and keep them open for its whole life.
    result = subprocess.run([setsid, "-f", *argv], env=session_env(), stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                            cwd=os.path.expanduser("~"))
    if result.returncode != 0:
        raise DesktopError(f"could not start {os.path.basename(argv[0])} (exit status {result.returncode})")


# ------------------------------------------------------------- applications


@dataclass
class AppEntry:
    id: str
    name: str
    generic: str
    keywords: str
    exec: str
    path: str


def _app_dirs() -> list[Path]:
    data_home = Path(os.environ.get("XDG_DATA_HOME") or "~/.local/share").expanduser()
    data_dirs = (os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share").split(":")
    dirs = [data_home / "applications"] + [Path(d) / "applications" for d in data_dirs if d]
    dirs += [Path("/var/lib/flatpak/exports/share/applications"), data_home / "flatpak/exports/share/applications",
             Path("/var/lib/snapd/desktop/applications")]
    seen, out = set(), []
    for d in dirs:
        if d not in seen and d.is_dir():
            seen.add(d)
            out.append(d)
    return out


def list_apps() -> list[AppEntry]:
    apps: dict[str, AppEntry] = {}
    for directory in _app_dirs():
        for file in sorted(directory.rglob("*.desktop")):
            app_id = str(file.relative_to(directory)).replace("/", "-").removesuffix(".desktop")
            if app_id in apps:
                continue  # earlier XDG directories take precedence
            parser = configparser.RawConfigParser(strict=False, interpolation=None)
            try:
                parser.read(file, encoding="utf-8")
                entry = parser["Desktop Entry"]
            except (configparser.Error, KeyError, UnicodeDecodeError):
                continue
            if entry.get("Type") != "Application" or entry.get("NoDisplay") == "true" or entry.get("Hidden") == "true":
                continue
            if not entry.get("Exec") or not entry.get("Name"):
                continue
            apps[app_id] = AppEntry(app_id, entry["Name"], entry.get("GenericName", ""), entry.get("Keywords", ""),
                                    entry["Exec"], str(file))
    return list(apps.values())


def find_apps(query: str, apps: list[AppEntry] | None = None) -> list[AppEntry]:
    """Installed applications matching ``query``, best match first."""
    q = query.strip().lower().removesuffix(".desktop")
    if not q:
        return []
    scored = []
    for app in apps if apps is not None else list_apps():
        name, app_id = app.name.lower(), app.id.lower()
        short_id = app_id.rsplit(".", 1)[-1]
        if q in (name, app_id, short_id):
            score = 100
        elif name.startswith(q) or short_id.startswith(q):
            score = 80
        elif q in name or q in app_id:
            score = 60
        elif q in app.generic.lower() or q in app.keywords.lower():
            score = 40
        else:
            continue
        scored.append((score, len(app.name), app))
    return [app for _, _, app in sorted(scored, key=lambda t: (-t[0], t[1]))]


def launch_app(app: AppEntry, targets: list[str] = ()) -> None:
    launcher = shutil.which("gtk-launch")
    if launcher:
        _detached([launcher, app.id, *targets])
        return
    import shlex

    argv = [a for a in shlex.split(app.exec) if not re.fullmatch(r"%[fFuUick]", a)]
    _detached(argv + list(targets))


def open_with_default(target: str) -> None:
    opener = shutil.which("xdg-open")
    if not opener:
        raise DesktopError("xdg-open is not installed")
    _detached([opener, target])


# ----------------------------------------------------------- notifications


def notify(title: str, body: str) -> int:
    _require_session()
    from jeepney import DBusAddress, new_method_call
    from jeepney.io.blocking import open_dbus_connection

    addr = DBusAddress("/org/freedesktop/Notifications", bus_name="org.freedesktop.Notifications",
                       interface="org.freedesktop.Notifications")
    with open_dbus_connection(bus="SESSION") as conn:
        msg = new_method_call(addr, "Notify", "susssasa{sv}i",
                              ("AgentLayer", 0, "dialog-information", title, body, [], {}, 8000))
        reply = conn.send_and_get_reply(msg, timeout=5)
    if reply.header.message_type.name == "error":
        raise DesktopError(f"notification service error: {reply.body}")
    return int(reply.body[0])


# ------------------------------------------------------------------ windows

_KWIN_WRAPPER = """
(function () {
  function report(payload) {
    callDBus("%(name)s", "/bridge", "org.agentlayer.Bridge", "report", JSON.stringify(payload));
  }
  function describe(w) {
    return {id: String(w.internalId), title: w.caption, app: w.resourceClass, pid: w.pid,
            minimized: w.minimized, active: w === workspace.activeWindow, desktop_file: w.desktopFileName};
  }
  function windows() {
    return workspace.windowList().filter(function (w) { return w.normalWindow && !w.skipTaskbar; });
  }
  try {
    %(body)s
  } catch (e) {
    report({ok: false, error: String(e)});
  }
})();
"""


def kwin_available() -> bool:
    if graphical_session():
        return False
    try:
        from jeepney.bus_messages import message_bus
        from jeepney.io.blocking import open_dbus_connection

        with open_dbus_connection(bus="SESSION") as conn:
            return bool(conn.send_and_get_reply(message_bus.NameHasOwner("org.kde.KWin"), timeout=3).body[0])
    except Exception:
        return False


def _kwin(body: str, timeout: float = 6.0):
    """Run a JavaScript snippet inside KWin and return what it reports."""
    _require_session()
    from jeepney import DBusAddress, MessageType, new_method_call, new_method_return
    from jeepney.bus_messages import message_bus
    from jeepney.io.blocking import open_dbus_connection

    tag = f"{os.getpid()}_{int(time.monotonic() * 1000) % 1_000_000}"
    name = f"org.agentlayer.bridge_{tag}"
    plugin = f"agentlayer_{tag}"
    scripting = DBusAddress("/Scripting", bus_name="org.kde.KWin", interface="org.kde.kwin.Scripting")
    with open_dbus_connection(bus="SESSION") as conn, tempfile.TemporaryDirectory(prefix="agentlayer-kwin-") as tmp:
        if conn.send_and_get_reply(message_bus.RequestName(name), timeout=3).body[0] != 1:
            raise DesktopError("could not claim a D-Bus name for the KWin bridge")
        script = Path(tmp) / "script.js"
        script.write_text(_KWIN_WRAPPER % {"name": name, "body": body})
        reply = conn.send_and_get_reply(new_method_call(scripting, "loadScript", "ss", (str(script), plugin)), timeout=5)
        if reply.header.message_type is MessageType.error:
            raise DesktopError("KWin scripting is not available (window control needs KDE Plasma)")
        try:
            conn.send_and_get_reply(new_method_call(scripting, "start"), timeout=5)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DesktopError("KWin did not answer in time")
                try:
                    msg = conn.receive(timeout=remaining)
                except TimeoutError as exc:
                    raise DesktopError("KWin did not answer in time") from exc
                if msg.header.message_type is MessageType.method_call and msg.header.fields.get(3) == "report":
                    conn.send(new_method_return(msg))
                    payload = json.loads(msg.body[0])
                    break
        finally:
            conn.send_and_get_reply(new_method_call(scripting, "unloadScript", "s", (plugin,)), timeout=5)
    if isinstance(payload, dict) and payload.get("ok") is False:
        raise DesktopError(f"KWin script error: {payload.get('error')}")
    return payload


def list_windows() -> list[dict]:
    return _kwin("report(windows().map(describe));")


WINDOW_ACTIONS = ("focus", "minimize", "maximize", "restore", "close")

_ACTION_JS = {
    "focus": "w.minimized = false; workspace.activeWindow = w;",
    "minimize": "w.minimized = true;",
    "maximize": "w.minimized = false; w.setMaximize(true, true);",
    "restore": "w.minimized = false; if (w.maximizable) { w.setMaximize(false, false); }",
    "close": "w.closeWindow();",
}


def window_action(window_id: str, action: str) -> dict:
    if action not in _ACTION_JS:
        raise DesktopError(f"unknown window action {action!r}")
    body = """
    var target = windows().filter(function (w) { return String(w.internalId) === %s; })[0];
    if (!target) { report({ok: false, error: "the window no longer exists"}); return; }
    var w = target;
    var before = describe(w);
    %s
    report({ok: true, window: before});
    """ % (json.dumps(window_id), _ACTION_JS[action])
    return _kwin(body)["window"]


def match_windows(query: str, windows: list[dict]) -> list[dict]:
    q = query.strip().lower()
    exact = [w for w in windows if w["id"].lower() == q or w["id"].strip("{}").lower().startswith(q.strip("{}"))]
    if exact and len(q.strip("{}")) >= 6:
        return exact
    return [w for w in windows if q in (w.get("title") or "").lower() or q in (w.get("app") or "").lower()]


# ---------------------------------------------------------------- clipboard


def clipboard_read(limit: int = 4000) -> str:
    _require_session()
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-paste"):
        argv = ["wl-paste", "--no-newline", "--type", "text"]
    elif shutil.which("xclip"):
        argv = ["xclip", "-selection", "clipboard", "-o"]
    else:
        raise DesktopError("no clipboard tool found (install wl-clipboard or xclip)")
    result = subprocess.run(argv, capture_output=True, timeout=5, env=session_env())
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace").strip()
        if "No selection" in err or "Nothing is copied" in err or "no suitable type" in err.lower():
            return ""
        raise DesktopError(err or "could not read the clipboard")
    return result.stdout[: limit * 4].decode(errors="replace")[:limit]


def clipboard_write(text: str) -> None:
    _require_session()
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        argv = ["wl-copy"]
    elif shutil.which("xclip"):
        argv = ["xclip", "-selection", "clipboard"]
    else:
        raise DesktopError("no clipboard tool found (install wl-clipboard or xclip)")
    result = subprocess.run(argv, input=text.encode(), capture_output=True, timeout=5, env=session_env())
    if result.returncode != 0:
        raise DesktopError(result.stderr.decode(errors="replace").strip() or "could not write the clipboard")


# -------------------------------------------------------------- screenshots

SCREENSHOT_MODES = ("screen", "monitor", "window")


def screenshot(output: Path, mode: str = "screen") -> None:
    """Capture to ``output`` (a fresh file in a private directory)."""
    _require_session()
    if shutil.which("spectacle"):
        flag = {"screen": "-f", "monitor": "-m", "window": "-a"}[mode]
        argv = ["spectacle", "-b", "-n", flag, "-o", str(output)]
    elif shutil.which("grim") and os.environ.get("WAYLAND_DISPLAY"):
        argv = ["grim", str(output)]
    elif shutil.which("gnome-screenshot"):
        argv = ["gnome-screenshot", "-f", str(output)] + (["-w"] if mode == "window" else [])
    elif shutil.which("import") and os.environ.get("DISPLAY"):
        argv = ["import", "-window", "root", str(output)]
    else:
        raise DesktopError("no screenshot tool found (spectacle, grim, gnome-screenshot or ImageMagick)")
    result = subprocess.run(argv, capture_output=True, timeout=30, env=session_env())
    deadline = time.monotonic() + 5
    while not (output.exists() and output.stat().st_size > 0) and time.monotonic() < deadline:
        time.sleep(0.1)  # some tools write the file just after exiting
    if not output.exists() or output.stat().st_size == 0:
        raise DesktopError(result.stderr.decode(errors="replace").strip()[:300] or "the screenshot tool produced no file")
