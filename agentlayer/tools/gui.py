"""Desktop tools: launch apps, open files/URLs, manage windows, screenshots, clipboard, notifications."""

from __future__ import annotations

import os
import stat
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from agentlayer import desktop, safeio
from agentlayer.broker import Requirement
from agentlayer.tools.base import Clip, Tool, ToolArgError, ToolContext, ToolOutput, fmt_bytes, param, register

# Opening these in their "default application" would run them.
_EXECUTABLE_SUFFIXES = {".desktop", ".sh", ".bash", ".run", ".bin", ".appimage", ".exe", ".msi", ".deb", ".rpm",
                        ".jar", ".command", ".py", ".pl"}


def _desktop_error(func):
    def wrapper(args, ctx, decision):
        try:
            return func(args, ctx, decision)
        except desktop.DesktopError as exc:
            return ToolOutput.fail(str(exc))
    return wrapper


def _resolve_app(name: str) -> desktop.AppEntry:
    matches = desktop.find_apps(name)
    if not matches:
        raise ToolArgError(f"no installed application matches '{name}'")
    best = matches[0]
    ties = [m for m in matches[1:4] if m.name.lower() == best.name.lower() and m.id != best.id]
    if ties and best.name.lower() != name.strip().lower():
        options = ", ".join(f"{m.name} ({m.id})" for m in [best, *ties])
        raise ToolArgError(f"'{name}' is ambiguous: {options}; pass the id")
    return best


# ------------------------------------------------------------------ open_app


def _open_app_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    app = _resolve_app(args["name"])
    return [Requirement("gui.launch", op="app", program=app.id, target=f"{app.name} ({app.id})")]


@_desktop_error
def _open_app(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    app = _resolve_app(args["name"])
    desktop.launch_app(app)
    return ToolOutput(True, {"launched": app.name, "id": app.id}, f"launched {app.name}")


register(Tool(
    name="open_app",
    description="Launch an installed desktop application by name (e.g. 'firefox', 'calculator', 'dolphin').",
    rights=("gui.launch",),
    params={"name": param("string", "Application name or desktop id")},
    required=("name",),
    requires=_open_app_requires,
    run=_open_app,
))


# ----------------------------------------------------------------- open_path


def _target(args: dict, ctx: ToolContext) -> tuple[str, Requirement]:
    raw = args["target"].strip()
    if "://" in raw or raw.startswith(("www.", "mailto:")):
        url = raw if "://" in raw or raw.startswith("mailto:") else "https://" + raw
        if urlsplit(url).scheme not in ("http", "https"):
            raise ToolArgError("only http(s) URLs, files and folders can be opened")
        return url, Requirement("net.read", url=url)
    path = ctx.real(raw)
    if not os.path.exists(path):
        raise ToolArgError(f"{ctx.show(path)} does not exist")
    st = os.stat(path)
    runnable = st.st_mode & 0o111 or Path(path).suffix.lower() in _EXECUTABLE_SUFFIXES
    if stat.S_ISREG(st.st_mode) and runnable and not args.get("app"):
        raise ToolArgError(f"{ctx.show(path)} is a program or script; its default handler might run it. "
                           "Name an app (e.g. app='kate') to open it for editing, or use run_command")
    return path, Requirement("fs.read", path=path, lexical=ctx.lexical(raw), op="read")


def _open_path_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    target, req = _target(args, ctx)
    reqs = [req]
    if args.get("app"):
        app = _resolve_app(args["app"])
        reqs.append(Requirement("gui.launch", op="app", program=app.id, target=f"{app.name} ({app.id})"))
    else:
        reqs.append(Requirement("gui.launch", op="open", target=ctx.show(target)))
    return reqs


@_desktop_error
def _open_path(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    target, _ = _target(args, ctx)
    if args.get("app"):
        app = _resolve_app(args["app"])
        desktop.launch_app(app, [target])
        how = app.name
    else:
        desktop.open_with_default(target)
        how = "the default application"
    shown = ctx.show(target) if not target.startswith("http") else target
    return ToolOutput(True, {"opened": shown, "with": how}, f"opened {shown} with {how}")


register(Tool(
    name="open_path",
    description="Open a file, folder or http(s) URL in a desktop application (the default one, or the named app).",
    rights=("gui.launch",),
    params={
        "target": param("string", "File/folder path or URL"),
        "app": param("string", "Optional application to open it with"),
    },
    required=("target",),
    requires=_open_path_requires,
    run=_open_path,
))


# -------------------------------------------------------------------- notify


@_desktop_error
def _notify(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    desktop.notify(args["title"][:120], args["message"][:500])
    return ToolOutput(True, {"shown": True}, f"notification shown: {args['title'][:60]}")


register(Tool(
    name="notify",
    description="Show a desktop notification to the user.",
    rights=("gui.launch",),
    params={"title": param("string", "Short title"), "message": param("string", "Body text", default="")},
    required=("title",),
    requires=lambda a, c: [Requirement("gui.launch", op="notify", target=a["title"][:60])],
    run=_notify,
))


# ------------------------------------------------------------------- windows


def _window(args: dict) -> dict:
    query = (args.get("window") or "").strip()
    if not query:
        raise ToolArgError("say which window: part of its title, its app, or its id from action=list")
    try:
        windows = desktop.list_windows()
    except desktop.DesktopError as exc:
        raise ToolArgError(str(exc)) from exc
    matches = desktop.match_windows(query, windows)
    if not matches:
        raise ToolArgError(f"no open window matches '{query}'")
    if len(matches) > 1:
        options = "; ".join(f"{w['id'][1:9]}: {w['title'][:50]}" for w in matches[:6])
        raise ToolArgError(f"{len(matches)} windows match '{query}' ({options}); be more specific or pass an id")
    return matches[0]


def _windows_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    if args["action"] == "list":
        return [Requirement("gui.windows", op="list")]
    w = _window(args)
    return [Requirement("gui.windows", op=args["action"], target=f"{w['title'][:60]} [{w['app']}]", pid=w["pid"])]


@_desktop_error
def _windows(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    if args["action"] == "list":
        wins = desktop.list_windows()
        lines = [f"{w['id'][1:9]}  {'*' if w['active'] else ' '}{'(min) ' if w['minimized'] else ''}"
                 f"{w['app']}: {w['title']}" for w in wins]
        text = "\n".join(lines) or "(no windows)"
        return ToolOutput(True, {"windows": text, "count": len(wins), "legend": "id  *=active"},
                          f"{len(wins)} open windows", text)
    w = _window(args)
    desktop.window_action(w["id"], args["action"])
    verb = {"focus": "focused", "minimize": "minimised", "maximize": "maximised", "restore": "restored",
            "close": "asked to close"}[args["action"]]
    return ToolOutput(True, {"window": w["title"], "app": w["app"], "action": args["action"]},
                      f"{verb} '{w['title'][:60]}'")


register(Tool(
    name="windows",
    description="List open windows, or focus/minimize/maximize/restore/close one (KDE Plasma). "
                "Typing and clicking into apps is not supported.",
    rights=("gui.windows",),
    params={
        "action": param("string", "What to do", enum=["list", *desktop.WINDOW_ACTIONS], default="list"),
        "window": param("string", "Part of the window title, app name, or id (not needed for list)"),
    },
    requires=_windows_requires,
    run=_windows,
))


# ---------------------------------------------------------------- screenshot


def _shot_path(args: dict, ctx: ToolContext) -> str:
    raw = args.get("path") or f"screenshots/screenshot-{time.strftime('%Y%m%d-%H%M%S')}.png"
    if not raw.lower().endswith(".png"):
        raise ToolArgError("the screenshot path must end in .png")
    return raw


def _screenshot_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    raw = _shot_path(args, ctx)
    return [Requirement("gui.screen", op=args["mode"]),
            Requirement("fs.write", path=ctx.real(raw), lexical=ctx.lexical(raw), op="write")]


@_desktop_error
def _screenshot(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    dest = decision.approved(1).path
    with tempfile.TemporaryDirectory(prefix="agentlayer-shot-") as tmp:
        shot = Path(tmp) / "shot.png"
        desktop.screenshot(shot, args["mode"])
        safeio.make_dirs(os.path.dirname(dest))
        size = safeio.copy_file(str(shot), dest)
    return ToolOutput(True, {"saved": ctx.show(dest), "bytes": size, "mode": args["mode"],
                             "note": "you cannot see the image; tell the user where it is"},
                      f"screenshot saved to {ctx.show(dest)} ({fmt_bytes(size)})")


register(Tool(
    name="screenshot",
    description="Take a screenshot (whole screen, current monitor or active window) and save it as a PNG.",
    rights=("gui.screen", "fs.write"),
    params={
        "mode": param("string", "What to capture", enum=list(desktop.SCREENSHOT_MODES), default="screen"),
        "path": param("string", "Where to save it (default screenshots/screenshot-<time>.png)"),
    },
    requires=_screenshot_requires,
    run=_screenshot,
))


# ----------------------------------------------------------------- clipboard


def _clipboard_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    if args["action"] == "write" and not args.get("text"):
        raise ToolArgError("give the text to copy")
    return [Requirement("gui.clipboard", op=args["action"])]


@_desktop_error
def _clipboard(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    if args["action"] == "write":
        desktop.clipboard_write(args["text"])
        return ToolOutput(True, {"copied_chars": len(args["text"])}, f"copied {len(args['text'])} characters")
    text = desktop.clipboard_read(4000)
    clip = Clip.of(text, 3000)
    return ToolOutput(True, {"text": clip.text, "chars": len(text)}, f"read {len(text)} characters from the clipboard")


register(Tool(
    name="clipboard",
    description="Read the clipboard, or copy text to it.",
    rights=("gui.clipboard",),
    params={
        "action": param("string", "read or write", enum=["read", "write"]),
        "text": param("string", "Text to copy (for write)"),
    },
    required=("action",),
    requires=_clipboard_requires,
    run=_clipboard,
))
