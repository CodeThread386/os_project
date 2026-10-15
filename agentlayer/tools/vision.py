"""Vision tools: let the agent see the screen or an image through a vision model.

The image is captured (or read), downscaled with ffmpeg so it costs few
tokens, and sent with the question to a vision-capable model. Only the
model's text answer reaches the main agent. Screen captures need the
gui.screen right (which asks the user), and the image is never stored.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from agentlayer import desktop, safeio
from agentlayer.broker import Requirement
from agentlayer.llm import LLMError
from agentlayer.tools.base import Clip, Tool, ToolArgError, ToolContext, ToolOutput, fmt_bytes, param, register

MAX_IMAGE_BYTES = 25 * 1024 * 1024
_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
         ".gif": "image/gif", ".bmp": "image/bmp"}


def shrink(image: bytes, width: int = 1600) -> tuple[bytes, str]:
    """Downscale to at most ``width`` pixels wide JPEG (ffmpeg), so the model reads it cheaply."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return image, "image/png"
    with tempfile.TemporaryDirectory(prefix="agentlayer-img-") as tmp:
        src, dst = Path(tmp) / "in", Path(tmp) / "out.jpg"
        src.write_bytes(image)
        result = subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(src), "-frames:v", "1",
                                 "-vf", f"scale='min({width},iw)':-2", "-q:v", "4", str(dst)],
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=60)
        if result.returncode != 0 or not dst.exists():
            return image, "image/png"
        return dst.read_bytes(), "image/jpeg"


def _ask(ctx: ToolContext, image: bytes, question: str) -> str:
    if ctx.vision is None:
        raise ToolArgError("no vision model is available in this session")
    small, mime = shrink(image)
    try:
        return ctx.vision(small, mime, question)
    except LLMError as exc:
        raise ToolArgError(str(exc)) from exc


def _look(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    with tempfile.TemporaryDirectory(prefix="agentlayer-look-") as tmp:
        shot = Path(tmp) / "screen.png"
        try:
            desktop.screenshot(shot, args["mode"])
        except desktop.DesktopError as exc:
            return ToolOutput.fail(str(exc))
        image = shot.read_bytes()
    try:
        answer = _ask(ctx, image, args["question"])
    except ToolArgError as exc:
        return ToolOutput.fail(str(exc))
    clip = Clip.of(answer, 3000)
    return ToolOutput(True, {"answer": clip.text, "captured": args["mode"],
                             "note": "the screenshot was not stored"}, f"looked at the {args['mode']}", clip.text)


register(Tool(
    name="look_at_screen",
    description="See what is on the user's screen (or the active window) and answer a question about it, "
                "e.g. read an error message or describe what is open.",
    rights=("gui.screen",),
    params={
        "question": param("string", "What to find out from the screen"),
        "mode": param("string", "What to capture", enum=list(desktop.SCREENSHOT_MODES), default="screen"),
    },
    required=("question",),
    requires=lambda a, c: [Requirement("gui.screen", op=f"look ({a['mode']})")],
    run=_look,
))


def _image_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    path = ctx.real(args["path"])
    if Path(path).suffix.lower() not in _MIME:
        raise ToolArgError(f"not an image file I can read ({', '.join(_MIME)})")
    return [Requirement("fs.read", path=path, lexical=ctx.lexical(args["path"]), op="read")]


def _describe(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    path = decision.approved().path
    fd = safeio.open_file(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size > MAX_IMAGE_BYTES:
            return ToolOutput.fail(f"the image is {fmt_bytes(size)}; the limit is {fmt_bytes(MAX_IMAGE_BYTES)}")
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
    finally:
        os.close(fd)
    try:
        answer = _ask(ctx, b"".join(chunks), args["question"])
    except ToolArgError as exc:
        return ToolOutput.fail(str(exc))
    clip = Clip.of(answer, 3000)
    return ToolOutput(True, {"image": ctx.show(path), "answer": clip.text}, f"looked at {ctx.show(path)}", clip.text)


register(Tool(
    name="describe_image",
    description="Look at an image file (png, jpg, webp, gif) and answer a question about it, e.g. describe a photo "
                "or read the text in a screenshot.",
    rights=("fs.read",),
    params={
        "path": param("string", "Image file"),
        "question": param("string", "What to find out", default="Describe this image in detail."),
    },
    required=("path",),
    requires=_image_requires,
    run=_describe,
))
