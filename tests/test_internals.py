"""OS-internals tools, inotify, syscall tracing, the monitor, vision and voice helpers."""

import os
import shutil
import struct
import subprocess
import zlib

import pytest

from agentlayer import inotify, voice
from agentlayer.broker import Verdict
from agentlayer.monitor import Monitor
from agentlayer.tools import REGISTRY, prepare_args
from agentlayer.tools.internals import PROCESS_TOPICS, SYSTEM_TOPICS, _addr
from agentlayer.tools.vision import shrink

from .conftest import make_env, needs_bwrap


def run(env, tool, expect=Verdict.ALLOW, **args):
    t = REGISTRY[tool]
    prepared = prepare_args(t, args)
    d = env.broker.authorize(t, prepared, env.ctx)
    assert d.verdict is expect, d.reason
    return t.run(prepared, env.ctx, d)


@pytest.mark.parametrize("topic", PROCESS_TOPICS)
def test_process_topics_on_ourselves(env, topic):
    out = run(env, "os_internals", topic=topic, pid=os.getpid())
    assert out.ok, out.data


@pytest.mark.parametrize("topic", SYSTEM_TOPICS)
def test_system_topics(env, topic):
    out = run(env, "os_internals", topic=topic)
    assert out.ok, out.data


def test_memory_map_has_the_classic_regions(env):
    out = run(env, "os_internals", topic="memory_map", pid=os.getpid())
    layout = " ".join(out.data["layout"])
    assert "heap" in layout and "shared libraries" in layout and "stack" in layout


def test_namespaces_show_sharing(env):
    out = run(env, "os_internals", topic="namespaces", pid=os.getpid())
    assert all("shared with AgentLayer" in v for v in out.data["namespaces"].values())


def test_process_topic_needs_pid(env):
    t = REGISTRY["os_internals"]
    d = env.broker.authorize(t, prepare_args(t, {"topic": "open_files"}), env.ctx)
    assert d.verdict is Verdict.DENY and "needs a pid" in d.reason


def test_socket_address_decoding():
    assert _addr("0100007F:1F90") == "127.0.0.1:8080"
    assert _addr("00000000000000000000000001000000:0016") == "[::1]:22"


def test_network_connections_sees_our_listener(env):
    srv = __import__("socket").socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]
    try:
        out = run(env, "network_connections", state="listening", protocol="tcp")
        assert f"127.0.0.1:{port}" in out.data["table"] and str(os.getpid()) in out.data["table"]
    finally:
        srv.close()


def test_hardware_and_services(env):
    assert run(env, "hardware_info", topic="summary").ok
    assert run(env, "hardware_info", topic="cpu").data["cpu"]["logical_cpus"] == os.cpu_count()
    if shutil.which("systemctl"):
        assert run(env, "services", action="failed").ok
        t = REGISTRY["services"]
        d = env.broker.authorize(t, prepare_args(t, {"action": "logs"}), env.ctx)
        assert d.verdict is Verdict.DENY and "sys.logs" in d.reason  # logs need their own right


def test_inotify_reports_a_new_file(tmp_path):
    proc = subprocess.Popen(["sh", "-c", f"sleep 0.3; echo hi > {tmp_path}/new.txt"])
    try:
        events = inotify.watch(str(tmp_path), 5)
    finally:
        proc.wait()
    descriptions = " ".join(e.describe() for e in events)
    assert "created" in descriptions and "new.txt" in descriptions


def test_watch_path_tool_times_out_quietly(env):
    out = run(env, "watch_path", path=".", seconds=1)
    assert out.ok and out.data["events"] == 0


@needs_bwrap
@pytest.mark.skipif(not shutil.which("strace"), reason="strace not installed")
def test_trace_syscalls_in_sandbox(tmp_path):
    env = make_env(tmp_path, sandbox="bwrap")
    (env.ws / "a.txt").write_text("x")
    out = run(env, "trace_syscalls", command="cat a.txt")
    names = out.data["table"]
    assert out.ok and "execve" in names and "openat" in names and out.data["total_calls"] > 10
    log = run(env, "trace_syscalls", command="cat a.txt", mode="log", filter="openat")
    assert "openat" in log.data["trace"]


def test_trace_requires_allowed_program(env):
    t = REGISTRY["trace_syscalls"]
    d = env.broker.authorize(t, prepare_args(t, {"command": "curl https://example.com"}), env.ctx)
    assert d.verdict is Verdict.DENY


def test_monitor_rates():
    m = Monitor()
    m.sample()
    s = m.sample()
    assert 0 <= s["cpu"]["total"] <= 100 and len(s["cpu"]["cores"]) == os.cpu_count()
    assert s["memory"]["total"] > 0 and s["processes"]["total"] > 10


def _png(w, h):
    raw = b"".join(b"\x00" + bytes((255, 0, 0)) * w for _ in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_images_are_shrunk_for_the_vision_model():
    data, mime = shrink(_png(3000, 200))
    assert mime == "image/jpeg" and data[:2] == b"\xff\xd8"


def test_describe_image_uses_the_vision_model(env):
    (env.ws / "photo.png").write_bytes(_png(8, 8))
    seen = {}

    def vision(image, mime, question):
        seen.update(mime=mime, question=question, size=len(image))
        return "a red square"

    env.ctx.vision = vision
    out = run(env, "describe_image", path="photo.png", question="what is it?")
    assert out.data["answer"] == "a red square" and seen["question"] == "what is it?"


def test_describe_image_without_vision(env):
    (env.ws / "p.png").write_bytes(_png(4, 4))
    assert not run(env, "describe_image", path="p.png").ok


def test_voice_helpers():
    assert voice.speakable("**Done.** Here is `code` and a [link](https://x.y).") == "Done. Here is code and a link."
    assert "code omitted" in voice.speakable("Run:\n```\nls -la\n```\nthen stop.")
    assert voice.recorder_argv("/tmp/x.wav")[-1] == "/tmp/x.wav"
