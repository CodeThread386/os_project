"""Voice input and output.

Input: the microphone is recorded through PipeWire (``pw-record``) or ALSA
(``arecord``) into a temporary 16 kHz mono WAV, and transcribed by Whisper.
Output: answers are read aloud locally by speech-dispatcher (``spd-say``).
"""

from __future__ import annotations

import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


class VoiceError(Exception):
    pass


def recorder_argv(path: str) -> list[str]:
    if shutil.which("pw-record"):
        return ["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16", path]
    if shutil.which("arecord"):
        return ["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", path]
    raise VoiceError("no recorder found (pw-record or arecord)")


def record(max_seconds: float = 30.0, stop_on_enter: bool = True) -> bytes:
    """Record from the default microphone until Enter is pressed or the time is up."""
    with tempfile.TemporaryDirectory(prefix="agentlayer-voice-") as tmp:
        path = str(Path(tmp) / "speech.wav")
        proc = subprocess.Popen(recorder_argv(path), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + max_seconds
        try:
            while time.monotonic() < deadline and proc.poll() is None:
                if stop_on_enter and sys.stdin.isatty():
                    ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                    if ready:
                        sys.stdin.readline()
                        break
                else:
                    time.sleep(0.2)
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)  # lets the recorder finish the WAV header
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        data = Path(path).read_bytes() if os.path.exists(path) else b""
    if len(data) < 16_000:  # under half a second of audio
        raise VoiceError("nothing was recorded (check the microphone)")
    return data


def speakable(text: str, limit: int = 600) -> str:
    text = re.sub(r"```.*?```", " (code omitted) ", text, flags=re.S)
    text = re.sub(r"^\s*\|.*\|\s*$", "", text, flags=re.M)  # tables do not read well
    text = re.sub(r"[`*_#>]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        cut = text[:limit]
        text = cut[: cut.rfind(". ") + 1] or cut
    return text


def speak(text: str) -> bool:
    """Read text aloud without blocking. Returns False if no speech engine exists."""
    engine = shutil.which("spd-say") or shutil.which("espeak-ng") or shutil.which("espeak")
    if not engine or not text.strip():
        return False
    setsid = shutil.which("setsid") or "/usr/bin/setsid"  # fully detached: never our zombie
    subprocess.run([setsid, "-f", engine, speakable(text)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=10)
    return True
