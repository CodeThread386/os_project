"""Unattended tasks on systemd user timers.

A scheduled task is a natural-language request plus a time. AgentLayer writes
a pair of systemd *user* units (a oneshot service and a timer), so the init
system starts ``agentlayer task-run NAME`` at the right moment, even after
reboots (Persistent=true catches up on missed runs). The run is headless:
anything that would need a confirmation is declined automatically and
nothing can run as root, so a task can only do what the policy allows
without asking.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from agentlayer.config import PROJECT_ROOT, state_dir

NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
_UNITS = Path(os.environ.get("XDG_CONFIG_HOME") or "~/.config").expanduser() / "systemd" / "user"
_DAYS = {"monday": "Mon", "tuesday": "Tue", "wednesday": "Wed", "thursday": "Thu", "friday": "Fri",
         "saturday": "Sat", "sunday": "Sun", "weekday": "Mon..Fri", "weekend": "Sat,Sun"}


class ScheduleError(Exception):
    pass


@dataclass
class Task:
    name: str
    request: str
    when: str  # as the user said it
    kind: str  # calendar | once
    spec: str  # OnCalendar expression or seconds for OnActiveSec
    policy: str
    created: float
    runs: int = 0
    last_run: float = 0.0
    last_status: str = ""
    next_run: str = ""


def _clock(text: str) -> str:
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text.strip())
    if not m:
        raise ScheduleError(f"cannot read the time '{text}'")
    hour, minute, half = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if half == "pm" and hour < 12:
        hour += 12
    if half == "am" and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ScheduleError(f"'{text}' is not a valid time")
    return f"{hour:02d}:{minute:02d}:00"


def interpret(when: str) -> tuple[str, str]:
    """Turn 'every day at 9am', 'in 20 minutes', 'Mon *-*-* 09:00' ... into (kind, spec)."""
    text = when.strip().lower()
    m = re.fullmatch(r"in\s+(\d+)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hours?|d|days?)", text)
    if m:
        unit = m.group(2)[0]
        seconds = int(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        if not 60 <= seconds <= 7 * 86400:
            raise ScheduleError("one-off tasks must be between 1 minute and 7 days away")
        return "once", str(seconds)
    m = re.fullmatch(r"every\s+(\d+)\s*(minutes?|mins?|hours?|h)", text)
    if m:
        n = int(m.group(1))
        if m.group(2).startswith("h"):
            return "calendar", f"*-*-* 0/{n}:00:00" if n < 24 else "daily"
        if n < 5:
            raise ScheduleError("tasks may not run more often than every 5 minutes")
        return "calendar", f"*:0/{n}"
    if text in ("hourly", "every hour"):
        return "calendar", "hourly"
    if text in ("daily", "every day"):
        return "calendar", "daily"
    if text in ("weekly", "every week"):
        return "calendar", "weekly"
    m = re.fullmatch(r"(?:every\s+)?(day|weekday|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?"
                     r"\s+(?:at\s+)?(.+)", text)
    if m:
        days = "" if m.group(1) == "day" else _DAYS[m.group(1)] + " "
        return "calendar", f"{days}*-*-* {_clock(m.group(2))}"
    m = re.fullmatch(r"(?:every day |daily )?at\s+(.+)", text)
    if m:
        return "calendar", f"*-*-* {_clock(m.group(1))}"
    return "calendar", when.strip()  # assume a systemd OnCalendar expression; validated below


class Scheduler:
    def __init__(self, root: Path | None = None, unit_dir: Path | None = None,
                 run: Callable[[list[str]], subprocess.CompletedProcess] | None = None) -> None:
        self.root = Path(root) if root else state_dir() / "tasks"
        self.units = Path(unit_dir) if unit_dir else _UNITS
        self._run = run or (lambda argv: subprocess.run(argv, capture_output=True, text=True, timeout=30))

    # --------------------------------------------------------------- helpers

    def _systemctl(self, *args: str) -> subprocess.CompletedProcess:
        return self._run(["systemctl", "--user", *args])

    def _file(self, name: str) -> Path:
        if not NAME.fullmatch(name):
            raise ScheduleError("task names use lowercase letters, digits and dashes (max 40)")
        return self.root / f"{name}.json"

    def _save(self, task: Task) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self._file(task.name)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(task), indent=1))
        os.replace(tmp, path)

    def load(self, name: str) -> Task:
        try:
            return Task(**json.loads(self._file(name).read_text()))
        except FileNotFoundError as exc:
            raise ScheduleError(f"no task named {name}") from exc

    def next_elapse(self, spec: str) -> str:
        result = self._run(["systemd-analyze", "calendar", "--iterations=1", spec])
        if result.returncode != 0:
            raise ScheduleError(f"'{spec}' is not a time systemd understands (examples: 'every day at 9:00', "
                                "'every monday at 18:30', 'in 20 minutes', 'every 15 minutes')")
        for line in result.stdout.splitlines():
            if "Next elapse:" in line:
                return line.split("Next elapse:", 1)[1].strip()
        return "?"

    # ------------------------------------------------------------------ api

    def plan(self, name: str, when: str) -> tuple[str, str, str]:
        """Validate; returns (kind, spec, human description of the next run)."""
        self._file(name)
        kind, spec = interpret(when)
        if kind == "once":
            return kind, spec, datetime.fromtimestamp(time.time() + int(spec)).strftime("%a %Y-%m-%d %H:%M:%S")
        return kind, spec, self.next_elapse(spec)

    def create(self, name: str, request: str, when: str, policy: str, max_tasks: int = 20) -> Task:
        if self._file(name).exists():
            raise ScheduleError(f"a task named {name} already exists; cancel it first or pick another name")
        if len(self.list()) >= max_tasks:
            raise ScheduleError(f"the policy allows at most {max_tasks} scheduled tasks")
        kind, spec, next_run = self.plan(name, when)
        task = Task(name, request.strip(), when, kind, spec, policy, time.time(), next_run=next_run)
        self.units.mkdir(parents=True, exist_ok=True)
        unit = f"agentlayer-task-{name}"
        (self.units / f"{unit}.service").write_text(
            f"[Unit]\nDescription=AgentLayer task {name}\n\n[Service]\nType=oneshot\n"
            f"ExecStart={sys.executable} -m agentlayer task-run {name}\nWorkingDirectory={PROJECT_ROOT}\n"
            f"Nice=10\nTimeoutStartSec=20min\n")
        timing = (f"OnCalendar={spec}\nPersistent=true\nAccuracySec=30s\n" if kind == "calendar"
                  else f"OnActiveSec={spec}s\nRemainAfterElapse=false\nAccuracySec=1s\n")
        (self.units / f"{unit}.timer").write_text(
            f"[Unit]\nDescription=AgentLayer timer {name}\n\n[Timer]\n{timing}Unit={unit}.service\n\n"
            f"[Install]\nWantedBy=timers.target\n")
        self._save(task)
        reload = self._systemctl("daemon-reload")
        enable = self._systemctl("enable", "--now", f"{unit}.timer")
        if enable.returncode != 0:
            self.cancel(name)
            raise ScheduleError(f"systemd refused the timer: {(enable.stderr or reload.stderr).strip()[:300]}")
        return task

    def list(self) -> list[Task]:
        tasks = []
        for file in sorted(self.root.glob("*.json")) if self.root.exists() else []:
            try:
                task = Task(**json.loads(file.read_text()))
            except (ValueError, TypeError):
                continue
            show = self._systemctl("show", f"agentlayer-task-{task.name}.timer", "-p", "NextElapseUSecRealtime",
                                   "-p", "ActiveState")
            props = dict(line.split("=", 1) for line in (show.stdout or "").splitlines() if "=" in line)
            if props.get("NextElapseUSecRealtime"):
                task.next_run = props["NextElapseUSecRealtime"]
            elif props.get("ActiveState") in ("inactive", "failed"):
                task.next_run = "not scheduled (finished)"
            tasks.append(task)
        return tasks

    def cancel(self, name: str) -> Task:
        task = self.load(name)
        unit = f"agentlayer-task-{name}"
        self._systemctl("disable", "--now", f"{unit}.timer")
        for suffix in (".timer", ".service"):
            (self.units / f"{unit}{suffix}").unlink(missing_ok=True)
        self._systemctl("daemon-reload")
        self._file(name).unlink(missing_ok=True)
        return task

    def run_now(self, name: str) -> None:
        self.load(name)
        result = self._systemctl("start", "--no-block", f"agentlayer-task-{name}.service")
        if result.returncode != 0:
            raise ScheduleError(result.stderr.strip()[:300] or "systemd could not start the task")

    def log_path(self, name: str) -> Path:
        return self.root / f"{name}.log"

    def log(self, name: str, lines: int = 60) -> str:
        self.load(name)
        try:
            return "\n".join(self.log_path(name).read_text().splitlines()[-lines:])
        except FileNotFoundError:
            return "(the task has not run yet)"

    def record_run(self, name: str, status: str) -> None:
        task = self.load(name)
        task.runs += 1
        task.last_run = time.time()
        task.last_status = status[:200]
        self._save(task)
