"""Process creation for the run_command tool, written with raw system calls.

``subprocess`` would hide the mechanics this project exists to show, so a
command line is executed the way a Unix shell does it:

* ``pipe()`` creates one channel between each pair of pipeline stages;
* ``fork()`` clones the agent once per stage;
* in each child, ``setsid()`` starts a new session and process group (so the
  whole job can be killed at once and cannot read the terminal),
  ``dup2()`` wires stdin/stdout/stderr, ``setrlimit()`` caps CPU time,
  memory, file size and open files, and ``execve()`` replaces the child's
  image with the program;
* the parent multiplexes the output pipes with ``select``/``poll`` under a
  deadline and collects every child with ``wait4()``, which also returns the
  child's resource usage (CPU time, peak memory, context switches).

No shell is ever involved: there is no variable expansion, command
substitution, redirection or command chaining to abuse.
"""

from __future__ import annotations

import os
import resource
import selectors
import signal
import time
from dataclasses import dataclass, field
from typing import Callable

from agentlayer.policy import ExecLimits

_READ_CHUNK = 65536


class CommandSyntaxError(ValueError):
    pass


_FORBIDDEN = {
    ";": "';' (command chaining) is not supported: run one command per call",
    "&": "'&', '&&' and background jobs are not supported: run one command per call",
    ">": "output redirection is not supported: use write_file to create files "
         "(stdout and stderr are already captured separately)",
    "<": "input redirection is not supported: pass the file name as an argument",
    "`": "command substitution is not supported: no shell is involved",
    "(": "subshells are not supported: no shell is involved",
    ")": "subshells are not supported: no shell is involved",
}


def split_pipeline(command: str) -> list[list[tuple[str, bool]]]:
    """Tokenise a command line into pipeline stages without invoking a shell.

    Supports quoting ('...', "..."), backslash escapes and ``|``. Returns
    stages of ``(word, globbable)`` pairs; a word is globbable when it had no
    quoting or escapes, mirroring how a shell decides whether to expand it.
    """
    stages: list[list[tuple[str, bool]]] = [[]]
    word: list[str] = []
    in_word = False
    globbable = True
    i, n = 0, len(command)

    def flush() -> None:
        nonlocal word, in_word, globbable
        if in_word:
            stages[-1].append(("".join(word), globbable))
        word, in_word, globbable = [], False, True

    def check_dollar(pos: int) -> None:
        nxt = command[pos + 1] if pos + 1 < n else ""
        if nxt.isalpha() or nxt in "_{(":
            raise CommandSyntaxError("shell variables and $(...) are not supported: no shell is involved")

    while i < n:
        c = command[i]
        if c in " \t\r\n":
            flush()
            i += 1
        elif c == "'":
            end = command.find("'", i + 1)
            if end < 0:
                raise CommandSyntaxError("unterminated single quote")
            word.append(command[i + 1 : end])
            in_word, globbable = True, False
            i = end + 1
        elif c == '"':
            i += 1
            while i < n and command[i] != '"':
                if command[i] == "\\" and i + 1 < n and command[i + 1] in '"\\$`':
                    word.append(command[i + 1])
                    i += 2
                    continue
                if command[i] == "$":
                    check_dollar(i)
                if command[i] == "`":
                    raise CommandSyntaxError(_FORBIDDEN["`"])
                word.append(command[i])
                i += 1
            if i >= n:
                raise CommandSyntaxError("unterminated double quote")
            in_word, globbable = True, False
            i += 1
        elif c == "\\":
            if i + 1 < n:
                word.append(command[i + 1])
            in_word, globbable = True, False
            i += 2
        elif c == "|":
            if command.startswith("||", i):
                raise CommandSyntaxError("'||' is not supported: run one command per call")
            flush()
            if not stages[-1]:
                raise CommandSyntaxError("a pipeline stage is empty")
            stages.append([])
            i += 1
        elif c in _FORBIDDEN:
            raise CommandSyntaxError(_FORBIDDEN[c])
        else:
            if c == "$":
                check_dollar(i)
            word.append(c)
            in_word = True
            i += 1
    flush()
    if not stages[-1]:
        raise CommandSyntaxError("empty command" if len(stages) == 1 else "the pipeline ends with '|'")
    return stages


@dataclass
class StageStatus:
    argv: list[str]
    pid: int
    exit_code: int | None = None
    signal: str | None = None
    exec_error: str | None = None

    def describe(self) -> str:
        if self.exec_error:
            return f"could not execute: {self.exec_error}"
        if self.signal:
            return f"killed by {self.signal}"
        return f"exited with status {self.exit_code}"


@dataclass
class RunResult:
    stdout: str
    stderr: str
    exit_code: int
    stages: list[StageStatus]
    duration: float
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    truncated: bool = False
    timed_out: bool = False
    user_cpu: float = 0.0
    system_cpu: float = 0.0
    max_rss_kb: int = 0
    voluntary_switches: int = 0
    involuntary_switches: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        last = self.stages[-1]
        if self.timed_out:
            return f"killed after the {self.duration:.0f}s time limit"
        if last.signal == "SIGXCPU":
            return "killed by SIGXCPU (CPU time limit reached)"
        if last.signal == "SIGXFSZ":
            return "killed by SIGXFSZ (file size limit reached)"
        return last.describe()


def _apply_limits(limits: ExecLimits) -> None:
    def cap(res: int, value: int) -> None:
        soft, hard = resource.getrlimit(res)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(res, (value, value))

    cap(resource.RLIMIT_CPU, limits.cpu_seconds)
    cap(resource.RLIMIT_AS, limits.memory_mb * 1024 * 1024)
    cap(resource.RLIMIT_FSIZE, limits.file_size_mb * 1024 * 1024)
    cap(resource.RLIMIT_NOFILE, limits.open_files)
    cap(resource.RLIMIT_CORE, 0)


def _child(path: str, argv: list[str], env: dict[str, str], cwd: str | int, fds: tuple[int, int, int],
           err_w: int, limits: ExecLimits) -> None:
    """Runs in the forked child. Never returns: execve() or _exit()."""
    try:
        os.setsid()
        if isinstance(cwd, int):
            os.fchdir(cwd)  # enter the directory the broker approved, via an fd
        else:
            os.chdir(cwd)
        for target, fd in enumerate(fds):
            os.dup2(fd, target)
        os.closerange(3, err_w)
        os.closerange(err_w + 1, os.sysconf("SC_OPEN_MAX"))
        # Python ignores SIGPIPE and SIGXFSZ; ignored dispositions survive
        # execve(), so restore the defaults (otherwise `yes | head` never ends).
        for sig in (signal.SIGPIPE, signal.SIGXFSZ, signal.SIGINT, signal.SIGQUIT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)
        _apply_limits(limits)
        os.execve(path, argv, env)
    except BaseException as exc:  # report to the parent through the CLOEXEC pipe
        try:
            msg = getattr(exc, "strerror", None) or str(exc)
            os.write(err_w, f"{msg}".encode()[:400])
        finally:
            os._exit(127)


def _read_all(fd: int) -> bytes:
    chunks = []
    while True:
        chunk = os.read(fd, 4096)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def run_pipeline(
    stages: list[tuple[str, list[str]]],
    *,
    cwd: str | int,
    env: dict[str, str],
    timeout: float,
    max_output: int,
    limits: ExecLimits,
    stdin_data: bytes | None = None,
    on_output: Callable[[str], None] | None = None,
) -> RunResult:
    """Run ``[(executable_path, argv), ...]`` connected by pipes.

    ``cwd`` is a path, or an open directory fd for the children to fchdir(2) into.
    ``stdin_data`` (small, e.g. a password for ``sudo -S``) is fed to the first stage.
    ``on_output`` receives output chunks as they arrive (for live display).
    """
    start = time.monotonic()
    count = len(stages)
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    if stdin_data is not None:
        devnull, feed = os.pipe()
        os.write(feed, stdin_data)  # fits in the pipe buffer; the reader sees EOF afterwards
        os.close(feed)
    else:
        devnull = os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC)
    links = [os.pipe() for _ in range(count - 1)]
    statuses: list[StageStatus] = []
    parent_side = [devnull, out_w, err_w] + [fd for pair in links for fd in pair]

    try:
        for i, (path, argv) in enumerate(stages):
            stdin = devnull if i == 0 else links[i - 1][0]
            stdout = out_w if i == count - 1 else links[i][1]
            exec_r, exec_w = os.pipe()  # O_CLOEXEC: closes by itself when execve() succeeds
            pid = os.fork()
            if pid == 0:
                _child(path, argv, env, cwd, (stdin, stdout, err_w), exec_w, limits)
            os.close(exec_w)
            failure = _read_all(exec_r)
            os.close(exec_r)
            status = StageStatus(argv=list(argv), pid=pid)
            if failure:
                status.exec_error = failure.decode(errors="replace")
            statuses.append(status)
    finally:
        for fd in parent_side:
            os.close(fd)

    out_buf, err_buf = bytearray(), bytearray()
    totals = {out_r: 0, err_r: 0}
    buffers = {out_r: out_buf, err_r: err_buf}
    timed_out = False
    sel = selectors.DefaultSelector()
    sel.register(out_r, selectors.EVENT_READ)
    sel.register(err_r, selectors.EVENT_READ)
    deadline = start + timeout
    pending = {s.pid: s for s in statuses}
    usage = {"utime": 0.0, "stime": 0.0, "maxrss": 0, "nvcsw": 0, "nivcsw": 0}

    def kill_all() -> None:
        for pid in pending:
            try:
                os.killpg(pid, signal.SIGKILL)  # pgid == pid because of setsid()
            except (ProcessLookupError, PermissionError):
                pass

    def reap(pid: int, flags: int) -> bool:
        got, status, ru = os.wait4(pid, flags)
        if got == 0:
            return False
        stage = pending.pop(pid)
        if os.WIFSIGNALED(status):
            stage.signal = signal.Signals(os.WTERMSIG(status)).name
            stage.exit_code = 128 + os.WTERMSIG(status)
        else:
            stage.exit_code = os.waitstatus_to_exitcode(status)
        usage["utime"] += ru.ru_utime
        usage["stime"] += ru.ru_stime
        usage["maxrss"] = max(usage["maxrss"], ru.ru_maxrss)
        usage["nvcsw"] += ru.ru_nvcsw
        usage["nivcsw"] += ru.ru_nivcsw
        return True

    try:
        while sel.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                kill_all()
                break
            for key, _ in sel.select(min(remaining, 0.25)):
                chunk = os.read(key.fd, _READ_CHUNK)
                if not chunk:
                    sel.unregister(key.fd)
                    continue
                totals[key.fd] += len(chunk)
                if on_output is not None:
                    try:
                        on_output(chunk.decode(errors="replace"))
                    except Exception:  # a display problem must never break the command
                        pass
                buf = buffers[key.fd]
                room = max_output - len(buf)
                if room > 0:
                    buf += chunk[:room]
        # Output is closed; wait for the processes themselves (they may linger).
        while pending:
            for pid in list(pending):
                reap(pid, os.WNOHANG)
            if not pending:
                break
            if time.monotonic() >= deadline and not timed_out:
                timed_out = True
                kill_all()
            if timed_out:
                for pid in list(pending):
                    reap(pid, 0)
            else:
                time.sleep(0.01)
    finally:
        if pending:  # interrupted (Ctrl-C) or an unexpected error: never leak children
            kill_all()
            for pid in list(pending):
                try:
                    reap(pid, 0)
                except ChildProcessError:
                    pending.pop(pid, None)
        sel.close()
        os.close(out_r)
        os.close(err_r)

    stderr = err_buf.decode(errors="replace")
    for s in statuses:
        if s.exec_error:
            stderr += f"\nagentlayer: cannot execute {s.argv[0]}: {s.exec_error}"
    return RunResult(
        stdout=out_buf.decode(errors="replace"),
        stderr=stderr.strip("\n") if not err_buf else stderr,
        exit_code=statuses[-1].exit_code if statuses[-1].exit_code is not None else 127,
        stages=statuses,
        duration=time.monotonic() - start,
        stdout_bytes=totals[out_r],
        stderr_bytes=totals[err_r],
        truncated=totals[out_r] > len(out_buf) or totals[err_r] > len(err_buf),
        timed_out=timed_out,
        user_cpu=usage["utime"],
        system_cpu=usage["stime"],
        max_rss_kb=usage["maxrss"],
        voluntary_switches=usage["nvcsw"],
        involuntary_switches=usage["nivcsw"],
    )
