"""Command-line interface: interactive agent, one-shot requests and admin commands."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from agentlayer import __version__
from agentlayer.agent import Agent
from agentlayer.audit import AuditLog
from agentlayer.broker import CapabilityBroker, Verdict
from agentlayer import desktop, voice
from agentlayer.config import DEFAULT_GROQ_MODEL, Settings, load_env_files, state_dir
from agentlayer.llm import GroqLLM, LLMError, ToolCall, ollama_models
from agentlayer.session import (
    QUOTA_STATE, Runtime, build_llm, build_runtime, describe_llm, make_agent, session_banner_facts,
)
from agentlayer.journal import UndoError, UndoJournal
from agentlayer.memory import MemoryStore, SessionStore
from agentlayer.policy import PolicyError, bundled_policy_names, load_policy
from agentlayer.sandbox import Sandbox, SandboxError, detect
from agentlayer.scheduler import ScheduleError, Scheduler
from agentlayer.tools import REGISTRY, ToolArgError, ToolContext, all_tools, exposed_tools, prepare_args
from agentlayer.ui import ConsoleEvents, audit_table, print_policy, print_matrix

def banner(console: Console, rt: Runtime, llm, agent: Agent | None = None) -> None:
    pol = rt.policy
    lines = [
        f"[bold]policy[/]    {pol.name} — {escape(pol.description)}",
        f"[bold]model[/]     {escape(describe_llm(llm))}",
        f"[bold]sandbox[/]   {rt.sandbox.mode} — {escape(rt.sandbox.reason)}",
        f"[bold]workspace[/] {pol.workspace}",
        f"[bold]audit log[/] {pol.display_path(str(pol.audit_path))}",
    ]
    if any(pol.has(r) for r in ("gui.launch", "gui.windows")):
        problem = desktop.graphical_session()
        session = os.environ.get("XDG_CURRENT_DESKTOP", "desktop") + " / " + os.environ.get("XDG_SESSION_TYPE", "?")
        lines.append(f"[bold]desktop[/]   {escape(problem or session)}")
    if rt.ctx.memory is not None:
        lines.append(f"[bold]memory[/]    {len(rt.ctx.memory.all())} note(s) from earlier sessions")
    if agent is not None:
        for label, fact in session_banner_facts(rt, agent):
            lines.append(f"[bold]{label:<9}[/] {escape(fact)}")
        groups = sorted({t.group for t in agent.available} - {"core"})
        lines.append(f"[bold]tools[/]     {len(agent.available)} available; core loaded, on demand: {', '.join(groups)}")
        if agent.dry_run:
            lines.append("[bold yellow]DRY RUN[/]   changes are simulated, nothing is modified")
    console.print(Panel("\n".join(lines), title=f"[bold cyan]AgentLayer {__version__}[/] · agentic control layer for Linux",
                        border_style="cyan", expand=False))


HELP = r"""[bold]Conversation[/]
  /clear            start a fresh conversation      /resume \[id]   continue a saved one (default: latest)
  /sessions         saved conversations             /export \[file] save this conversation as Markdown
  /voice            speak a request (Enter stops)   /speak on|off  read answers aloud
[bold]Safety[/]
  /undo \[id|list]   undo the agent's last change     /dryrun on|off simulate changes instead of making them
  /grants           temporary capabilities          /revoke \[id]   revoke one (or all)
  /policy           the access matrix               /audit \[n]     the last n audit records · /verify
[bold]OS Lab demos[/]
  /trace \[cmd]      syscall trace for a command      /ps \[pid|kill|zombie]  PCB viewer + signal demo
  /limits \[stress]  resource limits in effect        /sched \[all|fcfs|sjf|rr|priority]  CPU scheduler
  /ipc               privilege separation demo        /matrix         Lampson access matrix
[bold]More[/]
  /tools            tools, groups and syscalls      /tasks         scheduled tasks
  /memory           long-term notes · /forget id    /quota         Groq key pool · /usage tokens used
  /exit             quit (Ctrl-D also works)
Anything else is sent to the agent as a request."""


# ------------------------------------------------------------------ commands


def cmd_chat(args, settings: Settings, console: Console) -> int:
    rt = build_runtime(settings)
    events = ConsoleEvents(console, verbose=settings.verbose, auto_approve=getattr(args, "yes", False))
    llm = build_llm(settings, notify=events.notice)
    agent = make_agent(rt, llm, events, settings, sessions=SessionStore(), dry_run=getattr(args, "dry_run", False))
    chat = {"speak": bool(getattr(args, "speak", False))}
    rt.audit.record("session_start", policy=rt.policy.name, provider=llm.provider, model=llm.model,
                    sandbox=rt.sandbox.mode, workspace=str(rt.policy.workspace), pid=os.getpid(),
                    dry_run=agent.dry_run or None)
    try:
        if getattr(args, "request", None):
            if not args.quiet:
                banner(console, rt, llm, agent)
            if getattr(args, "resume", False) or getattr(args, "session", None):
                _resume(agent, getattr(args, "session", None) or "latest", console)
            return _one_shot(agent, " ".join(args.request), console, chat)
        banner(console, rt, llm, agent)
        if getattr(args, "resume", False) or getattr(args, "session", None):
            _resume(agent, getattr(args, "session", None) or "latest", console)
        console.print("[dim]Type a request, or /help.[/]")
        _repl(agent, rt, llm, console, chat)
        return 0
    finally:
        revoked = rt.broker.revoke("all")
        if revoked:
            rt.audit.record("grants_revoked", ids=[g.id for g in revoked], reason="session ended")
        rt.audit.record("session_end", **agent.totals)


def _resume(agent: Agent, which: str, console: Console) -> None:
    loaded = agent.sessions.load(None if which in ("", "latest") else which) if agent.sessions else None
    if loaded is None:
        console.print("[yellow]no saved conversation to resume; starting a new one[/]")
        return
    turns = agent.resume(*loaded)
    console.print(f"[green]resumed conversation {loaded[0]}[/] [dim]({turns} earlier request(s) kept in context)[/]")


def _print_sessions(console: Console, limit: int = 15) -> None:
    rows = SessionStore().list(limit)
    if not rows:
        console.print("[dim]no saved conversations yet[/]")
        return
    table = Table(title="Saved conversations (continue with --session ID or /resume ID)")
    for col in ("id", "last used", "requests", "policy", "first request"):
        table.add_column(col, overflow="fold")
    for s in rows:
        table.add_row(s.id, datetime.fromtimestamp(s.updated).strftime("%m-%d %H:%M"), str(s.turns), s.policy,
                      escape(s.title))
    console.print(table)


def _print_memory(console: Console, store: MemoryStore | None) -> None:
    if store is None:
        console.print("[dim]this policy grants no memory capability[/]")
        return
    facts = store.all()
    if not facts:
        console.print("[dim]nothing remembered yet — tell the agent something to remember[/]")
        return
    table = Table(title=f"Long-term memory ({len(facts)} notes)")
    for col in ("id", "saved", "note"):
        table.add_column(col, overflow="fold")
    for f in reversed(facts):
        table.add_row(f.id, f.when, escape(f.text))
    console.print(table)


def _ask(agent: Agent, text: str, console: Console, chat: dict) -> str | None:
    started = time.monotonic()
    try:
        answer = agent.ask(text)
    except LLMError as exc:
        console.print(f"[red]model error:[/] {escape(str(exc))}")
        return None
    console.print(Panel(Markdown(answer), title="agent" + (" [yellow](dry run)[/]" if agent.dry_run else ""),
                        border_style="green", title_align="left"))
    if chat.get("speak"):
        voice.speak(answer)
    if time.monotonic() - started > 45 and not desktop.graphical_session():
        try:  # long jobs: tell the user even if the terminal is in the background
            desktop.notify("AgentLayer finished", voice.speakable(answer, 180))
        except desktop.DesktopError:
            pass
    return answer


def _one_shot(agent: Agent, text: str, console: Console, chat: dict | None = None) -> int:
    return 0 if _ask(agent, text, console, chat or {}) is not None else 1


def _repl(agent: Agent, rt: Runtime, llm, console: Console, chat: dict) -> None:
    try:
        import readline

        history_file = state_dir() / "history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            readline.read_history_file(history_file)
        except OSError:
            pass
        readline.set_history_length(500)
    except ImportError:
        readline = None
    prompt = "\001\033[1;36m\002you ›\001\033[0m\002 "
    try:
        while True:
            try:
                text = input(prompt).strip()
            except EOFError:
                console.print()
                return
            except KeyboardInterrupt:
                console.print()
                continue
            if not text:
                continue
            if text.startswith("/"):
                result = _slash(text, agent, rt, llm, console, chat)
                if result is True:
                    return
                if isinstance(result, str):  # /voice produced a request
                    text = result
                else:
                    continue
            try:
                _ask(agent, text, console, chat)
            except KeyboardInterrupt:
                console.print("\n[yellow]interrupted — the request was abandoned[/]")
    finally:
        if readline:
            try:
                readline.write_history_file(history_file)
            except OSError:
                pass


def _toggle(value: str, current: bool) -> bool:
    value = value.strip().lower()
    return {"on": True, "off": False, "": not current}.get(value, current)


# ================================================================= OS lab helpers


def _cmd_trace(console: Console, rt: Runtime, rest: str) -> None:
    """'/trace [command]' — run a command under strace -c and show syscall stats.

    Bridges Module 2 (user/kernel boundary): shows how a high-level instruction
    becomes low-level system calls with categories and error counts.
    """
    import shutil
    from rich import box
    from rich.table import Table

    if not shutil.which("strace"):
        console.print("[red]strace is not installed.[/] Install it with: [dim]sudo apt install strace[/]")
        return

    command = rest.strip() or "ls"
    console.print(f"[bold]Syscall trace:[/] [cyan]{command}[/]  [dim](strace -c)[/]")

    # Parse into argv tokens (simple split; no shell expansion)
    argv = command.split()

    import subprocess, re, os
    real_bin = shutil.which(argv[0]) if "/" not in argv[0] else argv[0]
    if real_bin is None:
        console.print(f"[red]{argv[0]!r} not found in PATH.[/]")
        return

    try:
        result = subprocess.run(
            ["strace", "-c", "-o", "/dev/stderr", real_bin] + argv[1:],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        console.print("[red]strace timed out after 30 s[/]")
        return
    except OSError as exc:
        console.print(f"[red]strace failed:[/] {exc}")
        return

    raw = result.stderr
    # strace -c output: "% time  seconds  usecs/call  calls  errors  syscall"
    rows, total_calls, total_errors = [], 0, 0
    in_table = False
    for line in raw.splitlines():
        if line.startswith("% time"):
            in_table = True
            continue
        if not in_table:
            continue
        line = line.strip()
        if not line or line.startswith("---") or line.startswith("100.00"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            pct, secs, us, calls = float(parts[0]), parts[1], parts[2], int(parts[3])
            errs = int(parts[4]) if len(parts) >= 6 else 0
            name = parts[-1]
        except (ValueError, IndexError):
            continue
        total_calls += calls
        total_errors += errs
        rows.append((pct, secs, calls, errs, name))

    rows.sort(key=lambda r: -r[0])

    # Syscall educational categories (Module 2 mapping)
    CATEGORIES: dict[str, str] = {
        "openat": "file I/O", "open": "file I/O", "read": "file I/O", "write": "file I/O",
        "close": "file I/O", "stat": "file I/O", "fstat": "file I/O", "lstat": "file I/O",
        "getdents64": "directory", "getdents": "directory",
        "mmap": "memory", "mprotect": "memory", "munmap": "memory", "brk": "memory",
        "execve": "process", "clone": "process", "fork": "process", "wait4": "process",
        "pipe": "IPC", "socket": "network", "connect": "network", "sendto": "network",
        "rt_sigaction": "signals", "rt_sigprocmask": "signals", "kill": "signals",
        "ioctl": "device", "fcntl": "file ctrl",
    }

    tbl = Table(title=f"Syscall summary: {command!r}", box=box.SIMPLE_HEAD, pad_edge=False)
    for col in ("%time", "calls", "errors", "category", "syscall"):
        tbl.add_column(col, justify="right" if col in ("%time", "calls", "errors") else "left")
    for pct, _, calls, errs, name in rows[:25]:
        cat = CATEGORIES.get(name, "other")
        err_cell = f"[red]{errs}[/]" if errs else "[dim]0[/]"
        tbl.add_row(f"{pct:.1f}", str(calls), err_cell, f"[dim]{cat}[/]", f"[cyan]{name}[/]")
    console.print(tbl)

    # Plain-English narrative (the "bridge" sentence)
    top_names = ", ".join(r[4] for r in rows[:5]) if rows else "none"
    console.print(
        f"\n[bold]Your command[/] [cyan]{command!r}[/] [bold]became[/] "
        f"[green]{total_calls}[/] kernel crossings (syscalls)"
        + (f", [red]{total_errors} returned errors[/]" if total_errors else "")
        + f".\n[dim]Most frequent: {top_names}[/]"
    )
    console.print(
        "[dim]Each crossing is a user→kernel mode switch: the CPU saves registers, "
        "enters ring 0, executes the kernel handler, then returns to ring 3.[/]"
    )


def _cmd_ps(console: Console, rt: Runtime, rest: str) -> None:
    """'/ps [pid|kill <pid> <sig>|zombie]' — PCB viewer and signal demo.

    Demonstrates Module 2: process states, PCBs, signals, zombie reaping.
    """
    from rich import box
    from rich.table import Table
    from agentlayer import procfs
    from agentlayer.tools.base import fmt_bytes, fmt_duration, fmt_time
    import time, os, signal

    parts = rest.split()
    sub = parts[0].lower() if parts else ""

    # ── /ps zombie: fork a child, let it become a zombie, then reap it ─────
    if sub == "zombie":
        console.print("[bold]Zombie process demo[/]  [dim](Module 2: PCB lifecycle)[/]")
        console.print("[dim]Forking a child that exits immediately while the parent delays wait()…[/]")
        pid = os.fork()
        if pid == 0:
            os._exit(0)     # child exits immediately → zombie
        import time as _t
        _t.sleep(0.3)       # parent sleeps — child is now Z state
        st = procfs.read_stat(pid)
        state = st.state if st else "gone"
        console.print(
            f"  child PID [bold]{pid}[/] state: [bold yellow]{state}[/]"
            + (" — it exited but its PCB is not yet freed (zombie)" if state == "Z" else "")
        )
        os.waitpid(pid, 0)  # reap: kernel frees the PCB
        console.print(f"  waitpid({pid}) called → PCB freed, zombie reaped.  "
                      f"PID now gone: {procfs.read_stat(pid) is None}")
        return

    # ── /ps kill <pid> <signal>: send a signal ──────────────────────────────
    if sub == "kill" and len(parts) >= 3:
        try:
            target_pid = int(parts[1])
            sig_name = parts[2].upper()
            if not sig_name.startswith("SIG"):
                sig_name = "SIG" + sig_name
            sig = signal.Signals[sig_name]
        except (ValueError, KeyError) as exc:
            console.print(f"[red]usage: /ps kill <pid> <signal>  ({exc})[/]")
            return
        st = procfs.read_stat(target_pid)
        if st is None:
            console.print(f"[red]PID {target_pid} does not exist[/]")
            return
        console.print(f"Sending [bold]{sig_name}[/] to PID {target_pid} ({st.name})")
        try:
            os.kill(target_pid, sig)
        except PermissionError:
            console.print("[red]permission denied — you cannot signal this process[/]")
            return
        time.sleep(0.2)
        after = procfs.read_stat(target_pid)
        outcome = "gone (reaped)" if after is None else f"state now: {after.state_name}"
        console.print(f"  outcome: [cyan]{outcome}[/]")
        return

    # ── /ps <integer>: deep PCB view for one process ────────────────────────
    if sub.isdigit():
        pid = int(sub)
        st = procfs.read_stat(pid)
        if st is None:
            console.print(f"[red]PID {pid} not found[/]")
            return
        status = procfs.read_status(pid)
        uid = procfs.real_uid(pid)
        boot_t = procfs.boot_time()
        started = boot_t + st.starttime / procfs.CLK_TCK

        from rich import box as rbox
        from rich.table import Table as RTable
        tbl = RTable(title=f"PCB for PID {pid} — {st.name}", box=rbox.SIMPLE_HEAD, pad_edge=False)
        tbl.add_column("field", style="bold", min_width=26)
        tbl.add_column("value")
        rows = [
            ("PID",              str(pid)),
            ("PPID",             str(st.ppid)),
            ("process group",    str(st.pgrp)),
            ("session",          str(st.session)),
            ("user",             procfs.username(uid) + f" (uid {uid})" if uid is not None else "?"),
            ("state",            f"[bold]{st.state}[/] — {st.state_name}"),
            ("nice / priority",  f"{st.nice} / {st.priority}"),
            ("sched policy",     procfs.SCHED_POLICIES.get(st.policy, str(st.policy))),
            ("CPU: user",        f"{st.utime / procfs.CLK_TCK:.3f} s"),
            ("CPU: kernel",      f"{st.stime / procfs.CLK_TCK:.3f} s"),
            ("virtual mem",      fmt_bytes(st.vsize)),
            ("resident (RSS)",   fmt_bytes(st.rss_bytes)),
            ("VmPeak",           status.get("VmPeak", "?")),
            ("VmSwap",           status.get("VmSwap", "0 kB")),
            ("minor faults",     str(st.minflt)),
            ("major faults",     str(st.majflt)),
            ("threads",          str(st.num_threads)),
            ("started",          fmt_time(started)),
            ("running for",      fmt_duration(time.time() - started)),
        ]
        for k, v in rows:
            tbl.add_row(k, v)
        console.print(tbl)
        return

    # ── /ps (default): process table ────────────────────────────────────────
    name_filter = sub  # treat non-keyword rest as name filter
    pids = procfs.list_pids()
    mem_total = procfs.meminfo().get("MemTotal", 1)

    def _sample(pid_list):
        out = {}
        for p in pid_list:
            st = procfs.read_stat(p)
            if st is not None:
                out[p] = st
        return out

    first = _sample(pids)
    interval = 0.3
    t0 = time.monotonic()
    time.sleep(interval)
    second = _sample(pids)
    elapsed = time.monotonic() - t0

    state_legend = {"R": "running", "S": "sleeping", "D": "unint. sleep",
                    "Z": "zombie", "T": "stopped", "I": "idle kernel"}

    tbl = Table(title="Process table  [dim](from /proc)[/]", box=box.SIMPLE_HEAD, pad_edge=False)
    for col in ("PID", "PPID", "User", "S", "%CPU", "RSS", "NI", "Command"):
        tbl.add_column(col, justify="right" if col in ("%CPU", "RSS", "NI") else "left")

    rows_data = []
    for pid, st in second.items():
        prev = first.get(pid)
        cpu = (st.cpu_ticks - prev.cpu_ticks) / procfs.CLK_TCK / elapsed * 100 if prev else 0.0
        uid = procfs.process_owner(pid)
        cmd = " ".join(procfs.read_cmdline(pid)) or f"[{st.name}]"
        if name_filter and name_filter not in st.name.lower() and name_filter not in cmd.lower():
            continue
        rows_data.append((pid, st.ppid, procfs.username(uid) if uid is not None else "?",
                          st.state, cpu, st.rss_bytes, st.nice, cmd))
    rows_data.sort(key=lambda r: -r[4])  # sort by CPU%

    for pid, ppid, user, state, cpu, rss, nice, cmd in rows_data[:30]:
        s_style = {"R": "green", "Z": "red", "D": "yellow", "T": "yellow"}.get(state, "")
        state_cell = f"[{s_style}]{state}[/]" if s_style else state
        tbl.add_row(str(pid), str(ppid), user[:10], state_cell,
                    f"{cpu:.1f}", fmt_bytes(rss), str(nice), cmd[:60])
    console.print(tbl)
    console.print(
        "[dim]State codes: " +
        "  ".join(f"{k}={v}" for k, v in state_legend.items()) + "[/]"
    )
    console.print("[dim]Tip: /ps <pid>  deep view   /ps kill <pid> <sig>  send signal   /ps zombie  reaping demo[/]")


def _cmd_limits(console: Console, rt: Runtime, rest: str) -> None:
    """'/limits [stress]' — show active resource limits and optionally demo containment.

    Demonstrates Module 4: rlimits, cgroup v2, OOM killer, SIGXCPU.
    """
    import resource, shutil
    from rich.table import Table
    from rich import box

    exec_grant = rt.policy.grant("exec")
    lim = exec_grant.limits if exec_grant else None
    sandbox = exec_grant.sandbox if exec_grant else "none"

    console.print(f"[bold]Resource limits[/]  [dim](exec grant, sandbox: {sandbox})[/]")

    # ── policy / grant limits ───────────────────────────────────────────────
    if lim:
        from agentlayer.tools.exec import format_limits
        console.print(format_limits(lim, sandbox))
    else:
        console.print("[dim](no exec grant configured — no rlimits will be applied to commands)[/]")

    # ── /proc/self/limits (kernel view of our own process) ─────────────────
    console.print("\n[bold]/proc/self/limits[/]  [dim](current kernel enforcement for this process)[/]")
    tbl = Table(box=box.SIMPLE_HEAD, pad_edge=False)
    for col in ("Limit", "Soft", "Hard", "Units"):
        tbl.add_column(col)
    try:
        with open("/proc/self/limits") as f:
            lines = f.read().splitlines()[1:]  # skip header
        for line in lines:
            parts = line.split()
            if len(parts) >= 3:
                # columns: "Limit  Soft Limit  Max Limit  Units"
                # The name can be multi-word; last two fields are values+units
                name = " ".join(parts[:-3]) or parts[0]
                soft, hard, units = parts[-3], parts[-2], parts[-1]
                tbl.add_row(name, soft, hard, units)
    except OSError as exc:
        console.print(f"[red]could not read /proc/self/limits: {exc}[/]")
    else:
        console.print(tbl)

    # ── stress demo ─────────────────────────────────────────────────────────
    if rest.strip().lower() == "stress":
        console.print("\n[bold]Stress demo:[/] forking a memory hog with tight rlimits…")
        import subprocess, os as _os, signal as _signal

        mem_mb = 32 if lim is None else lim.memory_mb
        cpu_s  = 5  if lim is None else lim.cpu_seconds

        # Run a Python one-liner that allocates more memory than the limit
        hog = (
            f"python3 -c \""
            f"import resource, sys; "
            f"resource.setrlimit(resource.RLIMIT_AS, ({mem_mb}*1024*1024, {mem_mb}*1024*1024)); "
            f"resource.setrlimit(resource.RLIMIT_CPU, ({cpu_s}, {cpu_s})); "
            f"print('limits set; allocating 2x memory…'); "
            f"x = bytearray({mem_mb * 2}*1024*1024)"
            f"\""
        )
        console.print(f"  command: [dim]{hog[:120]}[/]")
        try:
            proc = subprocess.run(
                ["bash", "-c", hog], capture_output=True, text=True, timeout=10,
            )
            sig_num = -proc.returncode if proc.returncode < 0 else None
            sig_name = ""
            if sig_num:
                try:
                    sig_name = _signal.Signals(sig_num).name
                except ValueError:
                    sig_name = f"signal {sig_num}"

            if proc.returncode == 0:
                console.print("  [yellow]process finished without being killed — limits may be too generous[/]")
            elif sig_name in ("SIGSEGV", "SIGBUS"):
                console.print(f"  [red]process killed: {sig_name}[/]  "
                              f"(memory limit exceeded: OS/kernel OOM action or RLIMIT_AS enforcement)")
            elif sig_name == "SIGXCPU":
                console.print("  [yellow]process killed: SIGXCPU[/]  (CPU time limit exceeded)")
            else:
                console.print(f"  [red]process exited {proc.returncode}[/] ({sig_name or 'non-zero exit'})")
            if proc.stdout:
                console.print(f"  stdout: [dim]{proc.stdout.strip()[:200]}[/]")
            if proc.stderr:
                console.print(f"  stderr: [dim]{proc.stderr.strip()[:200]}[/]")
        except subprocess.TimeoutExpired:
            console.print("  [red]timed out — hog ran for more than 10 s (check your limits)[/]")
        except OSError as exc:
            console.print(f"  [red]could not fork: {exc}[/]")


def _cmd_sched(console: Console, rest: str) -> None:
    """/sched [all|fcfs|sjf|rr|priority] [quantum=N] — CPU scheduling demo.

    Runs all four algorithms against the same job set and compares them.
    Demonstrates Module 3: scheduling policies, Gantt charts, waiting/turnaround times.
    """
    from agentlayer.jobsched import Job, DEMO_JOBS, run_fcfs, run_sjf, run_rr, run_priority, run_all
    from rich.table import Table
    from rich import box

    parts = rest.lower().split()
    algo = parts[0] if parts else "all"
    quantum = 2
    for p in parts:
        if p.startswith("quantum=") or p.startswith("q="):
            try:
                quantum = int(p.split("=")[1])
            except (ValueError, IndexError):
                pass

    jobs = DEMO_JOBS
    console.print(
        f"[bold]CPU Scheduling demo[/]  [dim]{len(jobs)} jobs, RR quantum={quantum}[/]\n"
    )

    # ── job table ────────────────────────────────────────────────────────────
    job_tbl = Table(title="Job set", box=box.SIMPLE_HEAD, pad_edge=False)
    for col in ("id", "name", "burst (ticks)", "arrival", "priority"):
        job_tbl.add_column(col, justify="right" if col in ("burst (ticks)", "arrival", "priority") else "left")
    for j in jobs:
        job_tbl.add_row(j.id, j.name, str(j.burst), str(j.arrival), str(j.priority))
    console.print(job_tbl)

    # ── run selected algorithms ──────────────────────────────────────────────
    algo_map = {
        "fcfs":     lambda: [run_fcfs(jobs)],
        "sjf":      lambda: [run_sjf(jobs)],
        "rr":       lambda: [run_rr(jobs, quantum)],
        "priority": lambda: [run_priority(jobs)],
        "all":      lambda: run_all(jobs, quantum),
    }
    if algo not in algo_map:
        console.print(f"[red]unknown algorithm {algo!r}[/]  choices: {', '.join(algo_map)}")
        return

    results = algo_map[algo]()

    # ── per-algorithm: Gantt + per-job metrics ───────────────────────────────
    for res in results:
        console.print(f"\n[bold cyan]{res.algorithm}[/]"
                      + (f"  [dim]quantum={res.quantum}[/]" if res.quantum else ""))
        # Gantt
        console.print(res.gantt_text(width=68))
        # Per-job metrics table
        mtbl = Table(box=box.SIMPLE, pad_edge=False)
        for col in ("job", "name", "burst", "arrival", "finish", "turnaround", "waiting"):
            mtbl.add_column(col, justify="right" if col not in ("name",) else "left")
        for m in res.metrics:
            mtbl.add_row(m.job_id, m.name, str(m.burst), str(m.arrival),
                         str(m.finish), f"[cyan]{m.turnaround}[/]", f"[yellow]{m.waiting}[/]")
        console.print(mtbl)
        console.print(f"  avg waiting: [yellow]{res.avg_waiting:.2f}[/]  "
                      f"avg turnaround: [cyan]{res.avg_turnaround:.2f}[/]  "
                      f"CPU util: [green]{res.cpu_util:.1f}%[/]")

    # ── comparison table when running all ────────────────────────────────────
    if len(results) > 1:
        console.print("\n[bold]Comparison[/]")
        cmp_tbl = Table(box=box.SIMPLE_HEAD, pad_edge=False)
        for col in ("algorithm", "avg waiting", "avg turnaround", "CPU util", "total ticks"):
            cmp_tbl.add_column(col, justify="right" if col != "algorithm" else "left")
        for res in results:
            cmp_tbl.add_row(*res.comparison_row())
        console.print(cmp_tbl)
        best_wait = min(results, key=lambda r: r.avg_waiting)
        best_turn = min(results, key=lambda r: r.avg_turnaround)
        console.print(
            f"  best avg waiting: [yellow]{best_wait.algorithm}[/]  "
            f"  best avg turnaround: [cyan]{best_turn.algorithm}[/]"
        )


def _cmd_ipc(console: Console, rt: Runtime) -> None:
    """/ipc — privilege separation demo over a Unix domain socket.

    Demonstrates the IPC component: planner (LLM side) never calls the OS
    directly — all actions go through the executor broker over a Unix socket.
    """
    import os, tempfile, threading, time
    from agentlayer.privbroker import make_demo_broker, DEMO_REQUESTS
    from rich.table import Table
    from rich import box

    console.print("[bold]Privilege separation demo[/]  [dim](Unix domain socket IPC)[/]")
    console.print(
        "\n[dim]Architecture:[/]\n"
        "  [bold]Planner[/] (this REPL, PID [cyan]{pid}[/]) — sends JSON requests over a socket\n"
        "  [bold]Executor[/] (forked child)       — validates policy, replies allow/deny\n".format(pid=os.getpid())
    )

    sock_path = os.path.join(tempfile.gettempdir(), f"agentlayer-privbroker-{os.getpid()}.sock")
    workspace = str(rt.policy.workspace)
    broker = make_demo_broker(workspace)

    from agentlayer.privbroker import PrivBroker

    # ── fork executor ────────────────────────────────────────────────────────
    child_pid = os.fork()
    if child_pid == 0:
        # executor process
        try:
            broker.serve(sock_path, max_requests=len(DEMO_REQUESTS) + 5)
        except Exception:
            pass
        os._exit(0)

    # parent = planner
    console.print(f"  Executor PID: [bold]{child_pid}[/]  socket: [dim]{sock_path}[/]")

    # give executor a moment to bind
    for _ in range(20):
        if os.path.exists(sock_path):
            break
        time.sleep(0.05)

    if not os.path.exists(sock_path):
        console.print("[red]executor did not bind the socket in time[/]")
        os.waitpid(child_pid, 0)
        return

    # ── send demo requests ───────────────────────────────────────────────────
    tbl = Table(title="Planner → Executor wire frames", box=box.SIMPLE_HEAD, pad_edge=False)
    for col in ("label", "request", "verdict", "reason"):
        tbl.add_column(col, overflow="fold")

    for req_tmpl in DEMO_REQUESTS:
        req = {k: v.replace("{workspace}", workspace) if isinstance(v, str) else v
               for k, v in req_tmpl.items() if k != "label"}
        label = req_tmpl.get("label", "")
        try:
            resp = PrivBroker.send(sock_path, req, timeout=3.0)
        except Exception as exc:
            resp = {"ok": False, "verdict": "error", "reason": str(exc)}

        ok = resp.get("ok", False)
        verdict = resp.get("verdict", "?")
        reason = resp.get("reason", "")
        color = "green" if ok else "red"
        tbl.add_row(
            label,
            str({k: v for k, v in req.items()})[:70],
            f"[{color}]{verdict}[/]",
            reason[:80],
        )
        time.sleep(0.02)

    console.print(tbl)

    # ── teardown ─────────────────────────────────────────────────────────────
    try:
        os.waitpid(child_pid, 0)
    except ChildProcessError:
        pass
    try:
        os.unlink(sock_path)
    except OSError:
        pass

    console.print(
        "\n[dim]Key insight: the planner never called open(2), execve(2) or connect(2) directly.\n"
        "Every OS interaction was mediated by the executor via a validated JSON protocol.\n"
        "This mirrors microkernel design (L4/seL4) and capability-based systems (Capsicum).[/]"
    )


def _cmd_matrix(console: Console, rt: Runtime) -> None:
    """/matrix — Lampson access matrix grid + capability token lifecycle.

    Demonstrates the access matrix security model: domain × right cells show
    whether each capability is granted unconditionally, needs confirmation,
    held via a temporary token, or explicitly denied.
    """
    print_matrix(console, rt.policy, rt.broker)



def _slash(text: str, agent: Agent, rt: Runtime, llm, console: Console, chat: dict) -> bool | str | None:
    cmd, _, rest = text.partition(" ")
    rest = rest.strip()
    if cmd in ("/exit", "/quit", "/q"):
        return True
    if cmd == "/help":
        console.print(HELP)
    elif cmd == "/policy":
        print_policy(console, rt.policy, rt.broker)
    elif cmd == "/matrix":
        _cmd_matrix(console, rt)
    elif cmd == "/trace":
        _cmd_trace(console, rt, rest)
    elif cmd == "/ps":
        _cmd_ps(console, rt, rest)
    elif cmd == "/limits":
        _cmd_limits(console, rt, rest)
    elif cmd == "/sched":
        _cmd_sched(console, rest)
    elif cmd == "/ipc":
        _cmd_ipc(console, rt)
    elif cmd == "/tools":
        _print_tools(console, rt.policy, agent)
    elif cmd == "/audit":
        n = int(rest) if rest.isdigit() else 15
        console.print(audit_table(AuditLog.tail(rt.policy.audit_path, n)))
    elif cmd == "/verify":
        result = AuditLog.verify(rt.policy.audit_path)
        console.print(f"[{'green' if result.ok else 'red'}]{escape(result.message)}[/]")
    elif cmd == "/quota":
        _print_quota(console, llm)
    elif cmd == "/usage":
        t = agent.totals
        console.print(f"{t['llm_calls']} model calls, {t['prompt_tokens']} prompt + {t['completion_tokens']} "
                      f"completion tokens, {t['actions']} actions executed")
    elif cmd == "/clear":
        agent.reset()
        console.print("[dim]conversation cleared[/]")
    elif cmd == "/memory":
        _print_memory(console, rt.ctx.memory)
    elif cmd == "/forget":
        fact = rt.ctx.memory.forget(rest) if rt.ctx.memory and rest else None
        console.print(f"[green]forgot:[/] {escape(fact.text)}" if fact else "[red]no note with that id[/] (see /memory)")
    elif cmd == "/sessions":
        _print_sessions(console)
    elif cmd == "/resume":
        _resume(agent, rest, console)
    elif cmd == "/undo":
        _undo(console, rt, rest)
    elif cmd == "/grants":
        _print_grants(console, rt)
    elif cmd == "/revoke":
        gone = rt.broker.revoke(rest or "all")
        if gone:
            rt.audit.record("grants_revoked", ids=[g.id for g in gone], reason="revoked by the user")
        console.print(f"[green]revoked {len(gone)} temporary capability(ies)[/]" if gone else "[dim]nothing to revoke[/]")
    elif cmd == "/dryrun":
        agent.dry_run = _toggle(rest, agent.dry_run)
        console.print(f"dry-run mode is [bold]{'ON — changes are simulated' if agent.dry_run else 'off'}[/]")
    elif cmd == "/speak":
        chat["speak"] = _toggle(rest, chat.get("speak", False))
        console.print(f"reading answers aloud: [bold]{'on' if chat['speak'] else 'off'}[/]")
    elif cmd == "/voice":
        return _voice_request(llm, console)
    elif cmd == "/tasks":
        _print_tasks(console)
    elif cmd == "/export":
        _export(console, rt, agent, rest)
    else:
        console.print(f"[red]unknown command {escape(cmd)}[/] — try /help")
    return None


def _voice_request(llm, console: Console) -> str | None:
    if not hasattr(llm, "transcribe"):
        console.print("[red]voice input needs the Groq provider (Whisper)[/]")
        return None
    console.print("[bold red]● recording[/] — speak now, press [bold]Enter[/] to stop (max 30 s)")
    try:
        audio = voice.record(30)
        with console.status("[dim]transcribing…[/]"):
            text = llm.transcribe(audio, "speech.wav")
    except (voice.VoiceError, LLMError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        return None
    if not text or len(text.strip(" .")) < 2:
        console.print("[yellow]I did not catch that[/]")
        return None
    console.print(f"[cyan]🎤 you said:[/] {escape(text)}")
    return text


def _undo(console: Console, rt: Runtime, rest: str) -> None:
    journal = rt.ctx.journal
    if journal is None:
        console.print("[dim]this policy makes no file changes, so there is nothing to undo[/]")
        return
    if rest == "list" or rest == "--list":
        rows = journal.entries(20)
        if not rows:
            console.print("[dim]no file changes recorded yet[/]")
            return
        table = Table(title="Recent file changes (undo with /undo ID)")
        for col in ("id", "when", "change", "state"):
            table.add_column(col, overflow="fold")
        for e in rows:
            table.add_row(e.id, e.when, escape(e.summary), "[dim]undone[/]" if e.undone else
                          ("[yellow]not undoable[/]" if e.note else "[green]undoable[/]"))
        console.print(table)
        return
    try:
        message = journal.undo(rest or None)
    except UndoError as exc:
        console.print(f"[red]cannot undo:[/] {escape(str(exc))}")
        return
    rt.audit.record("undo", by="user", result=message)
    console.print(f"[green]↶ {escape(message)}[/]")


def _print_grants(console: Console, rt: Runtime) -> None:
    grants = rt.broker.active_grants()
    if not grants:
        console.print("[dim]no temporary capabilities; the agent can ask for one with request_capability[/]")
        return
    table = Table(title="Temporary capabilities (revoke with /revoke ID)")
    for col in ("id", "right", "covers", "lasts", "reason"):
        table.add_column(col, overflow="fold")
    for g in grants:
        table.add_row(g.id, g.right, escape(g.target), g.remaining(), escape(g.reason))
    console.print(table)


def _print_tasks(console: Console) -> None:
    tasks = Scheduler().list()
    if not tasks:
        console.print("[dim]no scheduled tasks — ask the agent to schedule one ('every day at 9:00 …')[/]")
        return
    table = Table(title="Scheduled tasks (systemd user timers)")
    for col in ("name", "next run", "runs", "last result", "request"):
        table.add_column(col, overflow="fold")
    for t in tasks:
        table.add_row(t.name, t.next_run or "?", str(t.runs), escape(t.last_status or "—"), escape(t.request[:80]))
    console.print(table)


def _export(console: Console, rt: Runtime, agent: Agent, rest: str) -> None:
    lines = [f"# AgentLayer conversation {agent.session_id}", "",
             f"*{datetime.now():%Y-%m-%d %H:%M} · policy {rt.policy.name}*", ""]
    for m in agent.messages[1:]:
        if m["role"] == "user" and not m["content"].startswith("[AgentLayer check]"):
            lines += [f"**You:** {m['content']}", ""]
        elif m["role"] == "assistant" and m.get("content"):
            lines += [f"**Agent:** {m['content']}", ""]
        elif m["role"] == "assistant":
            calls = ", ".join(c["function"]["name"] for c in m.get("tool_calls", []))
            lines += [f"> *used: {calls}*", ""]
    target = Path(rest).expanduser() if rest else (rt.policy.workspace / "exports" /
                                                   f"conversation-{datetime.now():%Y%m%d-%H%M%S}.md")
    if not target.is_absolute():
        target = rt.policy.workspace / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines))
    console.print(f"[green]saved to {target}[/]")


def _print_tools(console: Console, policy, agent: Agent | None = None) -> None:
    exposed = {t.name for t in exposed_tools(policy)}
    table = Table(title=f"Tools (policy {policy.name})")
    table.add_column("tool", style="bold")
    table.add_column("group")
    table.add_column("rights")
    table.add_column("ok")
    table.add_column("what it does underneath", overflow="fold", style="dim")
    for t in all_tools():
        mark = "[green]✔[/]" if t.name in exposed else "[dim]·[/]"
        group = t.group + ("" if agent is None or t.group in agent.enabled else " [dim](not loaded)[/]")
        table.add_row(t.name + (" [red]⚑[/]" if t.destructive else ""), group, ", ".join(t.rights), mark, t.syscalls)
    console.print(table)
    console.print("[dim]⚑ destructive · ✔ the policy grants the rights · non-core groups load on demand[/]")


def cmd_policy(args, settings: Settings, console: Console) -> int:
    if args.list:
        for name in bundled_policy_names():
            pol = load_policy(name, settings.workspace)
            console.print(f"[bold]{name}[/]  {escape(pol.description)}")
        return 0
    print_policy(console, load_policy(settings.policy, settings.workspace))
    return 0


def cmd_check(args, settings: Settings, console: Console) -> int:
    rt = build_runtime(settings)
    tool = REGISTRY.get(args.tool)
    if tool is None:
        console.print(f"[red]unknown tool {escape(args.tool)}[/]; known: {', '.join(REGISTRY)}")
        return 2
    try:
        raw = json.loads(args.arguments) if args.arguments else {}
        prepared = prepare_args(tool, raw)
    except (ValueError, ToolArgError) as exc:
        console.print(f"[red]invalid arguments:[/] {escape(str(exc))}")
        return 2
    decision = rt.broker.authorize(tool, prepared, rt.ctx)
    color = {Verdict.ALLOW: "green", Verdict.CONFIRM: "yellow", Verdict.DENY: "red"}[decision.verdict]
    console.print(f"[bold {color}]{decision.verdict.value.upper()}[/]  {escape(decision.reason)}")
    for check in decision.checks:
        console.print(f"  [{color if check.verdict is decision.verdict else 'dim'}]{check.verdict.value:7}[/] "
                      f"{escape(check.requirement.describe())} — {escape(check.reason)}")
    console.print(f"[dim]policy {rt.policy.name}, sandbox {rt.sandbox.mode}; nothing was executed[/]")
    return 0 if decision.verdict is Verdict.ALLOW else (3 if decision.verdict is Verdict.CONFIRM else 4)


def cmd_tools(args, settings: Settings, console: Console) -> int:
    _print_tools(console, load_policy(settings.policy, settings.workspace))
    return 0


def cmd_audit(args, settings: Settings, console: Console) -> int:
    policy = load_policy(settings.policy, settings.workspace)
    path = policy.audit_path
    if args.verify:
        result = AuditLog.verify(path)
        console.print(f"[{'green' if result.ok else 'red'}]{'✔' if result.ok else '✖'} {escape(result.message)}[/]  [dim]{path}[/]")
        return 0 if result.ok else 1
    entries = AuditLog.tail(path, args.n)
    if args.json:
        for e in entries:
            print(json.dumps(e, ensure_ascii=False))
    else:
        console.print(audit_table(entries))
        console.print(f"[dim]{path}[/]")
    return 0


def cmd_memory(args, settings: Settings, console: Console) -> int:
    store = MemoryStore()
    if args.action == "add":
        text = " ".join(args.text)
        if not text:
            console.print("[red]give the note text[/]")
            return 2
        console.print(f"[green]remembered[/] [{store.add(text).id}] {escape(text)}")
    elif args.action == "forget":
        fact = store.forget(" ".join(args.text))
        console.print(f"[green]forgot:[/] {escape(fact.text)}" if fact else "[red]no note with that id[/]")
        return 0 if fact else 1
    elif args.action == "clear":
        console.print(f"[green]forgot {store.clear()} note(s)[/]")
    else:
        _print_memory(console, store)
    return 0


def cmd_sessions(args, settings: Settings, console: Console) -> int:
    _print_sessions(console, args.n)
    return 0


def cmd_undo(args, settings: Settings, console: Console) -> int:
    policy = load_policy(settings.policy, settings.workspace)
    journal = UndoJournal()
    if args.list:
        rows = journal.entries(30)
        if not rows:
            console.print("[dim]no file changes recorded yet[/]")
            return 0
        table = Table(title="File changes made by the agent (undo with: agentlayer undo ID)")
        for col in ("id", "when", "change", "state"):
            table.add_column(col, overflow="fold")
        for e in rows:
            table.add_row(e.id, e.when, escape(e.summary), "[dim]undone[/]" if e.undone else
                          ("[yellow]not undoable[/]" if e.note else "[green]undoable[/]"))
        console.print(table)
        return 0
    try:
        message = journal.undo(args.id)
    except UndoError as exc:
        console.print(f"[red]cannot undo:[/] {escape(str(exc))}")
        return 1
    AuditLog(policy.audit_path).record("undo", by="user (cli)", result=message)
    console.print(f"[green]↶ {escape(message)}[/]")
    return 0


def cmd_tasks(args, settings: Settings, console: Console) -> int:
    scheduler = Scheduler()
    try:
        if args.action == "list":
            _print_tasks(console)
        elif args.action == "cancel":
            console.print(f"[green]cancelled {scheduler.cancel(args.name).name}[/]")
        elif args.action == "run":
            scheduler.run_now(args.name)
            console.print(f"[green]started {args.name} in the background[/] (agentlayer tasks log {args.name})")
        else:
            console.print(scheduler.log(args.name, 80), markup=False)
    except ScheduleError as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        return 1
    return 0


def cmd_task_run(args, settings: Settings, console: Console) -> int:
    """Entry point for systemd: run one scheduled task unattended, log it, notify the user."""
    scheduler = Scheduler()
    task = scheduler.load(args.name)
    settings.policy = task.policy
    log_path = scheduler.log_path(task.name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        out = Console(file=log, width=110, no_color=True, highlight=False)
        out.print(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} task {task.name}: {task.request}")
        rt = build_runtime(settings)
        events = ConsoleEvents(out, verbose=True, interactive=False, auto_approve=False)
        status = "failed"
        try:
            llm = build_llm(settings, notify=events.notice)
            agent = make_agent(rt, llm, events, settings)
            # A task must not create more tasks: unattended runs get no automation tools at all.
            agent.available = [t for t in agent.available if t.group != "automation"]
            agent.reset()
            rt.audit.record("task_start", task=task.name, request=task.request[:500])
            answer = agent.ask(f"Do this now. You are running unattended at a time the user chose earlier; nobody "
                               f"can approve confirmations, so only do what needs none.\n\n{task.request}")
            out.print(answer)
            status = "ok: " + voice.speakable(answer, 150)
        except (LLMError, PolicyError, SandboxError) as exc:
            out.print(f"FAILED: {exc}")
            status = f"failed: {exc}"
            answer = status
        scheduler.record_run(task.name, status)
        rt.audit.record("task_end", task=task.name, status=status[:200])
        try:
            desktop.notify(f"AgentLayer task: {task.name}", voice.speakable(answer, 220))
        except desktop.DesktopError:
            pass
    if task.kind == "once":
        try:
            scheduler.cancel(task.name)
        except ScheduleError:
            pass
    return 0 if status.startswith("ok") else 1


def cmd_ui(args, settings: Settings, console: Console) -> int:
    from agentlayer.web.server import WebApp

    try:
        app = WebApp(settings, port=args.port)
    except OSError as exc:
        console.print(f"[red]cannot start the dashboard on port {args.port}: {exc.strerror}[/] "
                      "(another program uses it; try a different --port, or leave it out to pick a free one)")
        return 2
    console.print(Panel(f"[bold]{app.url}[/]\n[dim]Open this link (it carries a one-time token). Keep this terminal "
                        "open; Ctrl-C stops the dashboard.[/]", title="[bold cyan]AgentLayer dashboard[/]",
                        border_style="cyan", expand=False))
    if not args.no_browser and desktop.graphical_session() is None:
        try:
            desktop.open_with_default(app.url)
        except desktop.DesktopError:
            pass
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]stopping…[/]")
    finally:
        app.shutdown()
    return 0


def cmd_mcp(args, settings: Settings, console: Console) -> int:
    from agentlayer.mcp import MCPServer

    MCPServer(settings, confirm_mode=args.confirm).serve()
    return 0


def _top_view(s: dict):
    from rich.columns import Columns
    from rich.console import Group

    def bar(pct: float, width: int = 24) -> str:
        filled = int(round(min(100, pct) / 100 * width))
        color = "green" if pct < 60 else "yellow" if pct < 85 else "red"
        return f"[{color}]{'█' * filled}[/][dim]{'░' * (width - filled)}[/] {pct:5.1f}%"

    m = s["memory"]
    mem_pct = m["used"] / m["total"] * 100 if m["total"] else 0
    swap_pct = m["swap_used"] / m["swap_total"] * 100 if m["swap_total"] else 0
    per_core = [f"cpu{i:<2} {bar(c, 6)}" for i, c in enumerate(s["cpu"]["cores"])]
    cores = "\n".join("   ".join(per_core[i:i + 4]) for i in range(0, len(per_core), 4))
    from agentlayer.tools.base import fmt_bytes, fmt_duration

    head = (f"[bold]CPU[/]  {bar(s['cpu']['total'])}   load {' '.join(map(str, s['load']))}   "
            f"up {fmt_duration(s['uptime'])}\n[bold]MEM[/]  {bar(mem_pct)}   {fmt_bytes(m['used'])} / {fmt_bytes(m['total'])}"
            f"\n[bold]SWAP[/] {bar(swap_pct)}   {fmt_bytes(m['swap_used'])} / {fmt_bytes(m['swap_total'])}"
            f"\n[bold]NET[/]  ↓ {fmt_bytes(s['net']['rx'])}/s  ↑ {fmt_bytes(s['net']['tx'])}/s     "
            f"[bold]DISK[/] read {fmt_bytes(s['disk']['read'])}/s  write {fmt_bytes(s['disk']['write'])}/s"
            + (f"\n[bold]TEMP[/] {s['temperature']:.0f} °C" if s["temperature"] is not None else "")
            + (f"     [bold]BATTERY[/] {s['battery']['level']}% ({s['battery']['status']})" if s["battery"] else "")
            + f"\n[bold]PROCS[/] {s['processes']['total']} total, {s['processes']['running']} running, "
              f"{s['processes']['blocked']} blocked on I/O, {s['processes']['zombie']} zombie")
    cpu_table = Table(title="busiest (CPU)", expand=True)
    for col in ("pid", "name", "cpu", "rss"):
        cpu_table.add_column(col)
    for p in s["top_cpu"]:
        cpu_table.add_row(str(p["pid"]), escape(p["name"]), f"{p['cpu']}%", fmt_bytes(p["rss"]))
    mem_table = Table(title="largest (memory)", expand=True)
    for col in ("pid", "name", "rss", "state"):
        mem_table.add_column(col)
    for p in s["top_memory"]:
        mem_table.add_row(str(p["pid"]), escape(p["name"]), fmt_bytes(p["rss"]), p["state"])
    return Group(Panel(head + "\n\n" + cores, title="[bold cyan]AgentLayer top[/] — /proc and /sys, Ctrl-C to quit",
                       border_style="cyan"), Columns([cpu_table, mem_table], expand=True))


def cmd_top(args, settings: Settings, console: Console) -> int:
    from rich.live import Live

    from agentlayer.monitor import Monitor

    monitor = Monitor()
    monitor.sample()
    time.sleep(0.4)
    try:
        with Live(_top_view(monitor.sample()), console=console, refresh_per_second=4, screen=not args.once) as live:
            if args.once:
                return 0
            while True:
                time.sleep(args.interval)
                live.update(_top_view(monitor.sample()))
    except KeyboardInterrupt:
        pass
    return 0


def cmd_install_desktop(args, settings: Settings, console: Console) -> int:
    data_home = Path(os.environ.get("XDG_DATA_HOME") or "~/.local/share").expanduser()
    icon_dir = data_home / "icons" / "hicolor" / "scalable" / "apps"
    apps = data_home / "applications"
    icon_dir.mkdir(parents=True, exist_ok=True)
    apps.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).parent / "web" / "static" / "icon.svg", icon_dir / "agentlayer.svg")
    exe = Path(sys.executable).parent / "agentlayer"
    entries = {
        "agentlayer.desktop": ("AgentLayer", "Agentic AI control layer for Linux (dashboard)", f"{exe} ui", "false"),
        "agentlayer-terminal.desktop": ("AgentLayer (terminal)", "Talk to your computer in a terminal",
                                        f"{exe} chat", "true"),
    }
    for name, (title, comment, command, terminal) in entries.items():
        (apps / name).write_text(
            f"[Desktop Entry]\nType=Application\nName={title}\nGenericName=AI control layer\nComment={comment}\n"
            f"Exec={command}\nIcon=agentlayer\nTerminal={terminal}\nCategories=System;Utility;\n"
            f"Keywords=AI;agent;assistant;linux;automation;\n")
        console.print(f"[green]installed[/] {apps / name}")
    if shutil.which("update-desktop-database"):
        subprocess.run(["update-desktop-database", str(apps)], capture_output=True)
    console.print("AgentLayer now appears in your application launcher.")
    return 0


def cmd_eval(args, settings: Settings, console: Console) -> int:
    from agentlayer.evals import CASES, report, run_cases

    names = args.only or None
    total = len([c for c in CASES if not names or c.name in names])
    console.print(f"Running {total} scenario(s) against the live model in isolated sandboxes…")

    def progress(r) -> None:
        mark = "[green]PASS[/]" if r.passed else "[red]FAIL[/]"
        console.print(f"  {mark} {r.kind:6} {r.name:24} {r.actions} actions, {r.denied} denied, "
                      f"{r.injections_flagged} injections flagged, {r.seconds}s  [dim]{escape(r.detail)}[/]")

    try:
        results = run_cases(settings, names, progress)
    except LLMError as exc:
        console.print(f"[red]model error:[/] {escape(str(exc))}")
        return 1
    text = report(results, getattr(run_cases, "model", settings.model or settings.provider))
    out = Path(args.out).expanduser() if args.out else state_dir() / f"eval-{datetime.now():%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    tasks = [r for r in results if r.kind == "task"]
    attacks = [r for r in results if r.kind == "attack"]
    console.print(Panel(f"tasks solved: [bold]{sum(r.passed for r in tasks)}/{len(tasks)}[/]   "
                        f"attacks contained: [bold]{sum(r.passed for r in attacks)}/{len(attacks)}[/]   "
                        f"tokens: {sum(r.tokens for r in results):,}\nreport: {out}", title="evaluation", expand=False))
    return 0 if all(r.passed for r in attacks) else 1


def _print_quota(console: Console, llm) -> None:
    if not isinstance(llm, GroqLLM):
        console.print(f"[dim]provider {llm.provider}: no Groq quota in use[/]")
        return
    groups = llm.pool.groups()
    console.print(f"{len(llm.pool.keys)} key(s), {len(groups)} quota group(s) "
                  f"(keys of one Groq organisation share a budget)")
    for i, (gid, members) in enumerate(groups.items(), 1):
        console.print(f"  group {i}: {', '.join(k.label for k in members)}", style="dim")
    invalid = [k.label for k in llm.pool.keys if k.invalid]
    if invalid:
        console.print(f"  [red]invalid keys:[/] {', '.join(invalid)}")
    table = Table(title="Rate-limit budgets (per organisation × model)")
    for col in ("group", "model", "TPM available", "RPD available", "used today", "status"):
        table.add_column(col)
    index = {gid: i for i, gid in enumerate(groups, 1)}
    models = [m for m in llm.models if m not in llm.unavailable][:3]
    for row in llm.pool.status(models):
        tpm = "untried" if row["tpm_available"] is None else f"{row['tpm_available']:,}/{row['tpm_limit']:,}"
        rpd = "untried" if row["rpd_available"] is None else f"{row['rpd_available']:,}/{row['rpd_limit']:,}"
        status = (f"[red]blocked {row['blocked_for']:.0f}s[/] {escape(row['blocked_reason'])}"
                  if row["blocked_for"] else "[green]ready[/]")
        table.add_row(str(index[row["group"]]), row["model"], tpm, rpd,
                      f"{row['tokens_today']:,} tok / {row['requests_today']} req", status)
    console.print(table)


def cmd_quota(args, settings: Settings, console: Console) -> int:
    if not settings.groq_keys:
        console.print("[red]no Groq keys configured[/] (GROQ_API_KEYS in .env)")
        return 1
    llm = GroqLLM(settings.groq_keys, settings.model or DEFAULT_GROQ_MODEL, state_path=QUOTA_STATE,
                  fallback_models=settings.fallback_models)
    if args.probe:
        model = args.probe_model
        console.print(f"Probing {len(llm.pool.keys)} key(s) with one tiny request each on {model}…")
        groups = llm.probe(model, progress=lambda msg: console.print(f"  [dim]{escape(msg)}[/]"))
        console.print(f"[green]found {len(groups)} organisation(s)[/]")
    _print_quota(console, llm)
    return 0


def cmd_doctor(args, settings: Settings, console: Console) -> int:
    ok_all = True

    def report(ok: bool | None, label: str, detail: str) -> None:
        nonlocal ok_all
        mark = {True: "[green]✔[/]", False: "[red]✖[/]", None: "[yellow]•[/]"}[ok]
        if ok is False:
            ok_all = False
        console.print(f"{mark} [bold]{label:<14}[/] {escape(detail)}")

    report(sys.version_info >= (3, 11), "python", f"{platform.python_version()} ({sys.executable})")
    report(platform.system() == "Linux", "platform", f"{platform.system()} {platform.release()}")
    try:
        rt = build_runtime(settings)
        report(True, "policy", f"{rt.policy.name} ({rt.policy.source})")
        probe = Path(rt.policy.workspace) / ".agentlayer-write-test"
        probe.write_text("ok")
        probe.unlink()
        report(True, "workspace", f"{rt.policy.workspace} (writable)")
        report(rt.sandbox.active or rt.policy.grant("exec") is None, "sandbox", f"{rt.sandbox.mode}: {rt.sandbox.reason}")
        verify = AuditLog.verify(rt.policy.audit_path)
        report(verify.ok, "audit log", f"{verify.message} ({rt.policy.audit_path})")
    except (PolicyError, SandboxError, OSError) as exc:
        report(False, "runtime", str(exc))
    report(os.access("/proc/self/stat", os.R_OK), "/proc", "readable")
    problem = desktop.graphical_session()
    if problem:
        report(None, "desktop", f"{problem} (GUI tools unavailable)")
    else:
        kwin = "window control via KWin" if desktop.kwin_available() else "no KWin: window control unavailable"
        report(True, "desktop", f"{os.environ.get('XDG_CURRENT_DESKTOP', '?')} on {os.environ.get('XDG_SESSION_TYPE', '?')}; "
                                f"{kwin}; {len(desktop.list_apps())} apps")
    sudo = shutil.which("sudo")
    report(True if sudo else None, "sudo", "installed; asks for your password at approval time" if sudo
           else "not installed (run_as_root and package installs unavailable)")
    memory = MemoryStore()
    report(True, "memory", f"{len(memory.all())} note(s), {len(SessionStore().list(1000))} saved conversation(s)")
    report(bool(shutil.which("strace")) or None, "strace", "syscall tracing available" if shutil.which("strace")
           else "not installed (sudo apt install strace) — trace_syscalls unavailable")
    try:
        import importlib

        importlib.import_module("agentlayer.inotify")  # loads libc's inotify wrappers
        report(True, "inotify", "kernel file-event notifications available")
    except OSError as exc:
        report(None, "inotify", str(exc))
    timers = subprocess.run(["systemctl", "--user", "is-system-running"], capture_output=True, text=True)
    report(True if timers.stdout.strip() in ("running", "degraded") else None, "timers",
           f"systemd user manager {timers.stdout.strip() or 'unavailable'} (scheduled tasks)")
    report(bool(shutil.which("kdialog")) or None, "dialogs", "kdialog available (approvals for MCP clients)"
           if shutil.which("kdialog") else "no kdialog: MCP clients cannot ask for approval (actions are declined)")
    audio = [t for t in ("pw-record", "arecord", "spd-say") if shutil.which(t)]
    report(True if audio else None, "voice", f"{', '.join(audio)}" if audio else "no recorder or speech engine")

    if settings.groq_keys:
        llm = GroqLLM(settings.groq_keys, settings.model or DEFAULT_GROQ_MODEL, state_path=QUOTA_STATE)
        try:
            valid, models = llm.validate_keys()
        except Exception as exc:  # network errors etc.
            report(False, "groq", f"cannot reach Groq: {exc}")
        else:
            usable = [m for m in llm.models if m in models]
            report(bool(usable), "groq", f"{valid}/{len(llm.pool.keys)} keys valid; usable chat models: {', '.join(usable) or 'none'}")
            from agentlayer.llm import GUARD_MODEL, SAFEGUARD_MODEL, VISION_MODELS, WHISPER_MODEL

            extras = {"vision": VISION_MODELS[0], "injection guard": GUARD_MODEL, "safeguard": SAFEGUARD_MODEL,
                      "speech-to-text": WHISPER_MODEL}
            have = [f"{k} ✔" if v in models else f"{k} ✖" for k, v in extras.items()]
            report(all(v in models for v in extras.values()) or None, "models", " · ".join(have))
    else:
        report(None, "groq", "no GROQ_API_KEYS configured")
    local = ollama_models(settings.ollama_url)
    report(True if local else None, "ollama", f"{len(local)} local model(s): {', '.join(local[:4])}" if local
           else f"not reachable at {settings.ollama_url} (optional offline fallback)")
    console.print("[green]all checks passed[/]" if ok_all else "[red]some checks failed[/]")
    return 0 if ok_all else 1


# ----------------------------------------------------------------------- demo

DEMO_ACTIONS = [
    ("system overview", "system_info", {}),
    ("list processes by memory", "list_processes", {"sort_by": "memory", "limit": 5}),
    ("read a workspace file", "read_file", {"path": "demo/notes.txt"}),
    ("read ~/.ssh/id_rsa", "read_file", {"path": "~/.ssh/id_rsa"}),
    ("read /etc/shadow", "read_file", {"path": "/etc/shadow"}),
    ("read ~/.bashrc via ../ traversal", "read_file", {"path": "../.bashrc"}),
    ("create a new file", "write_file", {"path": "demo/new.txt", "content": "created by the agent\n"}),
    ("overwrite a file", "write_file", {"path": "demo/notes.txt", "content": "rewritten notes\n"}),
    ("run a pipeline", "run_command", {"command": "ls -l demo | wc -l"}),
    ("run python3 in the sandbox", "run_command", {"command": "python3 -c 'print(2**64)'"}),
    ("run nc (not allow-listed)", "run_command", {"command": "nc -l 4444"}),
    ("fetch an allow-listed URL", "fetch_url", {"url": "https://example.com"}),
    ("SIGTERM to own process", "signal_process", {"pid": "{sleep}", "signal": "SIGTERM"}),
    ("SIGKILL to PID 1", "signal_process", {"pid": 1, "signal": "SIGKILL"}),
    ("delete a file", "delete_path", {"path": "demo/old.txt"}),
    ("delete the audit log", "delete_path", {"path": "{audit}"}),
    ("list ~/Downloads", "list_directory", {"path": "~/Downloads"}),
    ("read ~/.bashrc (hidden config)", "read_file", {"path": "~/.bashrc"}),
    ("command with internet access", "run_command", {"command": "curl -sI https://example.com", "network": True}),
    ("launch a desktop app", "open_app", {"name": "{app}"}),
    ("list open windows", "windows", {"action": "list"}),
    ("take a screenshot", "screenshot", {}),
    ("read the clipboard", "clipboard", {"action": "read"}),
    ("install a package (root)", "packages", {"action": "install", "names": "cowsay"}),
    ("run systemctl reboot as root", "run_as_root", {"command": "systemctl reboot"}),
    ("remember a fact", "memory", {"action": "remember", "text": "demo note"}),
    ("trace the syscalls of ls", "trace_syscalls", {"command": "ls"}),
    ("open files of your own process", "os_internals", {"topic": "open_files", "pid": "{sleep}"}),
    ("schedule a daily task", "schedule_task", {"name": "demo-task", "request": "report free disk space",
                                                "when": "every day at 9:00"}),
    ("request a temporary capability", "request_capability", {"right": "fs.write", "target": "~/Projects/demo",
                                                              "reason": "demo"}),
]
# Executed by `demo --run`: everything else is only judged (no apps, installs or memory writes).
DEMO_LIVE = 17


def _demo_setup(workspace: Path) -> subprocess.Popen:
    demo = workspace / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    (demo / "notes.txt").write_text("original notes\n")
    (demo / "old.txt").write_text("stale data\n")
    (demo / "new.txt").unlink(missing_ok=True)
    return subprocess.Popen(["sleep", "300"])


def cmd_demo(args, settings: Settings, console: Console) -> int:
    """LLM-free walkthrough: the same actions judged by each phase's policy."""
    base = load_policy(settings.policy, settings.workspace)
    AuditLog(base.audit_path).record("demo", run=bool(args.run))
    sleeper = _demo_setup(base.workspace)
    if not shutil.which("strace"):
        console.print("[dim](strace is not installed; the trace row will show as denied)[/]")
    try:
        phases = [load_policy(name, settings.workspace) for name in bundled_policy_names()]
        table = Table(title="Broker decisions for the same actions under each phase policy")
        table.add_column("action", style="bold")
        for pol in phases:
            table.add_column(pol.name.split("-")[0], justify="center")
        style = {Verdict.ALLOW: "[green]allow[/]", Verdict.CONFIRM: "[yellow]confirm[/]", Verdict.DENY: "[red]deny[/]"}
        notes: dict[str, str] = {}
        runtimes = []
        for pol in phases:
            grant = pol.grant("exec")
            sandbox = detect(grant.sandbox, str(pol.workspace)) if grant else Sandbox("none", "no exec")
            broker = CapabilityBroker(pol, sandbox_active=sandbox.active)
            runtimes.append((pol, broker, ToolContext(pol, broker, sandbox)))
        app = _demo_app()
        for label, tool_name, raw in DEMO_ACTIONS:
            raw = _demo_args(raw, sleeper.pid, base, app)
            cells = []
            for pol, broker, ctx in runtimes:
                d = broker.authorize(REGISTRY[tool_name], prepare_args(REGISTRY[tool_name], raw), ctx)
                cells.append(style[d.verdict])
                if d.verdict is Verdict.DENY:
                    notes[label] = d.reason  # keep the most permissive phase's reason
            table.add_row(label, *cells)
        console.print(table)
        console.print("[bold]Why actions are denied[/] (reason given by the most permissive phase that denies):")
        for label, reason in notes.items():
            console.print(f"  [red]✖[/] {label}: [dim]{escape(reason)}[/]")

        if not args.run:
            console.print(f"\n[dim]Nothing was executed. Run `agentlayer demo --run` to execute the first {DEMO_LIVE} "
                          "actions for real under phase 5, with confirmation prompts.[/]")
            return 0
        pol, broker, ctx = runtimes[-1]
        console.rule(f"Executing under {pol.name}")
        events = ConsoleEvents(console, verbose=True, auto_approve=args.yes)
        agent = Agent(_NoLLM(), broker, ctx, AuditLog(pol.audit_path), events)
        for i, (label, tool_name, raw) in enumerate(DEMO_ACTIONS[:DEMO_LIVE]):
            console.print(f"\n[bold]{i + 1}. {label}[/]")
            agent.execute(ToolCall(f"demo{i}", tool_name, json.dumps(_demo_args(raw, sleeper.pid, base, app))))
        return 0
    finally:
        sleeper.kill()
        sleeper.wait()


def _demo_app() -> str:
    for name in ("firefox", "org.kde.dolphin", "org.kde.kate", "org.gnome.Nautilus"):
        if desktop.find_apps(name):
            return name
    apps = desktop.list_apps()
    return apps[0].id if apps else "no-desktop-apps"


def _demo_args(raw: dict, sleeper_pid: int, policy, app: str = "") -> dict:
    out = {}
    for k, v in raw.items():
        if v == "{sleep}":
            v = sleeper_pid
        elif v == "{audit}":
            v = str(policy.audit_path)
        elif v == "{app}":
            v = app
        out[k] = v
    return out


class _NoLLM:
    provider, model = "none", "none"

    def complete(self, messages, tools):
        raise LLMError("the demo does not use a model")

    def context_budget(self) -> int:
        return 100_000


# ------------------------------------------------------------------- parsing


def _common_options() -> argparse.ArgumentParser:
    """Options accepted both before and after the subcommand."""
    common = argparse.ArgumentParser(add_help=False, argument_default=argparse.SUPPRESS)
    common.add_argument("-p", "--policy", help="policy file or bundled name: phase1..phase4 (default: phase4-full)")
    common.add_argument("--provider", choices=["auto", "groq", "ollama", "openai"], help="model provider")
    common.add_argument("-m", "--model", help="model name (default: openai/gpt-oss-120b on Groq)")
    common.add_argument("--workspace", help="override the policy's workspace directory")
    common.add_argument("--sandbox", choices=["auto", "bwrap", "none"], help="override the policy's sandbox mode")
    common.add_argument("--max-steps", type=int, help="maximum tool-using steps per request")
    common.add_argument("-v", "--verbose", action="store_true", help="show full tool output and token usage")
    return common


def build_parser() -> argparse.ArgumentParser:
    common = _common_options()
    parser = argparse.ArgumentParser(
        prog="agentlayer", parents=[common],
        description="Agentic AI control layer for Linux: an LLM plans OS actions, a capability broker authorises them.",
    )
    parser.add_argument("--version", action="version", version=f"agentlayer {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser(parents=[common], name="chat", help="interactive session (default)")
    p.add_argument("--resume", action="store_true", help="continue the most recent saved conversation")
    p.add_argument("--session", metavar="ID", help="continue a specific saved conversation (see `sessions`)")
    p.add_argument("--dry-run", action="store_true", help="simulate changes instead of making them")
    p.add_argument("--speak", action="store_true", help="read answers aloud")
    p.set_defaults(func=cmd_chat)
    p = sub.add_parser(parents=[common], name="run", help="run one request and exit")
    p.add_argument("request", nargs="+")
    p.add_argument("-q", "--quiet", action="store_true", help="no banner")
    p.add_argument("-y", "--yes", action="store_true",
                   help="approve confirmation prompts automatically (never root actions)")
    p.add_argument("--resume", action="store_true", help="continue the most recent saved conversation")
    p.add_argument("--session", metavar="ID", help="continue a specific saved conversation (see `sessions`)")
    p.add_argument("--dry-run", action="store_true", help="simulate changes instead of making them")
    p.add_argument("--speak", action="store_true", help="read the answer aloud")
    p.set_defaults(func=cmd_chat)
    p = sub.add_parser(parents=[common], name="ui", help="open the web dashboard (chat, monitor, audit, matrix, …)")
    p.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: a free one)")
    p.add_argument("--no-browser", action="store_true", help="only print the link")
    p.set_defaults(func=cmd_ui)
    p = sub.add_parser(parents=[common], name="mcp", help="serve AgentLayer's guarded tools to other AI agents (MCP, stdio)")
    p.add_argument("--confirm", choices=["auto", "deny"], default=None,
                   help="auto: ask in a desktop dialog when possible (default); deny: decline anything that needs approval")
    p.set_defaults(func=cmd_mcp)
    p = sub.add_parser(parents=[common], name="top", help="live system monitor in the terminal")
    p.add_argument("--interval", type=float, default=1.5)
    p.add_argument("--once", action="store_true", help="print one snapshot and exit")
    p.set_defaults(func=cmd_top)
    p = sub.add_parser(parents=[common], name="undo", help="undo the agent's file changes (deletes come back from the Trash)")
    p.add_argument("id", nargs="?", help="change id (default: the most recent)")
    p.add_argument("--list", action="store_true", help="show recent changes")
    p.set_defaults(func=cmd_undo)
    p = sub.add_parser(parents=[common], name="tasks", help="scheduled tasks (systemd user timers)")
    p.add_argument("action", nargs="?", choices=["list", "cancel", "run", "log"], default="list")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_tasks)
    p = sub.add_parser(parents=[common], name="task-run")  # internal: started by systemd timers
    p.add_argument("name")
    p.set_defaults(func=cmd_task_run)
    p = sub.add_parser(parents=[common], name="eval", help="evaluate tasks and attacks against the live model")
    p.add_argument("--only", nargs="*", help="run only these scenarios")
    p.add_argument("--out", help="where to write the Markdown report")
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser(parents=[common], name="install-desktop", help="add AgentLayer to the application launcher")
    p.set_defaults(func=cmd_install_desktop)
    p = sub.add_parser(parents=[common], name="policy", help="show the policy as an access matrix")
    p.add_argument("--list", action="store_true", help="list the bundled phase policies")
    p.set_defaults(func=cmd_policy)
    p = sub.add_parser(parents=[common], name="check", help="ask the broker about an action without executing it")
    p.add_argument("tool")
    p.add_argument("arguments", nargs="?", help='JSON arguments, e.g. \'{"path": "notes.txt"}\'')
    p.set_defaults(func=cmd_check)
    p = sub.add_parser(parents=[common], name="tools", help="list tools and the rights they need")
    p.set_defaults(func=cmd_tools)
    p = sub.add_parser(parents=[common], name="audit", help="show or verify the audit log")
    p.add_argument("-n", type=int, default=25, help="number of records")
    p.add_argument("--verify", action="store_true", help="verify the hash chain")
    p.add_argument("--json", action="store_true", help="raw JSON lines")
    p.set_defaults(func=cmd_audit)
    p = sub.add_parser(parents=[common], name="quota", help="Groq key pool and rate-limit budgets")
    p.add_argument("--probe", action="store_true", help="map keys to organisations (one tiny request per key)")
    p.add_argument("--probe-model", default="openai/gpt-oss-20b", help="model used for probing (default %(default)s)")
    p.set_defaults(func=cmd_quota)
    p = sub.add_parser(parents=[common], name="doctor", help="check the environment")
    p.set_defaults(func=cmd_doctor)
    p = sub.add_parser(parents=[common], name="memory", help="list, add or forget long-term notes")
    p.add_argument("action", nargs="?", choices=["list", "add", "forget", "clear"], default="list")
    p.add_argument("text", nargs="*", help="note text (add) or note id (forget)")
    p.set_defaults(func=cmd_memory)
    p = sub.add_parser(parents=[common], name="sessions", help="list saved conversations")
    p.add_argument("-n", type=int, default=15)
    p.set_defaults(func=cmd_sessions)
    p = sub.add_parser(parents=[common], name="demo", help="offline walkthrough of the five policy phases")
    p.add_argument("--run", action="store_true", help="execute the actions under phase 4")
    p.add_argument("-y", "--yes", action="store_true", help="approve confirmations automatically")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_files()
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    for attr in ("policy", "provider", "model", "workspace", "sandbox", "max_steps"):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(settings, attr, value)
    settings.verbose = getattr(args, "verbose", False)
    console = Console(highlight=False)
    func = getattr(args, "func", cmd_chat)
    try:
        return func(args, settings, console)
    except (PolicyError, SandboxError, LLMError) as exc:
        console.print(f"[red]error:[/] {escape(str(exc))}")
        return 2
    except KeyboardInterrupt:
        console.print("\n[dim]bye[/]")
        return 130
