"""Terminal rendering with rich: live agent events, access matrix, audit log."""

from __future__ import annotations

import getpass
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from rich import box
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agentlayer.agent import AgentEvents
from agentlayer.broker import Decision, Verdict
from agentlayer.llm import LLMReply
from agentlayer.policy import Policy
from agentlayer.tools import Tool, ToolOutput

ALWAYS_SHOW_OUTPUT = {"run_command", "list_processes", "list_directory", "search_files", "windows", "memory",
                      "trace_syscalls", "os_internals", "network_connections", "watch_path", "undo", "tasks",
                      "services", "look_at_screen", "describe_image"}

MATRIX_COLUMNS = [
    ("fs.read", "read"), ("fs.write", "write"), ("fs.delete", "del"), ("exec", "exec"),
    ("proc.read", "ps"), ("proc.signal", "kill"), ("proc.priority", "nice"),
    ("net.read", "net"), ("sys.read", "info"), ("gui.launch", "apps"), ("gui.windows", "win"),
    ("gui.screen", "shot"), ("gui.clipboard", "clip"), ("sys.admin", "root"), ("memory", "mem"),
    ("sys.logs", "logs"), ("task.schedule", "tasks"), ("cap.request", "jit"),
]

INVARIANTS = [
    "The policy file and the audit log can never be modified or deleted by the agent.",
    "Paths are checked after resolving symlinks and '..'; files are then opened with O_NOFOLLOW.",
    "/proc/*/environ, /proc/*/mem and /proc/kcore are never readable (secrets, raw memory).",
    "PID 1, the agent itself and its parent processes can never be signalled.",
    "The workspace root itself cannot be deleted or moved; a recursive delete may not sweep up denied files.",
    "Commands run without a shell, with a minimal environment (no API keys) and rlimits on CPU, memory, files.",
    "Hosts that resolve to loopback or private addresses are never fetched; redirects are re-authorised.",
    "Anything that runs as root needs a live approval every time (no --yes, no 'always'); the sudo password "
    "goes straight to sudo, never to the model or the log.",
    "Files matching deny rules inside mounted folders are masked inside the sandbox.",
    "Windows of the program AgentLayer runs in can never be closed; GUI apps get an environment without secrets.",
    "AgentLayer's state (memory, sessions, quota) is only changed through its own tools, never by file tools.",
    "Temporary capabilities widen only existing rights, never override deny rules, expire, and need a live 'y'.",
    "Outside content is screened for prompt injection; after a hit, every change in that request needs approval.",
    "Deletes move files to the Trash and overwrites are backed up first, so the agent's changes can be undone.",
]


def fmt_args(args: dict, width: int = 70) -> str:
    parts = []
    for key, value in args.items():
        if isinstance(value, str) and ("\n" in value or len(value) > 60):
            shown = f"<{len(value)} chars>" if key == "content" else repr(value[:57] + "…")
        else:
            shown = repr(value) if isinstance(value, str) else str(value)
        parts.append(f"{key}={shown}")
    text = "  ".join(parts)
    return text if len(text) <= width else text[: width - 1] + "…"


class ConsoleEvents(AgentEvents):
    def __init__(self, console: Console, verbose: bool = False, interactive: bool | None = None,
                 auto_approve: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.auto_approve = auto_approve
        self._streamed = False
        self._partial = ""

    @contextmanager
    def planning(self) -> Iterator[None]:
        with self.console.status("[dim]planning…[/]", spinner="dots"):
            yield

    def tool_call(self, name: str, args: dict) -> None:
        self._streamed = False
        self._partial = ""
        self.console.print(f"[bold cyan]●[/] [bold]{escape(name)}[/] [dim]{escape(fmt_args(args))}[/]")

    def output(self, text: str) -> None:
        """Live output of a running command, line by line."""
        self._streamed = True
        self._partial += text
        *lines, self._partial = self._partial.split("\n")
        for line in lines:
            self.console.print(Text("    " + line[:300], style="dim"), soft_wrap=False, overflow="ellipsis")

    def warning(self, message: str) -> None:
        self.console.print(f"  [bold red]⚠ {escape(message)}[/]")

    def decision(self, name: str, args: dict, decision: Decision) -> None:
        reason = escape(decision.reason)
        if decision.verdict is Verdict.ALLOW:
            self.console.print(f"  [green]✔ allowed[/] [dim]{reason}[/]")
        elif decision.verdict is Verdict.CONFIRM:
            self.console.print(f"  [yellow]⚠ confirmation required[/] [dim]{reason}[/]")
        else:
            self.console.print(f"  [red]✖ denied[/] {reason}")

    def confirm(self, tool: Tool, args: dict, decision: Decision, preview: str | None) -> bool | str:
        root = decision.privileged
        human = decision.requires_human
        color = "red" if root else ("magenta" if human else "yellow")
        body: list = [Text(f"{tool.name}  {fmt_args(args, 200)}", style="bold"), Text(decision.reason, style=color)]
        if preview:
            if preview.startswith("---") or "\n@@" in preview:
                body.append(Syntax(preview, "diff", theme="ansi_dark", background_color="default"))
            else:
                body.append(Text(preview))
        title = ("[bold red]Run as ROOT? Approve this action[/]" if root else
                 "[bold magenta]Approve? (asked every time)[/]" if human else "[yellow]Approve this action?[/]")
        self.console.print(Panel(Group(*body), title=title, border_style=color, expand=False))
        if self.auto_approve and not human:
            self.console.print("  [yellow]auto-approved (--yes)[/]")
            return True
        if not self.interactive:
            why = "this kind of action always needs a person at the keyboard" if human else "no terminal to ask"
            self.console.print(f"  [red]declined automatically: {why}[/]")
            return False
        choices = ["y", "n"] if human else ["y", "n", "a"]
        hint = "" if human else " [dim](a = yes, and don't ask again for this tool during this request)[/]"
        try:
            answer = Prompt.ask(f"  [bold]Approve?[/]{hint}", choices=choices, default="n", console=self.console)
        except EOFError:
            answer = "n"
        if answer == "a":
            self.console.print(f"  [green]approved for the rest of this request ({tool.name})[/]")
            return "always"
        self.console.print("  [green]approved by you[/]" if answer == "y" else "  [red]declined by you[/]")
        return answer == "y"

    def secret(self, prompt: str) -> str | None:
        if not self.interactive:
            return None
        self.console.print("  [dim]typed characters are hidden; the password goes only to sudo[/]")
        try:
            return getpass.getpass(f"  {prompt}: ")
        except (EOFError, KeyboardInterrupt):
            return None

    def result(self, name: str, output: ToolOutput) -> None:
        if self._partial:
            self.console.print(Text("    " + self._partial[:300], style="dim"))
            self._partial = ""
        mark = "[green]↳[/]" if output.ok else "[red]↳[/]"
        self.console.print(f"  {mark} {escape(output.summary)}")
        if output.data.get("security_notice"):
            self.console.print("  [bold red]⚠ the result above was flagged as a possible prompt injection[/]")
        if self._streamed and not self.verbose:
            return  # already shown live
        if output.display and (self.verbose or name in ALWAYS_SHOW_OUTPUT or name == "write_file"):
            lines = output.display.splitlines()
            limit = 60 if self.verbose else 12
            shown = "\n".join(lines[:limit]) + (f"\n… {len(lines) - limit} more lines" if len(lines) > limit else "")
            if name == "write_file":
                self.console.print(Syntax(shown, "diff", theme="ansi_dark", background_color="default"), style="dim")
            else:
                self.console.print(Text(shown, style="dim"), soft_wrap=False, overflow="ellipsis")

    def reply(self, reply: LLMReply) -> None:
        for note in reply.notes:
            self.console.print(f"  [yellow]· {escape(note)}[/]")
        if self.verbose:
            key = f" · key {reply.key}" if reply.key else ""
            self.console.print(f"  [dim]{reply.provider}/{reply.model}{key} · "
                               f"{reply.prompt_tokens} prompt + {reply.completion_tokens} completion tokens[/]")

    def notice(self, message: str) -> None:
        self.console.print(f"  [yellow]· {escape(message)}[/]")


# ------------------------------------------------------------ access matrix


def matrix_data(policy: Policy, broker=None) -> dict:
    """The access matrix as data: columns (rights) and rows (objects) with cells ✔ / ✔* / ask / ✖."""
    rows: dict[str, dict[str, str]] = {}

    def put(obj: str, right: str, cell: str) -> None:
        rows.setdefault(obj, {})[right] = cell

    for g in policy.grants:
        cell = "ask" if g.confirm else ("✔*" if g.confirm_overwrite else "✔")
        if g.right.startswith("fs."):
            for rule in g.paths:
                put(rule.raw, g.right, cell)
        elif g.right.startswith("proc."):
            put(f"processes ({'all users' if g.scope == 'all' else 'own'})", g.right, cell)
        elif g.right == "exec":
            put(f"programs: {len(g.programs)} allow-listed", "exec", cell)
        elif g.right == "net.read":
            put(f"hosts: {len(g.hosts)} allow-listed", "net.read", cell)
            if g.confirm_unlisted:
                put("hosts: other public hosts", "net.read", "ask")
        elif g.right == "gui.launch":
            put(f"apps: {len(g.apps)} allow-listed", "gui.launch", cell)
            if g.confirm_unlisted:
                put("apps: other installed apps", "gui.launch", "ask")
        elif g.right == "gui.windows":
            put("desktop windows", "gui.windows", "✔*" if g.confirm_close and not g.confirm else cell)
        elif g.right == "gui.clipboard":
            put("clipboard", "gui.clipboard", "✔*" if g.confirm_read and not g.confirm else cell)
        elif g.right == "gui.screen":
            put("screen", "gui.screen", cell)
        elif g.right == "sys.admin":
            put(f"root: {len(g.programs)} programs, {len(g.managers)} package managers", "sys.admin", "ask")
        elif g.right == "memory":
            put("long-term memory", "memory", cell)
        elif g.right == "sys.logs":
            put("systemd journal", "sys.logs", cell)
        elif g.right == "task.schedule":
            put("systemd user timers", "task.schedule", "ask")
        elif g.right == "cap.request":
            put(f"temporary grants (≤ {g.max_minutes} min)", "cap.request", "ask")
        else:
            put("this machine", g.right, cell)
        if g.right == "exec" and g.network != "never":
            put("internet for commands", "exec", "ask" if g.network == "confirm" else "✔")
        if g.right == "exec":
            for d in g.extra_dirs:
                put(policy.display_path(d) + " (commands)", "exec", cell)
    for rule in policy.deny:
        for right in ("fs.read", "fs.write", "fs.delete"):
            put(rule.raw, right, "✖")

    temporary = []
    for g in (broker.active_grants() if broker is not None else []):
        put(f"{g.target} (temporary {g.id})", g.right, "temp")
        temporary.append({"id": g.id, "right": g.right, "target": g.target, "lasts": g.remaining(), "reason": g.reason})
    columns = [(r, label) for r, label in MATRIX_COLUMNS if r in _rights_in(policy) or any(r in c for c in rows.values())]
    return {"policy": policy.name, "columns": [{"right": r, "label": label} for r, label in columns],
            "rows": [{"object": obj, "cells": cells} for obj, cells in rows.items()], "temporary": temporary,
            "invariants": INVARIANTS}


def matrix_table(policy: Policy, broker=None) -> Table:
    data = matrix_data(policy, broker)
    columns = [(c["right"], c["label"]) for c in data["columns"]]
    table = Table(title=f"Access matrix — {policy.name}", box=box.SIMPLE_HEAD, title_justify="left", pad_edge=False,
                  caption="✔ allowed  ask = needs your approval  ✔* overwrite/close/read asks  ✖ denied "
                          "(overrides grants)  temp = temporary grant  · no capability", caption_justify="left")
    table.add_column("object \\ right", style="bold", overflow="fold", min_width=22)
    for _, label in columns:
        table.add_column(label, justify="center")
    style = {"✔": "green", "✔*": "green", "ask": "yellow", "✖": "red", "temp": "magenta"}
    for row in data["rows"]:
        cells = row["cells"]
        table.add_row(escape(row["object"]),
                      *(f"[{style[c]}]{c}[/]" if (c := cells.get(r)) else "[dim]·[/]" for r, _ in columns))
    return table


def _rights_in(policy: Policy) -> set[str]:
    present = {g.right for g in policy.grants}
    return present | ({"fs.read", "fs.write", "fs.delete"} if policy.deny else set())


def print_policy(console: Console, policy: Policy, broker=None) -> None:
    console.print(matrix_table(policy, broker))
    grants = broker.active_grants() if broker is not None else []
    if grants:
        console.print("[bold]Temporary capabilities[/] (granted this session, expiring):")
        for g in grants:
            console.print(f"  [magenta]{g.id}[/] {g.right} on {escape(g.target)} — {g.remaining()}")
    console.print(f"[dim]source: {policy.source}\nworkspace: {policy.workspace}[/]")
    exec_grant = policy.grant("exec")
    if exec_grant:
        console.print(f"\n[bold]exec allow-list[/] ({exec_grant.sandbox} sandbox, network "
                      f"{'on' if exec_grant.network else 'off'}, {exec_grant.timeout}s timeout):")
        console.print("  " + ", ".join(exec_grant.programs), style="dim")
        if exec_grant.sandboxed_only:
            console.print("  only inside the sandbox: " + ", ".join(exec_grant.sandboxed_only), style="dim")
    net = policy.grant("net.read")
    if net:
        console.print("\n[bold]net.read allow-list[/] (GET only): " + ", ".join(net.hosts), style="dim")
    console.print("\n[bold]Built-in invariants[/] (mechanism, hold under every policy):")
    for line in INVARIANTS:
        console.print(f"  • {line}", style="dim")


def _local_time(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%m-%d %H:%M:%S")
    except ValueError:
        return ts[:19]


def audit_table(entries: list[dict]) -> Table:
    table = Table(title="Audit log", show_lines=False)
    for col in ("seq", "time", "event", "tool", "verdict", "detail"):
        table.add_column(col, overflow="fold")
    colors = {"allow": "green", "confirm": "yellow", "deny": "red", "rejected": "red"}
    for e in entries:
        verdict = e.get("verdict", "")
        if verdict == "confirm":
            verdict = "confirmed" if e.get("confirmed") else "declined"
        detail = e.get("summary") or e.get("reason") or e.get("text") or e.get("model") or ""
        if e.get("event") == "action" and e.get("verdict") in ("deny", "rejected"):
            detail = e.get("reason", "")
        color = colors.get(e.get("verdict", ""), "white")
        if verdict == "declined":
            color = "red"
        table.add_row(str(e.get("seq", "")), _local_time(e.get("ts", "")), e.get("event", ""), e.get("tool", ""),
                      f"[{color}]{verdict}[/]" if verdict else "", escape(" ".join(str(detail).split())[:110]))
    return table
