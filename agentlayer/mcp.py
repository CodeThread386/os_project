"""AgentLayer as a Model Context Protocol (MCP) server.

Any MCP-capable AI agent (Claude Code, Claude Desktop, Cursor, ...) can use
AgentLayer's tools, and every call goes through the same capability broker,
sandbox, injection screening, undo journal and audit log as the built-in
agent. AgentLayer becomes a reference monitor for *other people's* agents.

Transport: JSON-RPC 2.0, one message per line on stdin/stdout. Everything
else (logs, notices) goes to stderr. When a call needs the user's approval,
a KDE dialog (kdialog) asks on the desktop; without one the action is
declined.

    claude mcp add agentlayer -- /path/to/.venv/bin/agentlayer mcp
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid

from agentlayer import __version__, desktop
from agentlayer.agent import AgentEvents
from agentlayer.config import Settings
from agentlayer.llm import GroqLLM, LLMError, ToolCall
from agentlayer.session import build_llm, build_runtime, make_agent

SUPPORTED_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")


def _log(message: str) -> None:
    print(f"[agentlayer-mcp] {message}", file=sys.stderr, flush=True)


class DialogEvents(AgentEvents):
    """Approvals through a desktop dialog, because stdio belongs to the protocol."""

    def __init__(self, mode: str = "auto") -> None:
        self.mode = mode  # auto (kdialog if possible) | deny
        self.kdialog = shutil.which("kdialog")

    def _can_ask(self) -> bool:
        return self.mode != "deny" and bool(self.kdialog) and desktop.graphical_session() is None

    def confirm(self, tool, args, decision, preview):
        if not self._can_ask():
            _log(f"declined {tool.name}: no way to ask the user ({decision.reason})")
            return False
        who = "an AI agent connected over MCP"
        text = (f"{who} wants to run:\n\n{tool.name}  {json.dumps(args)[:400]}\n\n{decision.reason}"
                + (f"\n\n{preview[:1500]}" if preview else "")
                + ("\n\nThis runs as ROOT." if decision.privileged else ""))
        try:
            result = subprocess.run([self.kdialog, "--title", "AgentLayer — approve this action?", "--warningyesno",
                                     text, "--yes-label", "Approve", "--no-label", "Deny"],
                                    env=desktop.session_env(), timeout=600, stdin=subprocess.DEVNULL,
                                    capture_output=True)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def secret(self, prompt):
        if not self._can_ask():
            return None
        try:
            result = subprocess.run([self.kdialog, "--title", "AgentLayer", "--password", prompt],
                                    env=desktop.session_env(), timeout=300, stdin=subprocess.DEVNULL,
                                    capture_output=True, text=True)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.rstrip("\n") if result.returncode == 0 else None

    def notice(self, message):
        _log(message)

    def warning(self, message):
        _log("WARNING: " + message)


class _NoModel:
    """MCP clients bring their own model; AgentLayer only needs one for vision and screening."""

    provider, model = "mcp-client", "client"

    def complete(self, messages, tools):
        raise LLMError("the MCP server does not plan; the connected client does")

    def context_budget(self):
        return 100_000


class MCPServer:
    def __init__(self, settings: Settings, confirm_mode: str | None = None) -> None:
        self.settings = settings
        self.rt = build_runtime(settings)
        self.events = DialogEvents(confirm_mode or os.environ.get("AGENTLAYER_MCP_CONFIRM", "auto"))
        try:
            llm = build_llm(settings, notify=_log)
        except LLMError:
            llm = None
        model = llm if isinstance(llm, GroqLLM) else _NoModel()
        self.agent = make_agent(self.rt, model, self.events, settings, all_tools=True)
        self.client = "unknown client"

    def tools(self) -> list[dict]:
        out = []
        for tool in self.agent.available:
            schema = tool.schema()["function"]
            out.append({"name": tool.name, "description": schema["description"] + f" [OS: {tool.syscalls}]",
                        "inputSchema": schema["parameters"]})
        return out

    def handle(self, message: dict) -> dict | None:
        method, params, mid = message.get("method"), message.get("params") or {}, message.get("id")
        if mid is None:  # a notification: nothing to answer
            return None
        try:
            result = self._dispatch(method, params)
        except _RPCError as exc:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": exc.code, "message": exc.message}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def _dispatch(self, method: str, params: dict):
        if method == "initialize":
            wanted = params.get("protocolVersion")
            info = params.get("clientInfo") or {}
            self.client = f"{info.get('name', 'unknown')} {info.get('version', '')}".strip()
            self.rt.audit.record("session_start", frontend="mcp", client=self.client, policy=self.rt.policy.name)
            return {
                "protocolVersion": wanted if wanted in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "agentlayer", "version": __version__},
                "instructions": (f"AgentLayer enforces the '{self.rt.policy.name}' policy on every tool call. Denials "
                                 "are final; do not work around them. Destructive actions ask the user in a desktop "
                                 "dialog. Deleted files go to the Trash and most changes can be undone with 'undo'."),
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self.tools()}
        if method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments") or {}
            text = self.agent.execute(ToolCall(f"mcp-{uuid.uuid4().hex[:8]}", name, json.dumps(arguments)))
            try:
                is_error = "error" in json.loads(text)
            except ValueError:
                is_error = False
            return {"content": [{"type": "text", "text": text}], "isError": is_error}
        if method in ("resources/list", "prompts/list"):
            return {method.split("/")[0]: []}
        raise _RPCError(-32601, f"method not found: {method}")

    def serve(self, stdin=None, stdout=None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        _log(f"ready: policy {self.rt.policy.name}, {len(self.agent.available)} tools, sandbox {self.rt.sandbox.mode}")
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
            else:
                reply = self.handle(message) if isinstance(message, dict) else None
            if reply is not None:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()
        self.rt.audit.record("session_end", frontend="mcp", client=self.client, **self.agent.totals)


class _RPCError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message
