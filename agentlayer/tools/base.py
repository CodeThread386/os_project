"""Tool framework: declarations, argument validation and the shared context.

A tool is three functions. ``requires`` turns arguments into the capability
requirements the broker checks, ``run`` performs the system calls once the
broker has approved, and the optional ``preview`` describes what a
destructive action will do so the user can decide on a confirmation prompt.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from agentlayer.policy import Policy
from agentlayer.sandbox import Sandbox

if TYPE_CHECKING:
    from agentlayer.broker import CapabilityBroker, Decision, Requirement
    from agentlayer.journal import UndoJournal
    from agentlayer.memory import MemoryStore


class ToolArgError(ValueError):
    """The arguments are malformed or name something that does not exist."""


@dataclass
class ToolOutput:
    ok: bool
    data: dict
    summary: str
    display: str | None = None  # longer human-readable text for the terminal

    @classmethod
    def fail(cls, message: str, **data) -> "ToolOutput":
        return cls(False, {"error": message, **data}, message)


@dataclass
class ToolContext:
    policy: Policy
    broker: "CapabilityBroker"
    sandbox: Sandbox
    memory: "MemoryStore | None" = None
    session: str = ""
    # Asks the person at the keyboard for a secret (the sudo password). The
    # answer goes straight to the program that needs it: never to the model,
    # never to the audit log, never to disk.
    ask_secret: Callable[[str], str | None] | None = None
    journal: "UndoJournal | None" = None  # makes file changes undoable
    vision: Callable[[bytes, str, str], str] | None = None  # (image, mime, question) -> answer
    stream: Callable[[str], None] | None = None  # live output of long-running commands

    @property
    def workspace(self) -> str:
        return str(self.policy.workspace)

    def lexical(self, path: str) -> str:
        """Expand ``~``, anchor relative paths in the workspace, normalise ``..``."""
        path = os.path.expanduser((path or ".").strip())
        if not os.path.isabs(path):
            path = os.path.join(self.workspace, path)
        return os.path.normpath(path)

    def real(self, path: str, follow: bool = True) -> str:
        """Canonical path the kernel will reach (symlinks resolved).

        With ``follow=False`` the final component is kept as-is, so an
        operation acts on a symlink itself rather than on its target.
        """
        lex = self.lexical(path)
        if follow or lex == "/":
            return os.path.realpath(lex)
        return os.path.join(os.path.realpath(os.path.dirname(lex)), os.path.basename(lex))

    def show(self, path: str) -> str:
        return self.policy.display_path(path)


@dataclass
class Tool:
    name: str
    description: str
    rights: tuple[str, ...]
    params: dict[str, dict]
    requires: Callable[[dict, ToolContext], list["Requirement"]]
    run: Callable[[dict, ToolContext, "Decision"], ToolOutput]
    required: tuple[str, ...] = ()
    preview: Callable[[dict, ToolContext], str | None] | None = None
    destructive: bool = False
    # Filled in from tools/__init__.py:
    group: str = "core"  # tools outside "core" are loaded on demand
    mutating: bool | Callable[[dict], bool] = False  # changes something (simulated in dry-run mode)
    untrusted: bool = False  # returns outside content that is screened for prompt injection
    syscalls: str = ""  # what it does underneath, for the curious (and the OS lab)

    def changes_state(self, args: dict) -> bool:
        return self.mutating(args) if callable(self.mutating) else self.mutating

    def schema(self) -> dict:
        props = {}
        for name, spec in self.params.items():
            props[name] = {k: v for k, v in spec.items() if k in ("type", "description", "enum", "items")}
            if name not in self.required:  # models often send null for "not given"; we treat it as the default
                props[name]["type"] = [spec["type"], "null"]
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": props, "required": list(self.required)},
            },
        }


REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    REGISTRY[tool.name] = tool
    return tool


def param(type_: str, description: str, **extra) -> dict:
    return {"type": type_, "description": description, **extra}


def prepare_args(tool: Tool, raw) -> dict:
    """Validate and coerce model-supplied arguments against the tool's schema."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ToolArgError("arguments must be a JSON object")
    out: dict = {}
    for name, spec in tool.params.items():
        value = raw.get(name)
        if value is None or (value == "" and spec["type"] != "string"):
            if name in tool.required:
                raise ToolArgError(f"missing required argument '{name}'")
            if "default" in spec:
                out[name] = spec["default"]
            continue
        out[name] = _coerce(name, value, spec)
    return out


def _coerce(name: str, value, spec: dict):
    kind = spec["type"]
    if kind == "string":
        if isinstance(value, (dict, list)):
            raise ToolArgError(f"'{name}' must be a string")
        value = str(value)
        if "enum" in spec and value not in spec["enum"]:
            raise ToolArgError(f"'{name}' must be one of {', '.join(spec['enum'])}")
        return value
    if kind == "integer":
        if isinstance(value, bool):
            raise ToolArgError(f"'{name}' must be an integer")
        try:
            number = int(value) if not isinstance(value, float) or value.is_integer() else None
        except (TypeError, ValueError):
            number = None
        if number is None:
            raise ToolArgError(f"'{name}' must be an integer")
        if "minimum" in spec and number < spec["minimum"]:
            raise ToolArgError(f"'{name}' must be at least {spec['minimum']}")
        if "maximum" in spec and number > spec["maximum"]:
            raise ToolArgError(f"'{name}' must be at most {spec['maximum']}")
        return number
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in ("true", "yes", "1"):
            return True
        if text in ("false", "no", "0"):
            return False
        raise ToolArgError(f"'{name}' must be true or false")
    return value


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@dataclass
class Clip:
    """Text cut to a size limit, remembering whether anything was dropped."""

    text: str
    truncated: bool = field(default=False)

    @classmethod
    def of(cls, text: str, limit: int) -> "Clip":
        if len(text) <= limit:
            return cls(text)
        return cls(text[:limit] + f"\n…[{len(text) - limit} more characters]", True)
