"""Locations, .env loading and runtime settings."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = Path(os.environ.get("XDG_STATE_HOME") or "~/.local/state").expanduser() / "agentlayer"
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME") or "~/.config").expanduser() / "agentlayer"

DEFAULT_POLICY = "phase5-desktop"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def state_dir() -> Path:
    """The state directory as the environment says *now* (STATE_DIR is fixed at import)."""
    return Path(os.environ.get("XDG_STATE_HOME") or "~/.local/state").expanduser() / "agentlayer"


def load_dotenv(path: Path) -> list[str]:
    """Load KEY=VALUE lines into os.environ without overriding existing values."""
    loaded: list[str] = []
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return loaded
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def load_env_files() -> None:
    seen: set[Path] = set()
    for candidate in (Path.cwd() / ".env", PROJECT_ROOT / ".env", CONFIG_DIR / ".env"):
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            load_dotenv(resolved)


def groq_keys_from_env() -> list[str]:
    raw = f"{os.environ.get('GROQ_API_KEYS', '')},{os.environ.get('GROQ_API_KEY', '')}"
    keys: list[str] = []
    for key in re.split(r"[,\s]+", raw):
        if key and key not in keys:
            keys.append(key)
    return keys


@dataclass
class Settings:
    provider: str = "auto"  # auto | groq | ollama | openai
    model: str | None = None
    groq_keys: list[str] = field(default_factory=list)
    fallback_models: list[str] | None = None
    base_url: str | None = None
    api_key: str | None = None
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_model: str | None = None
    local_fallback: bool = True
    reasoning_effort: str = "low"
    guard: bool = True
    policy: str = DEFAULT_POLICY
    workspace: str | None = None
    sandbox: str | None = None
    max_steps: int | None = None
    verbose: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        fallbacks = env("AGENTLAYER_FALLBACK_MODELS")
        return cls(
            provider=env("AGENTLAYER_PROVIDER", "auto"),
            model=env("AGENTLAYER_MODEL") or None,
            groq_keys=groq_keys_from_env(),
            fallback_models=[m.strip() for m in fallbacks.split(",") if m.strip()] if fallbacks else None,
            base_url=env("AGENTLAYER_BASE_URL") or None,
            api_key=env("AGENTLAYER_API_KEY") or env("OPENAI_API_KEY") or None,
            ollama_url=env("OLLAMA_HOST", DEFAULT_OLLAMA_URL),
            ollama_model=env("AGENTLAYER_OLLAMA_MODEL") or None,
            local_fallback=env("AGENTLAYER_LOCAL_FALLBACK", "1").lower() not in ("0", "false", "no"),
            reasoning_effort=env("AGENTLAYER_REASONING_EFFORT", "low"),
            guard=env("AGENTLAYER_GUARD", "1").lower() not in ("0", "false", "no"),
            policy=env("AGENTLAYER_POLICY", DEFAULT_POLICY),
            workspace=env("AGENTLAYER_WORKSPACE") or None,
            sandbox=env("AGENTLAYER_SANDBOX") or None,
        )
