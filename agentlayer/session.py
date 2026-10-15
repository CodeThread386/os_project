"""Assembling a working AgentLayer: policy, sandbox, broker, stores, model and agent.

Every front end (terminal, web dashboard, MCP server, scheduled tasks, the
evaluation harness) builds its agent through these functions, so they all
enforce exactly the same policy with the same mechanisms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agentlayer.audit import AuditLog
from agentlayer.broker import CapabilityBroker
from agentlayer.config import DEFAULT_GROQ_MODEL, STATE_DIR, Settings
from agentlayer.guard import InjectionGuard
from agentlayer.journal import UndoJournal
from agentlayer.llm import (
    OLLAMA_VISION_HINTS, GroqLLM, LLMError, OllamaLLM, OpenAICompatLLM, ollama_models, pick_ollama_model,
)
from agentlayer.memory import MemoryStore
from agentlayer.policy import Policy, load_policy
from agentlayer.sandbox import Sandbox, detect
from agentlayer.tools import ToolContext

QUOTA_STATE = STATE_DIR / "groq-quota.json"


@dataclass
class Runtime:
    policy: Policy
    sandbox: Sandbox
    broker: CapabilityBroker
    ctx: ToolContext
    audit: AuditLog


def build_runtime(settings: Settings, *, audit_path=None, state=None) -> Runtime:
    """``audit_path``/``state`` redirect the audit log and the stores (used by the evaluation harness)."""
    policy = load_policy(settings.policy, settings.workspace)
    if audit_path is not None:
        policy.audit_path = audit_path
    grant = policy.grant("exec")
    if grant is None:
        sandbox = Sandbox("none", "this policy grants no exec capability")
    else:
        sandbox = detect(settings.sandbox or grant.sandbox, str(policy.workspace))
    broker = CapabilityBroker(policy, sandbox_active=sandbox.active)
    memory_grant = policy.grant("memory")
    memory = (MemoryStore(state / "memory.json" if state else None, max_items=memory_grant.max_items)
              if memory_grant else None)
    journal = None
    if policy.has("fs.write"):
        from agentlayer.journal import Trash

        journal = UndoJournal(state / "undo", trash=Trash(state / "trash")) if state else UndoJournal()
    ctx = ToolContext(policy, broker, sandbox, memory=memory, journal=journal)
    return Runtime(policy, sandbox, broker, ctx, AuditLog(policy.audit_path))


def build_llm(settings: Settings, notify=None):
    provider = settings.provider
    local_models: list[str] | None = None

    def local() -> list[str]:
        nonlocal local_models
        if local_models is None:
            local_models = ollama_models(settings.ollama_url, timeout=1.5)
        return local_models

    if provider == "auto":
        if settings.groq_keys:
            provider = "groq"
        elif local():
            provider = "ollama"
        else:
            raise LLMError("no model provider configured: put GROQ_API_KEYS=... in .env (see .env.example), "
                           "or start Ollama for a local model")
    if provider == "groq":
        if not settings.groq_keys:
            raise LLMError("GROQ_API_KEYS (or GROQ_API_KEY) is not set")
        fallback = None
        if settings.local_fallback:
            name = settings.ollama_model or pick_ollama_model(local())
            if name:
                fallback = OllamaLLM(name, settings.ollama_url)
        return GroqLLM(settings.groq_keys, settings.model or DEFAULT_GROQ_MODEL,
                       fallback_models=settings.fallback_models, state_path=QUOTA_STATE,
                       reasoning_effort=settings.reasoning_effort, local_fallback=fallback, notify=notify)
    if provider == "ollama":
        name = settings.model or settings.ollama_model or pick_ollama_model(local())
        if not name:
            raise LLMError(f"Ollama at {settings.ollama_url} is not reachable or has no models "
                           "(ollama pull qwen2.5:7b-instruct)")
        llm = OllamaLLM(name, settings.ollama_url)
        llm.vision_model = next((m for m in local() if any(h in m for h in OLLAMA_VISION_HINTS)), None)
        return llm
    if provider == "openai":
        if not settings.base_url or not settings.model:
            raise LLMError("the openai provider needs AGENTLAYER_BASE_URL and AGENTLAYER_MODEL")
        return OpenAICompatLLM(settings.base_url, settings.model, settings.api_key)
    raise LLMError(f"unknown provider {provider!r} (use groq, ollama or openai)")


def attach_models(rt: Runtime, llm, settings: Settings) -> InjectionGuard | None:
    """Give the tools a vision model and return the injection guard for this LLM."""
    if isinstance(llm, GroqLLM):
        rt.ctx.vision = llm.vision
    elif isinstance(llm, OllamaLLM) and llm.vision_model:
        rt.ctx.vision = llm.vision
    if not settings.guard:
        return None
    if isinstance(llm, GroqLLM):
        return InjectionGuard(llm.guard_score, classifier=llm.safeguard_check)
    return InjectionGuard()


def describe_llm(llm) -> str:
    if isinstance(llm, GroqLLM):
        groups = len(llm.pool.groups())
        chain = [m for m in llm.models if m != llm.model][:2]
        text = f"groq · {llm.model} → fallbacks {', '.join(chain)}"
        if llm.local:
            text += f", local {llm.local.model}"
        return text + f"\n          {len(llm.pool.keys)} API key(s) in {groups} quota group(s)"
    return f"{llm.provider} · {llm.model}"


def make_agent(rt: Runtime, llm, events, settings: Settings, *, sessions=None, dry_run: bool = False,
               all_tools: bool = False):
    from agentlayer.agent import Agent

    guard = attach_models(rt, llm, settings)
    return Agent(llm, rt.broker, rt.ctx, rt.audit, events, max_steps=settings.max_steps, sessions=sessions,
                 guard=guard, dry_run=dry_run, all_tools=all_tools)


def session_banner_facts(rt: Runtime, agent) -> list[tuple[str, str]]:
    """Facts shown by front ends when a session starts."""
    facts = []
    if agent.guard is not None:
        facts.append(("shield", "prompt-injection screening: " + ("patterns + Llama Prompt Guard 2 + gpt-oss-safeguard"
                                                                  if agent.guard.classifier else "patterns")))
    if rt.ctx.journal is not None:
        facts.append(("undo", "file changes are journaled; deletes go to the Trash"))
    if rt.ctx.vision is not None:
        facts.append(("vision", os.environ.get("AGENTLAYER_VISION_MODEL") or "qwen/qwen3.8-27b"))
    return facts
