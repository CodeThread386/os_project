"""Rate-limit-aware pool of Groq API keys.

Groq enforces limits per *organisation* and per *model*: requests per day
(RPD), tokens per minute (TPM) and tokens per day (TPD). Several keys from
the same organisation therefore share one budget, while keys from different
organisations do not. The pool:

* groups keys into organisations, learning the grouping from the shared
  request counters Groq reports (``agentlayer quota --probe`` maps it
  explicitly) and correcting itself if a guess turns out wrong;
* models each (organisation, model) budget as the token bucket Groq uses,
  refilled continuously from the ``x-ratelimit-*`` response headers;
* sends each request to the bucket with the most headroom, round-robin over
  the keys inside an organisation, and paces requests instead of letting
  them fail;
* parks a bucket after a 429 until the time Groq names (minutes for TPM,
  hours for daily limits), and persists all of this between sessions.

The state file stores only key fingerprints, never the keys themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_GROUP_WINDOW = 180.0  # seconds within which two counter snapshots are comparable


def parse_duration(text: str | None) -> float:
    """'2m59.56s' -> 179.56, '600ms' -> 0.6, '1h2m' -> 3720."""
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        pass
    scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(num) * scale[unit] for num, unit in _DURATION.findall(text))


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class Budget:
    """One token bucket pair (TPM and RPD) for an (organisation, model)."""

    limit_tokens: int | None = None
    remaining_tokens: int | None = None
    tokens_full_at: float = 0.0
    limit_requests: int | None = None
    remaining_requests: int | None = None
    requests_full_at: float = 0.0
    observed_at: float = 0.0
    blocked_until: float = 0.0
    blocked_reason: str = ""
    day: str = ""
    tokens_today: int = 0
    requests_today: int = 0

    @staticmethod
    def _refilled(limit: int | None, remaining: int | None, full_at: float, observed: float, now: float) -> float | None:
        if limit is None or remaining is None:
            return None
        if now >= full_at or full_at <= observed:
            return float(limit)
        rate = (limit - remaining) / (full_at - observed)
        return min(float(limit), remaining + rate * (now - observed))

    def tokens_available(self, now: float) -> float | None:
        return self._refilled(self.limit_tokens, self.remaining_tokens, self.tokens_full_at, self.observed_at, now)

    def requests_available(self, now: float) -> float | None:
        return self._refilled(self.limit_requests, self.remaining_requests, self.requests_full_at, self.observed_at, now)

    def wait_for(self, tokens: int, now: float) -> float:
        """Seconds until this bucket can accept a request of ``tokens`` tokens."""
        wait = max(0.0, self.blocked_until - now)
        avail = self.tokens_available(now)
        if avail is not None and self.limit_tokens:
            need = min(tokens, self.limit_tokens)
            if avail < need:
                span = self.tokens_full_at - self.observed_at
                rate = (self.limit_tokens - (self.remaining_tokens or 0)) / span if span > 0 else 0
                wait = max(wait, (need - avail) / rate if rate > 0 else max(1.0, self.tokens_full_at - now))
        reqs = self.requests_available(now)
        if reqs is not None and reqs < 1 and self.limit_requests:
            span = self.requests_full_at - self.observed_at
            rate = (self.limit_requests - (self.remaining_requests or 0)) / span if span > 0 else 0
            wait = max(wait, (1 - reqs) / rate if rate > 0 else max(1.0, self.requests_full_at - now))
        return wait

    def update(self, headers, now: float) -> None:
        def num(name: str) -> int | None:
            value = headers.get(name)
            try:
                return int(float(value)) if value is not None else None
            except ValueError:
                return None

        if num("x-ratelimit-limit-tokens") is not None:
            self.limit_tokens = num("x-ratelimit-limit-tokens")
            self.remaining_tokens = num("x-ratelimit-remaining-tokens")
            self.tokens_full_at = now + parse_duration(headers.get("x-ratelimit-reset-tokens"))
        if num("x-ratelimit-limit-requests") is not None:
            self.limit_requests = num("x-ratelimit-limit-requests")
            self.remaining_requests = num("x-ratelimit-remaining-requests")
            self.requests_full_at = now + parse_duration(headers.get("x-ratelimit-reset-requests"))
        self.observed_at = now

    def count(self, tokens: int) -> None:
        if self.day != _today():
            self.day, self.tokens_today, self.requests_today = _today(), 0, 0
        self.tokens_today += tokens
        self.requests_today += 1


@dataclass
class ApiKey:
    secret: str = field(repr=False)
    fp: str
    group: str
    invalid: bool = False
    uses: int = 0
    last_used: float = 0.0

    @property
    def label(self) -> str:
        return f"…{self.secret[-4:]}"


class KeyPool:
    def __init__(self, keys: list[str], state_path: Path | None = None) -> None:
        if not keys:
            raise ValueError("no Groq API keys configured")
        self.state_path = state_path
        self.keys = [ApiKey(secret=k, fp=fingerprint(k), group=fingerprint(k)) for k in keys]
        self.budgets: dict[str, Budget] = {}
        self._load()

    # ------------------------------------------------------------ state

    def _bucket(self, group: str, model: str) -> str:
        return f"{group}|{model}"

    def budget(self, group: str, model: str) -> Budget:
        return self.budgets.setdefault(self._bucket(group, model), Budget())

    def groups(self) -> dict[str, list[ApiKey]]:
        out: dict[str, list[ApiKey]] = {}
        for key in self.keys:
            if not key.invalid:
                out.setdefault(key.group, []).append(key)
        return out

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return
        groups = data.get("groups", {})
        invalid = set(data.get("invalid", []))
        for key in self.keys:
            key.group = groups.get(key.fp, key.group)
            key.invalid = key.fp in invalid
        for name, raw in data.get("budgets", {}).items():
            try:
                self.budgets[name] = Budget(**raw)
            except TypeError:
                continue
        # Wall-clock timestamps are stored, so stale buckets simply read as refilled.

    def save(self) -> None:
        if not self.state_path:
            return
        known = {k.group for k in self.keys}
        data = {
            "groups": {k.fp: k.group for k in self.keys},
            "invalid": [k.fp for k in self.keys if k.invalid],
            "budgets": {n: asdict(b) for n, b in self.budgets.items() if n.split("|", 1)[0] in known},
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = self.state_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, self.state_path)

    # --------------------------------------------------------- selection

    def choose(self, model: str, tokens: int, now: float | None = None) -> tuple[ApiKey | None, float]:
        """Best key for a request of ``tokens`` tokens, or (None, seconds to wait)."""
        now = time.time() if now is None else now
        groups = self.groups()
        if not groups:
            raise RuntimeError("every configured Groq API key was rejected as invalid")
        best: tuple[float, float, ApiKey] | None = None
        soonest = float("inf")
        for group, members in groups.items():
            b = self.budget(group, model)
            wait = b.wait_for(tokens, now)
            if wait > 0:
                soonest = min(soonest, wait)
                continue
            headroom = b.tokens_available(now)
            score = headroom if headroom is not None else float("inf")  # untried buckets first
            key = min(members, key=lambda k: k.last_used)
            # Most headroom wins; on a tie prefer the bucket used longest ago.
            rank = (score, -key.last_used)
            if best is None or rank > best[:2]:
                best = (rank[0], rank[1], key)
        if best:
            return best[2], 0.0
        return None, soonest

    def soonest_wait(self, model: str, tokens: int) -> float:
        now = time.time()
        waits = [self.budget(g, model).wait_for(tokens, now) for g in self.groups()]
        return min(waits) if waits else float("inf")

    # ------------------------------------------------------- observation

    def observe(self, key: ApiKey, model: str, headers, tokens_used: int) -> None:
        now = time.time()
        self._learn_group(key, model, headers, now)
        b = self.budget(key.group, model)
        b.update(headers, now)
        b.count(tokens_used)
        key.uses += 1
        key.last_used = now
        self.save()

    def _learn_group(self, key: ApiKey, model: str, headers, now: float) -> None:
        """Keys of one organisation share a request counter: use it to group them."""
        try:
            remaining = int(headers.get("x-ratelimit-remaining-requests"))
            limit = int(headers.get("x-ratelimit-limit-requests"))
        except (TypeError, ValueError):
            return
        own = self.budgets.get(self._bucket(key.group, model))
        if own and own.remaining_requests is not None and now - own.observed_at < _GROUP_WINDOW:
            refill = (now - own.observed_at) / 86400 * (own.limit_requests or limit) + 1
            if remaining > own.remaining_requests + refill and len(self.groups().get(key.group, [])) > 1:
                key.group = key.fp  # counter does not fit the group's: a different organisation
                return
        if len(self.groups().get(key.group, [])) > 1:
            return
        # Evidence of sharing: another group was observed *after* this key's own last
        # observation, and this key's counter has dropped just past it (it includes
        # that group's request). A drop explained by this key's own requests is not evidence.
        own_seen = own.observed_at if own else 0.0
        candidates = []
        for group in self.groups():
            if group == key.group:
                continue
            other = self.budgets.get(self._bucket(group, model))
            if (other and other.remaining_requests is not None and other.limit_requests == limit
                    and own_seen < other.observed_at and now - other.observed_at < _GROUP_WINDOW
                    and 0 < other.remaining_requests - remaining <= 2):
                candidates.append(group)
        if len(candidates) == 1:  # ambiguous evidence is ignored until usage diverges
            key.group = candidates[0]

    def penalize(self, key: ApiKey, model: str, seconds: float, reason: str, headers=None) -> None:
        now = time.time()
        b = self.budget(key.group, model)
        if headers is not None:
            b.update(headers, now)
        b.blocked_until = max(b.blocked_until, now + max(seconds, 1.0))
        b.blocked_reason = reason
        key.last_used = now
        self.save()

    def invalidate(self, key: ApiKey) -> None:
        key.invalid = True
        self.save()

    def regroup(self, assignment: dict[str, str]) -> None:
        """Apply an explicit fingerprint -> group mapping (from a probe)."""
        for key in self.keys:
            if key.fp in assignment:
                key.group = assignment[key.fp]
        self.save()

    def status(self, models: list[str]) -> list[dict]:
        now = time.time()
        rows = []
        for group, members in self.groups().items():
            for model in models:
                b = self.budgets.get(self._bucket(group, model))
                tok = b.tokens_available(now) if b else None
                req = b.requests_available(now) if b else None
                rows.append({
                    "group": group,
                    "keys": [k.label for k in members],
                    "model": model,
                    "tpm_available": None if tok is None else int(tok),
                    "tpm_limit": b.limit_tokens if b else None,
                    "rpd_available": None if req is None else int(req),
                    "rpd_limit": b.limit_requests if b else None,
                    "blocked_for": max(0.0, b.blocked_until - now) if b else 0.0,
                    "blocked_reason": b.blocked_reason if b and b.blocked_until > now else "",
                    "tokens_today": b.tokens_today if b and b.day == _today() else 0,
                    "requests_today": b.requests_today if b and b.day == _today() else 0,
                })
        return rows
