"""GroqLLM against a local fake of the Groq API (no network, no real quota)."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentlayer.llm import GroqLLM, LLMError, LLMReply

KEYS = {"gsk_org1_a": "org1", "gsk_org1_b": "org1", "gsk_org2_a": "org2"}


class FakeGroq:
    def __init__(self):
        self.log = []  # (key, model)
        self.behavior = lambda key, model, n: "ok"
        self.used = {}

    def handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body, headers=None):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"object": "list", "data": [{"id": m, "object": "model", "created": 0, "owned_by": "x"}
                                                            for m in ("m1", "m2")]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                key = self.headers["authorization"].split()[1]
                model = body["model"]
                fake.log.append((key, model))
                org = KEYS.get(key)
                n = sum(1 for k, m in fake.log if k == key and m == model)
                action = fake.behavior(key, model, n)
                fake.used[(org, model)] = fake.used.get((org, model), 0) + 1
                rl = {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "7000",
                      "x-ratelimit-reset-tokens": "7.5s", "x-ratelimit-limit-requests": "1000",
                      "x-ratelimit-remaining-requests": str(1000 - fake.used[(org, model)]),
                      "x-ratelimit-reset-requests": "1m26.4s"}
                if org is None or action == "401":
                    return self._send(401, {"error": {"message": "Invalid API Key", "type": "invalid_request_error",
                                                      "code": "invalid_api_key"}})
                if action == "429":
                    return self._send(429, {"error": {"message": f"Rate limit reached for model `{model}` on tokens per "
                                                                 "minute (TPM): Limit 8000. Please try again in 30s.",
                                                      "type": "tokens", "code": "rate_limit_exceeded"}},
                                      {**rl, "retry-after": "30", "x-ratelimit-remaining-tokens": "0"})
                if action == "429-day":
                    return self._send(429, {"error": {"message": "Rate limit reached on tokens per day (TPD). "
                                                                 "Please try again in 2h3m.", "code": "rate_limit_exceeded"}},
                                      {**rl, "retry-after": "7380"})
                if action == "404":
                    return self._send(404, {"error": {"message": f"The model `{model}` does not exist",
                                                      "code": "model_not_found"}})
                if action == "tool_use_failed":
                    return self._send(400, {"error": {"message": "Failed to call a function", "code": "tool_use_failed",
                                                      "failed_generation": "<garbage>"}})
                message = {"role": "assistant", "content": None,
                           "tool_calls": [{"id": "call_1", "type": "function",
                                           "function": {"name": "system_info", "arguments": "{}"}}]} \
                    if body.get("tools") else {"role": "assistant", "content": "hello from " + model}
                self._send(200, {"id": "x", "object": "chat.completion", "created": int(time.time()), "model": model,
                                 "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128}}, rl)

        return Handler


@pytest.fixture
def fake_groq():
    fake = FakeGroq()
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{server.server_address[1]}"
    yield fake
    server.shutdown()
    server.server_close()
    thread.join()


def make(fake, tmp_path, keys=("gsk_org1_a", "gsk_org2_a"), **kw):
    notes = []
    llm = GroqLLM(list(keys), "m1", fallback_models=["m1", "m2"], state_path=tmp_path / "q.json",
                  base_url=fake.url, notify=notes.append, max_pacing_wait=0.5, **kw)
    return llm, notes


MSGS = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "system_info", "description": "x",
                                           "parameters": {"type": "object", "properties": {}}}}]


def test_success_parses_tool_calls_and_tracks_budget(fake_groq, tmp_path):
    llm, _ = make(fake_groq, tmp_path)
    reply = llm.complete(MSGS, TOOLS)
    assert reply.tool_calls[0].name == "system_info" and reply.prompt_tokens == 120
    key = next(k for k in llm.pool.keys if k.label == reply.key)
    assert llm.pool.budget(key.group, "m1").remaining_tokens == 7000


def test_429_reroutes_to_another_organisation(fake_groq, tmp_path):
    fake_groq.behavior = lambda key, model, n: "429" if key == "gsk_org1_a" else "ok"
    llm, notes = make(fake_groq, tmp_path, keys=("gsk_org1_a", "gsk_org2_a"))
    first = llm.complete(MSGS, [])
    second = llm.complete(MSGS, [])
    assert first.content == "hello from m1" and second.key == first.key == "…g2_a"
    assert ("gsk_org1_a", "m1") in fake_groq.log and fake_groq.log.count(("gsk_org1_a", "m1")) == 1
    assert any("rate limited" in n for n in notes)


def test_invalid_key_is_dropped(fake_groq, tmp_path):
    llm, notes = make(fake_groq, tmp_path, keys=("gsk_unknown", "gsk_org2_a"))
    for _ in range(2):
        assert llm.complete(MSGS, []).content
    assert llm.pool.keys[0].invalid
    assert fake_groq.log.count(("gsk_unknown", "m1")) <= 1


def test_model_unavailable_falls_back(fake_groq, tmp_path):
    fake_groq.behavior = lambda key, model, n: "404" if model == "m1" else "ok"
    llm, _ = make(fake_groq, tmp_path)
    reply = llm.complete(MSGS, [])
    assert reply.model == "m2" and "fallback model m2" in reply.notes[0]
    assert "m1" in llm.unavailable


def test_tool_use_failed_is_retried(fake_groq, tmp_path):
    fake_groq.behavior = lambda key, model, n: "tool_use_failed" if len(fake_groq.log) == 1 else "ok"
    llm, _ = make(fake_groq, tmp_path)
    assert llm.complete(MSGS, TOOLS).tool_calls


class FakeLocal:
    provider, model = "ollama", "local-model"

    def complete(self, messages, tools):
        return LLMReply("answered locally", [], self.model, self.provider)

    def context_budget(self):
        return 6000


def test_everything_exhausted_uses_local_model(fake_groq, tmp_path):
    fake_groq.behavior = lambda key, model, n: "429-day"
    llm, notes = make(fake_groq, tmp_path, local_fallback=FakeLocal())
    reply = llm.complete(MSGS, [])
    assert reply.content == "answered locally" and reply.provider == "ollama"
    assert len(fake_groq.log) == 4  # each (organisation, model) bucket was tried exactly once
    # A second request goes straight to the local model: the buckets are known to be parked.
    llm.complete(MSGS, [])
    assert len(fake_groq.log) == 4


def test_everything_exhausted_without_local_raises(fake_groq, tmp_path):
    fake_groq.behavior = lambda key, model, n: "429-day"
    llm, _ = make(fake_groq, tmp_path)
    with pytest.raises(LLMError, match="rate-limited"):
        llm.complete(MSGS, [])


def test_parked_buckets_survive_restart(fake_groq, tmp_path):
    fake_groq.behavior = lambda key, model, n: "429-day" if key == "gsk_org1_a" else "ok"
    llm, _ = make(fake_groq, tmp_path)
    llm.complete(MSGS, [])
    llm.complete(MSGS, [])
    count = fake_groq.log.count(("gsk_org1_a", "m1"))
    fresh, _ = make(fake_groq, tmp_path)  # new process, same state file
    fresh.complete(MSGS, [])
    assert fake_groq.log.count(("gsk_org1_a", "m1")) == count


def test_shared_organisation_is_learned(fake_groq, tmp_path):
    llm, _ = make(fake_groq, tmp_path, keys=("gsk_org1_a", "gsk_org1_b", "gsk_org2_a"))
    for _ in range(4):
        llm.complete(MSGS, [])
    groups = llm.pool.groups()
    assert len(groups) == 2
    assert {k.secret for k in llm.pool.keys if k.group == llm.pool.keys[0].group} == {"gsk_org1_a", "gsk_org1_b"}


def test_probe_maps_organisations(fake_groq, tmp_path):
    llm, _ = make(fake_groq, tmp_path, keys=("gsk_org1_a", "gsk_org1_b", "gsk_org2_a"))
    groups = llm.probe("m2")
    assert sorted(len(v) for v in groups.values()) == [1, 2]
