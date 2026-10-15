import pytest

from agentlayer.quota import Budget, KeyPool, fingerprint, parse_duration


@pytest.mark.parametrize("text,seconds", [("2m59.56s", 179.56), ("7.66s", 7.66), ("600ms", 0.6),
                                          ("1h2m3s", 3723), ("12", 12), ("", 0), (None, 0)])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == pytest.approx(seconds)


def headers(rem_tokens, reset_tokens="6s", rem_req=990, reset_req="14m", limit_tokens=8000, limit_req=1000):
    return {"x-ratelimit-limit-tokens": str(limit_tokens), "x-ratelimit-remaining-tokens": str(rem_tokens),
            "x-ratelimit-reset-tokens": reset_tokens, "x-ratelimit-limit-requests": str(limit_req),
            "x-ratelimit-remaining-requests": str(rem_req), "x-ratelimit-reset-requests": reset_req}


def test_budget_token_bucket_refill():
    b = Budget()
    b.update(headers(2000, "45s"), now=1000.0)
    assert b.tokens_available(1000.0) == 2000
    assert b.tokens_available(1022.5) == pytest.approx(5000)  # halfway through the refill
    assert b.tokens_available(1100.0) == 8000
    assert b.wait_for(1000, 1000.0) == 0
    assert b.wait_for(5000, 1000.0) == pytest.approx(22.5)


def test_budget_blocked_until():
    b = Budget(blocked_until=1050.0)
    assert b.wait_for(10, 1000.0) == pytest.approx(50)


def test_pool_prefers_headroom_and_untried(tmp_path):
    pool = KeyPool(["k1", "k2"], tmp_path / "q.json")
    k1, k2 = pool.keys
    pool.observe(k1, "m", headers(500, "50s", rem_req=900), 100)
    key, wait = pool.choose("m", 300)
    assert key is k2 and wait == 0  # untried bucket beats a nearly empty one
    pool.observe(k2, "m", headers(7000, rem_req=500), 100)
    key, _ = pool.choose("m", 300)
    assert key is k2


def test_pool_waits_when_everything_is_spent(tmp_path):
    pool = KeyPool(["k1"], tmp_path / "q.json")
    pool.observe(pool.keys[0], "m", headers(100, "50s"), 100)
    key, wait = pool.choose("m", 4000)
    assert key is None and 0 < wait <= 50


def test_penalize_and_persistence(tmp_path):
    state = tmp_path / "q.json"
    pool = KeyPool(["k1", "k2"], state)
    pool.penalize(pool.keys[0], "m", 3600, "daily limit reached")
    reloaded = KeyPool(["k1", "k2"], state)
    key, _ = reloaded.choose("m", 100)
    assert key.secret == "k2"
    assert "k1" not in state.read_text()  # only fingerprints are stored
    assert fingerprint("k1") in state.read_text()


def test_group_learning_merges_keys_of_one_org(tmp_path):
    pool = KeyPool(["a1", "a2", "b1"], tmp_path / "q.json")
    a1, a2, b1 = pool.keys
    pool.observe(a1, "m", headers(7900, rem_req=999), 100)
    pool.observe(b1, "m", headers(7900, rem_req=990), 100)  # different org: unrelated counter
    pool.observe(a2, "m", headers(7800, rem_req=998), 100)  # shared counter: 999 -> 998
    assert a2.group == a1.group and b1.group != a1.group
    assert len(pool.groups()) == 2


def test_group_learning_ignores_ambiguous_evidence(tmp_path):
    pool = KeyPool(["a", "b", "c"], tmp_path / "q.json")
    a, b, c = pool.keys
    pool.observe(a, "m", headers(7900, rem_req=999), 10)
    pool.observe(b, "m", headers(7900, rem_req=999), 10)
    pool.observe(c, "m", headers(7900, rem_req=998), 10)  # could belong to either
    assert len(pool.groups()) == 3


def test_group_learning_splits_wrong_merge(tmp_path):
    pool = KeyPool(["a", "b"], tmp_path / "q.json")
    a, b = pool.keys
    pool.regroup({b.fp: a.group})
    pool.observe(a, "m", headers(7000, rem_req=500), 10)
    pool.observe(b, "m", headers(7900, rem_req=999), 10)  # far more requests left: not the same org
    assert b.group != a.group


def test_invalid_keys_are_skipped(tmp_path):
    pool = KeyPool(["bad", "good"], tmp_path / "q.json")
    pool.invalidate(pool.keys[0])
    assert pool.choose("m", 10)[0].secret == "good"
    pool.invalidate(pool.keys[1])
    with pytest.raises(RuntimeError):
        pool.choose("m", 10)
