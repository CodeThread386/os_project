import json
import multiprocessing

from agentlayer.audit import AuditLog


def test_chain_and_verify(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl", session="s1")
    for i in range(5):
        log.record("action", n=i)
    result = AuditLog.verify(tmp_path / "a.jsonl")
    assert result.ok and result.entries == 5
    entries = AuditLog.read(tmp_path / "a.jsonl")
    assert [e["seq"] for e in entries] == [1, 2, 3, 4, 5]
    assert entries[1]["prev"] == entries[0]["hash"]


def test_tampering_detected(tmp_path):
    path = tmp_path / "a.jsonl"
    log = AuditLog(path)
    for verdict in ("allow", "deny", "allow"):
        log.record("action", verdict=verdict)
    lines = path.read_text().splitlines()
    record = json.loads(lines[1])
    record["verdict"] = "allow"  # cover up a denial
    lines[1] = json.dumps(record)
    path.write_text("\n".join(lines) + "\n")
    result = AuditLog.verify(path)
    assert not result.ok and result.broken_at == 2 and "modified" in result.message


def test_deleted_record_detected(tmp_path):
    path = tmp_path / "a.jsonl"
    log = AuditLog(path)
    for i in range(4):
        log.record("action", n=i)
    lines = path.read_text().splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n")
    result = AuditLog.verify(path)
    assert not result.ok and result.broken_at == 2


def test_permissions(tmp_path):
    path = tmp_path / "state" / "a.jsonl"
    AuditLog(path).record("x")
    assert path.stat().st_mode & 0o777 == 0o600


def _writer(path, n):
    log = AuditLog(path)
    for i in range(n):
        log.record("action", n=i)


def test_concurrent_writers_keep_chain(tmp_path):
    path = tmp_path / "a.jsonl"
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_writer, args=(path, 25)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    result = AuditLog.verify(path)
    assert result.ok and result.entries == 100
