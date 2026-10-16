"""Tests for agentlayer.privbroker — Unix domain socket privilege separation."""

from __future__ import annotations

import os
import tempfile
import threading
import time

import pytest

from agentlayer.privbroker import PrivBroker, make_demo_broker, DEMO_REQUESTS


# ──────────────────────────── fixtures ───────────────────────────────────────


@pytest.fixture
def workspace(tmp_path):
    """Temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return str(ws)


@pytest.fixture
def broker(workspace):
    return make_demo_broker(workspace)


@pytest.fixture
def sock_path(tmp_path):
    return str(tmp_path / "test-privbroker.sock")


# ──────────────────────────── _evaluate unit tests ───────────────────────────


def test_evaluate_unknown_action(broker):
    result = broker._evaluate({"action": "super.hack"})
    assert result["ok"] is False
    assert result["verdict"] == "deny"
    assert "unknown" in result["reason"]


def test_evaluate_fs_read_allowed(broker, workspace):
    path = os.path.join(workspace, "notes.txt")
    result = broker._evaluate({"action": "fs.read", "path": path})
    assert result["ok"] is True
    assert result["verdict"] == "allow"


def test_evaluate_fs_read_denied_shadow(broker):
    result = broker._evaluate({"action": "fs.read", "path": "/etc/shadow"})
    assert result["ok"] is False
    assert result["verdict"] == "deny"


def test_evaluate_fs_read_missing_path(broker):
    result = broker._evaluate({"action": "fs.read"})
    assert result["ok"] is False


def test_evaluate_exec_allowed(broker):
    result = broker._evaluate({"action": "exec", "program": "ls"})
    assert result["ok"] is True


def test_evaluate_exec_denied_netcat(broker):
    result = broker._evaluate({"action": "exec", "program": "nc"})
    assert result["ok"] is False


def test_evaluate_exec_denied_bash_fullpath(broker):
    # /bin/bash — basename is 'bash' which is NOT on the allow-list
    result = broker._evaluate({"action": "exec", "program": "/bin/bash"})
    assert result["ok"] is False


def test_evaluate_exec_missing_program(broker):
    result = broker._evaluate({"action": "exec"})
    assert result["ok"] is False


def test_evaluate_net_read_allowed(broker):
    result = broker._evaluate({"action": "net.read", "url": "https://pypi.org/simple/"})
    assert result["ok"] is True


def test_evaluate_net_read_denied(broker):
    result = broker._evaluate({"action": "net.read", "url": "https://malware.example.com/payload"})
    assert result["ok"] is False


def test_evaluate_net_read_subdomain_allowed(broker):
    result = broker._evaluate({"action": "net.read", "url": "https://api.github.com/repos"})
    assert result["ok"] is True


def test_evaluate_net_read_missing_url(broker):
    result = broker._evaluate({"action": "net.read"})
    assert result["ok"] is False


def test_evaluate_sys_read_always_allowed(broker):
    result = broker._evaluate({"action": "sys.read"})
    assert result["ok"] is True


# ──────────────────────────── serve / send integration ───────────────────────


def _start_server(broker, sock_path, n_requests=5):
    """Run the broker's serve() in a background thread."""
    t = threading.Thread(target=broker.serve, args=(sock_path,), kwargs={"max_requests": n_requests}, daemon=True)
    t.start()
    return t


def _wait_for_socket(sock_path, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not os.path.exists(sock_path):
        if time.monotonic() > deadline:
            raise TimeoutError(f"socket {sock_path} did not appear")
        time.sleep(0.02)


def test_send_allow(broker, workspace, sock_path):
    t = _start_server(broker, sock_path)
    _wait_for_socket(sock_path)
    resp = PrivBroker.send(sock_path, {"action": "exec", "program": "ls"})
    assert resp["ok"] is True
    assert resp["verdict"] == "allow"


def test_send_deny(broker, sock_path):
    t = _start_server(broker, sock_path)
    _wait_for_socket(sock_path)
    resp = PrivBroker.send(sock_path, {"action": "fs.read", "path": "/etc/shadow"})
    assert resp["ok"] is False
    assert resp["verdict"] == "deny"


def test_send_multiple_requests(broker, workspace, sock_path):
    t = _start_server(broker, sock_path, n_requests=6)
    _wait_for_socket(sock_path)

    pairs = [
        ({"action": "exec", "program": "ls"},        True),
        ({"action": "exec", "program": "nc"},         False),
        ({"action": "net.read", "url": "https://pypi.org/"}, True),
        ({"action": "net.read", "url": "https://evil.com/"}, False),
    ]
    for req, expected_ok in pairs:
        resp = PrivBroker.send(sock_path, req)
        assert resp["ok"] is expected_ok, f"unexpected verdict for {req}: {resp}"


def test_send_invalid_json_recovers(broker, sock_path):
    """The server must not crash on malformed input and continue serving."""
    t = _start_server(broker, sock_path, n_requests=3)
    _wait_for_socket(sock_path)

    import socket as _socket
    # send garbage then close
    client = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    client.connect(sock_path)
    client.sendall(b"not json at all\n")
    client.recv(1024)  # read error response
    client.close()

    # server should still respond to a valid request
    resp = PrivBroker.send(sock_path, {"action": "sys.read"})
    assert resp["ok"] is True


def test_socket_file_removed_after_serve(broker, sock_path):
    t = _start_server(broker, sock_path, n_requests=1)
    _wait_for_socket(sock_path)
    PrivBroker.send(sock_path, {"action": "sys.read"})
    t.join(timeout=2.0)
    assert not os.path.exists(sock_path)


def test_socket_owner_only_permissions(broker, sock_path):
    t = _start_server(broker, sock_path, n_requests=1)
    _wait_for_socket(sock_path)
    mode = oct(os.stat(sock_path).st_mode)
    # Should be 0o600 (owner rw only)
    assert mode.endswith("600"), f"unexpected socket permissions: {mode}"
    PrivBroker.send(sock_path, {"action": "sys.read"})


# ──────────────────────────── demo requests ──────────────────────────────────


def test_demo_requests_cover_allow_and_deny(broker, workspace, sock_path):
    t = _start_server(broker, sock_path, n_requests=len(DEMO_REQUESTS) + 2)
    _wait_for_socket(sock_path)
    verdicts = []
    for tmpl in DEMO_REQUESTS:
        req = {k: v.replace("{workspace}", workspace) if isinstance(v, str) else v
               for k, v in tmpl.items() if k != "label"}
        resp = PrivBroker.send(sock_path, req)
        verdicts.append(resp["ok"])
    # Some should allow, some should deny
    assert any(verdicts), "expected at least one ALLOW"
    assert not all(verdicts), "expected at least one DENY"


def test_demo_requests_labeled(broker):
    for req in DEMO_REQUESTS:
        assert "label" in req, f"demo request missing label: {req}"
        assert "action" in req, f"demo request missing action: {req}"


# ──────────────────────────── make_demo_broker ───────────────────────────────


def test_make_demo_broker_allowed_programs(workspace):
    b = make_demo_broker(workspace)
    assert "ls" in b.allowed_programs
    assert "python3" in b.allowed_programs


def test_make_demo_broker_allowed_paths(workspace):
    b = make_demo_broker(workspace)
    assert workspace in b.allowed_paths


def test_make_demo_broker_allowed_hosts(workspace):
    b = make_demo_broker(workspace)
    assert "pypi.org" in b.allowed_hosts
    assert "github.com" in b.allowed_hosts
