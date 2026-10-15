import os

import pytest

from agentlayer.broker import Verdict
from agentlayer.tools import REGISTRY, ToolArgError, prepare_args


def run(env, tool, **args):
    t = REGISTRY[tool]
    prepared = prepare_args(t, args)
    decision = env.broker.authorize(t, prepared, env.ctx)
    assert decision.verdict is Verdict.ALLOW, decision.reason
    return t.run(prepared, env.ctx, decision)


def test_write_read_append(env):
    out = run(env, "write_file", path="notes/today.txt", content="line 1\n")
    assert out.ok and out.data["created"]
    assert (env.ws / "notes/today.txt").read_text() == "line 1\n"
    run(env, "write_file", path="notes/today.txt", content="line 2\n", mode="append")
    out = run(env, "read_file", path="notes/today.txt")
    assert out.data["content"] == "line 1\nline 2\n"


def test_overwrite_reports_diff(env):
    (env.ws / "f.txt").write_text("a\nb\nc\n")
    out = run(env, "write_file", path="f.txt", content="a\nB\nc\nd\n")
    assert out.data["lines_added"] == 2 and out.data["lines_removed"] == 1
    assert "-b" in out.display and "+B" in out.display


def test_read_file_chunks(env):
    (env.ws / "big.txt").write_text("x" * 10000)
    out = run(env, "read_file", path="big.txt", max_bytes=4000)
    assert out.data["truncated"] and out.data["next_offset"] == 4000
    out = run(env, "read_file", path="big.txt", offset=8000, max_bytes=4000)
    assert len(out.data["content"]) == 2000 and "truncated" not in out.data


def test_read_binary_file(env):
    (env.ws / "bin").write_bytes(b"\x7fELF\x00\x01\x02")
    out = run(env, "read_file", path="bin")
    assert out.data["binary"] and "content" not in out.data


def test_read_directory_fails_cleanly(env):
    (env.ws / "d").mkdir()
    assert not run(env, "read_file", path="d").ok


def test_list_directory(env):
    (env.ws / "sub").mkdir()
    (env.ws / "a.txt").write_text("hello")
    (env.ws / ".hidden").write_text("")
    out = run(env, "list_directory", path=".")
    assert out.data["entries"] == 2 and out.data["hidden_entries_not_shown"] == 1
    assert out.data["listing"].splitlines()[0].endswith("sub/")


def test_file_info(env):
    p = env.ws / "x.sh"
    p.write_text("#!/bin/sh\n")
    p.chmod(0o750)
    out = run(env, "file_info", path="x.sh")
    assert "0750" in out.data["permissions"] and out.data["inode"] == p.stat().st_ino


def test_search_skips_denied_files(env):
    (env.ws / "src").mkdir()
    (env.ws / "src" / "app.py").write_text("TOKEN = load()\n")
    (env.ws / "src" / ".env").write_text("TOKEN=supersecret\n")
    out = run(env, "search_files", path=".", contains="TOKEN", include_hidden=True)
    assert "app.py:1" in out.data["matches"]
    assert "supersecret" not in out.data["matches"] and out.data["skipped_by_policy"] == 1


def test_search_by_name(env):
    for name in ("a.py", "b.py", "c.txt"):
        (env.ws / name).write_text("")
    out = run(env, "search_files", name="*.py")
    assert out.data["count"] == 2


def test_move_and_copy(env):
    (env.ws / "a.txt").write_text("A")
    (env.ws / "dir").mkdir()
    run(env, "copy_path", source="a.txt", destination="dir")
    assert (env.ws / "dir/a.txt").read_text() == "A"
    run(env, "move_path", source="a.txt", destination="b.txt")
    assert not (env.ws / "a.txt").exists() and (env.ws / "b.txt").read_text() == "A"


def test_move_refuses_to_clobber_without_overwrite(env):
    (env.ws / "a").write_text("1")
    (env.ws / "b").write_text("2")
    t = REGISTRY["move_path"]
    d = env.broker.authorize(t, prepare_args(t, {"source": "a", "destination": "b"}), env.ctx)
    assert d.verdict is Verdict.DENY and "already exists" in d.reason


def test_delete_symlink_removes_link_not_target(env):
    target = env.ws / "keep.txt"
    target.write_text("important")
    (env.ws / "link").symlink_to(target)
    run(env, "delete_path", path="link")
    assert target.exists() and not (env.ws / "link").exists()


def test_delete_directory_requires_recursive(env):
    (env.ws / "d").mkdir()
    (env.ws / "d" / "f").write_text("x")
    t = REGISTRY["delete_path"]
    d = env.broker.authorize(t, prepare_args(t, {"path": "d"}), env.ctx)
    assert d.verdict is Verdict.DENY and "recursive=true" in d.reason
    out = run(env, "delete_path", path="d", recursive=True)
    assert out.data["files_removed"] == 1 and not (env.ws / "d").exists()


def test_write_refuses_swapped_symlink(env):
    """O_NOFOLLOW: a symlink planted after the check cannot redirect the write."""
    t = REGISTRY["write_file"]
    args = prepare_args(t, {"path": "victim.txt", "content": "x"})
    decision = env.broker.authorize(t, args, env.ctx)
    (env.ws / "victim.txt").symlink_to(env.root / "outside.txt")
    with pytest.raises(OSError):
        t.run(args, env.ctx, decision)
    assert not (env.root / "outside.txt").exists()


def test_disk_usage(env):
    (env.ws / "big").write_bytes(b"x" * 100_000)
    out = run(env, "disk_usage", path=".")
    assert "big" in out.data["largest"]


def test_prepare_args_validation():
    t = REGISTRY["read_file"]
    with pytest.raises(ToolArgError, match="missing"):
        prepare_args(t, {})
    assert prepare_args(t, {"path": "a", "offset": "10"})["offset"] == 10
    with pytest.raises(ToolArgError):
        prepare_args(t, {"path": "a", "offset": -1})
    with pytest.raises(ToolArgError):
        prepare_args(REGISTRY["write_file"], {"path": "a", "content": "b", "mode": "truncate"})
    assert prepare_args(REGISTRY["delete_path"], {"path": "a", "recursive": "true"})["recursive"] is True


def test_system_info(env):
    out = run(env, "system_info")
    assert out.data["kernel"].startswith("Linux") and out.data["processes"]["total"] > 0


def test_list_processes_and_info(env):
    out = run(env, "list_processes", sort_by="memory", limit=3)
    assert out.ok and len(out.data["table"].splitlines()) == 4
    info = run(env, "process_info", pid=os.getpid())
    assert info.data["identification"]["pid"] == os.getpid()
    assert "SCHED" in info.data["scheduling"]["policy"]


def test_swapped_parent_directory_is_refused(env):
    """openat(2) walk: an intermediate directory replaced by a symlink is caught too."""
    (env.ws / "sub").mkdir()
    outside = env.root / "outside"
    outside.mkdir()
    t = REGISTRY["write_file"]
    args = prepare_args(t, {"path": "sub/file.txt", "content": "x"})
    decision = env.broker.authorize(t, args, env.ctx)
    assert decision.verdict is Verdict.ALLOW
    (env.ws / "sub").rmdir()
    (env.ws / "sub").symlink_to(outside)
    with pytest.raises(OSError):
        t.run(args, env.ctx, decision)
    assert not (outside / "file.txt").exists()


def test_read_refuses_swapped_file(env):
    (env.ws / "doc.txt").write_text("public")
    t = REGISTRY["read_file"]
    args = prepare_args(t, {"path": "doc.txt"})
    decision = env.broker.authorize(t, args, env.ctx)
    (env.ws / "doc.txt").unlink()
    (env.ws / "doc.txt").symlink_to(env.root / "home/.ssh/id_rsa")
    with pytest.raises(OSError):
        t.run(args, env.ctx, decision)
