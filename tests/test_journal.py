"""Undo journal and the freedesktop Trash."""

import os
import time

import pytest

from agentlayer.broker import Verdict
from agentlayer.journal import Trash, UndoError
from agentlayer.tools import REGISTRY, prepare_args

from .conftest import make_env


def run(env, tool, expect=Verdict.ALLOW, **args):
    t = REGISTRY[tool]
    prepared = prepare_args(t, args)
    d = env.broker.authorize(t, prepared, env.ctx)
    assert d.verdict is expect, d.reason
    return t.run(prepared, env.ctx, d)


@pytest.fixture
def jenv(tmp_path):
    return make_env(tmp_path, journal=True)


def test_overwrite_is_undoable(jenv):
    f = jenv.ws / "notes.txt"
    f.write_text("original\n")
    out = run(jenv, "write_file", path="notes.txt", content="replaced\n")
    assert f.read_text() == "replaced\n" and out.data["undo_id"]
    assert "restored" in jenv.ctx.journal.undo()
    assert f.read_text() == "original\n"


def test_append_undo_truncates(jenv):
    f = jenv.ws / "log.txt"
    f.write_text("one\n")
    run(jenv, "write_file", path="log.txt", content="two\n", mode="append")
    jenv.ctx.journal.undo()
    assert f.read_text() == "one\n"


def test_create_undo_moves_new_file_to_trash(jenv):
    run(jenv, "write_file", path="new.txt", content="x")
    jenv.ctx.journal.undo()
    assert not (jenv.ws / "new.txt").exists()
    assert (jenv.root / "trash" / "files" / "new.txt").read_text() == "x"


def test_delete_goes_to_trash_and_comes_back(jenv):
    d = jenv.ws / "project"
    d.mkdir()
    (d / "a.txt").write_text("A")
    out = run(jenv, "delete_path", path="project", recursive=True)
    assert out.data["undo_id"] and not d.exists()
    info = (jenv.root / "trash" / "info" / "project.trashinfo").read_text()
    assert "[Trash Info]" in info and f"Path={d}".replace(" ", "%20") in info and "DeletionDate=" in info
    assert "restored" in jenv.ctx.journal.undo(out.data["undo_id"])
    assert (d / "a.txt").read_text() == "A" and not (jenv.root / "trash" / "info" / "project.trashinfo").exists()


def test_permanent_delete_skips_trash(jenv):
    (jenv.ws / "tmp.txt").write_text("x")
    out = run(jenv, "delete_path", path="tmp.txt", permanent=True)
    assert "undo_id" not in out.data and not list((jenv.root / "trash" / "files").iterdir())


def test_trash_name_collisions(tmp_path):
    trash = Trash(tmp_path / "trash")
    for i in range(3):
        f = tmp_path / "same.txt"
        f.write_text(str(i))
        trash.put(str(f))
    names = sorted(p.name for p in (tmp_path / "trash" / "files").iterdir())
    assert names == ["same.txt", "same.txt.2", "same.txt.3"]


def test_move_copy_mkdir_undo(jenv):
    (jenv.ws / "a.txt").write_text("A")
    run(jenv, "move_path", source="a.txt", destination="b.txt")
    jenv.ctx.journal.undo()
    assert (jenv.ws / "a.txt").exists() and not (jenv.ws / "b.txt").exists()
    run(jenv, "copy_path", source="a.txt", destination="c.txt")
    jenv.ctx.journal.undo()
    assert not (jenv.ws / "c.txt").exists()
    run(jenv, "make_directory", path="newdir")
    jenv.ctx.journal.undo()
    assert not (jenv.ws / "newdir").exists()


def test_undo_refuses_when_file_changed_since(jenv):
    f = jenv.ws / "doc.txt"
    f.write_text("v1")
    run(jenv, "write_file", path="doc.txt", content="v2")
    time.sleep(0.01)
    f.write_text("v3 written by the user")
    with pytest.raises(UndoError, match="changed after"):
        jenv.ctx.journal.undo()
    assert f.read_text() == "v3 written by the user"


def test_undo_tool_goes_through_the_broker(tmp_path):
    env = make_env(tmp_path, confirm=True, journal=True)
    f = env.ws / "a.txt"
    f.write_text("old")
    t = REGISTRY["write_file"]
    args = prepare_args(t, {"path": "a.txt", "content": "new"})
    t.run(args, env.ctx, env.broker.authorize(t, args, env.ctx))
    out = run(env, "undo", expect=Verdict.CONFIRM, action="undo")  # restoring overwrites: the policy asks
    assert out.ok and f.read_text() == "old"
    listing = run(env, "undo", action="list")
    assert "(undone)" in listing.data["changes"]


def test_nothing_to_undo(jenv):
    with pytest.raises(UndoError, match="nothing to undo"):
        jenv.ctx.journal.undo()
    t = REGISTRY["undo"]
    d = jenv.broker.authorize(t, prepare_args(t, {"action": "undo"}), jenv.ctx)
    assert d.verdict is Verdict.DENY and "nothing to undo" in d.reason


def test_journal_files_are_private(jenv):
    (jenv.ws / "x").write_text("1")
    run(jenv, "write_file", path="x", content="2")
    assert os.stat(jenv.root / "undo" / "journal.json").st_mode & 0o777 == 0o600
