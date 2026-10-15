import os

import pytest

from agentlayer.policy import PolicyError, bundled_policy_names, canonical_pattern, glob_to_regex, load_policy


@pytest.mark.parametrize("pattern,path,expected", [
    ("/ws/**", "/ws", True),
    ("/ws/**", "/ws/a/b/c.txt", True),
    ("/ws/**", "/wsx/a", False),
    ("/ws/*.txt", "/ws/a.txt", True),
    ("/ws/*.txt", "/ws/sub/a.txt", False),
    ("**/.env", "/home/u/project/.env", True),
    ("**/.env", "/home/u/project/.envrc", False),
    ("/proc/*/environ", "/proc/123/environ", True),
    ("/proc/*/environ", "/proc/123/task/5/environ", False),
    ("/a/**/b", "/a/b", True),
    ("/a/**/b", "/a/x/y/b", True),
    ("/a/?", "/a/bc", False),
])
def test_glob_semantics(pattern, path, expected):
    assert (glob_to_regex(pattern).fullmatch(path) is not None) is expected


def test_canonical_pattern_resolves_symlinked_prefix(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real)
    assert canonical_pattern(f"{tmp_path}/link/**", tmp_path) == f"{real}/**"
    assert canonical_pattern("{workspace}/*.txt", real) == f"{real}/*.txt"
    assert canonical_pattern("**/.env", real) == "**/.env"


def test_relative_pattern_rejected(tmp_path):
    with pytest.raises(PolicyError):
        canonical_pattern("relative/**", tmp_path)


@pytest.mark.parametrize("name", bundled_policy_names())
def test_bundled_policies_load(name, tmp_path):
    policy = load_policy(name, workspace=str(tmp_path / "ws"))
    assert policy.workspace == (tmp_path / "ws").resolve()
    assert policy.has("fs.read") and policy.has("sys.read")


def test_phases_grow_monotonically(tmp_path):
    rights = [{g.right for g in load_policy(n, workspace=str(tmp_path / "ws")).grants} for n in bundled_policy_names()]
    assert rights[0] < rights[1] < rights[2] <= rights[3]
    assert "fs.write" not in rights[0]
    full = load_policy("phase4", workspace=str(tmp_path / "ws"))
    assert full.grant("fs.delete").confirm and full.grant("proc.signal").confirm


def _write(tmp_path, body):
    p = tmp_path / "p.toml"
    p.write_text(f'[workspace]\nroot = "{tmp_path}/ws"\n' + body)
    return str(p)


def test_unknown_right_rejected(tmp_path):
    with pytest.raises(PolicyError, match="unknown right"):
        load_policy(_write(tmp_path, '[[grant]]\nright = "fs.everything"\n'))


def test_unknown_key_rejected(tmp_path):
    with pytest.raises(PolicyError, match="unknown key"):
        load_policy(_write(tmp_path, '[[grant]]\nright = "fs.read"\npaths = ["/x"]\nprogams = []\n'))


def test_unknown_section_rejected(tmp_path):
    with pytest.raises(PolicyError, match="unknown section"):
        load_policy(_write(tmp_path, "[grants]\n"))


def test_home_as_workspace_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = tmp_path / "p.toml"
    p.write_text('[workspace]\nroot = "~"\n')
    with pytest.raises(PolicyError, match="dedicated directory"):
        load_policy(str(p))


def test_display_path(env):
    assert env.policy.display_path(str(env.ws / "a.txt")) == "{workspace}/a.txt"
    assert env.policy.display_path(os.path.expanduser("~/x")) == "~/x"
