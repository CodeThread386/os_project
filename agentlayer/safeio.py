"""Race-free access to broker-approved paths.

The broker approves a *canonical* path: absolute, with every symlink already
resolved. Between that check and the actual system call, another process
could replace a directory on the path with a symlink pointing elsewhere (a
time-of-check-to-time-of-use race). Opening the path normally would follow
the new link.

These helpers instead walk the approved path one component at a time with
``openat(2)`` and ``O_NOFOLLOW``, holding a directory file descriptor at each
step. A canonical path contains no symlinks, so meeting one means the tree
changed after the check: the kernel returns ``ELOOP`` and nothing is touched.
"""

from __future__ import annotations

import errno
import os
import shutil
from contextlib import contextmanager
from typing import Iterator

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _components(path: str) -> list[str]:
    if not os.path.isabs(path):
        raise ValueError(f"expected an absolute canonical path, got {path!r}")
    parts = [p for p in path.split("/") if p]
    if any(p in (".", "..") for p in parts):
        raise ValueError(f"path is not canonical: {path!r}")
    return parts


def _walk(parts: list[str], create: bool = False) -> int:
    """Return an fd for the directory /parts[0]/.../parts[-1], refusing symlinks."""
    fd = os.open("/", _DIR_FLAGS)
    try:
        for name in parts:
            try:
                nxt = os.open(name, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(name, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
                nxt = os.open(name, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def parent_dir(path: str, create: bool = False) -> Iterator[tuple[int, str]]:
    """Yield (fd of the parent directory, final component name)."""
    parts = _components(path)
    if not parts:
        raise IsADirectoryError(errno.EISDIR, "the root directory has no parent", path)
    fd = _walk(parts[:-1], create=create)
    try:
        yield fd, parts[-1]
    finally:
        os.close(fd)


def open_file(path: str, flags: int, mode: int = 0o644, create_parents: bool = False) -> int:
    """open(2) an approved path without following a symlink in any component."""
    parts = _components(path)
    if not parts:
        return os.open("/", flags | os.O_CLOEXEC)
    with parent_dir(path, create=create_parents) as (dfd, name):
        return os.open(name, flags | os.O_NOFOLLOW | os.O_CLOEXEC, mode, dir_fd=dfd)


def open_dir(path: str, create: bool = False) -> int:
    return _walk(_components(path), create=create)


def make_dirs(path: str) -> bool:
    """mkdir -p along an approved path. Returns True if the leaf was created."""
    existed = True
    try:
        os.close(_walk(_components(path)))
    except FileNotFoundError:
        existed = False
        os.close(_walk(_components(path), create=True))
    return not existed


def unlink(path: str) -> None:
    with parent_dir(path) as (dfd, name):
        os.unlink(name, dir_fd=dfd)  # removes a symlink itself, never its target


def rmdir(path: str) -> None:
    with parent_dir(path) as (dfd, name):
        os.rmdir(name, dir_fd=dfd)


def rmtree(path: str) -> None:
    with parent_dir(path) as (dfd, name):
        shutil.rmtree(name, dir_fd=dfd)  # fd-based walk: immune to symlink swaps inside the tree


def rename(src: str, dst: str) -> None:
    with parent_dir(src) as (sfd, sname), parent_dir(dst) as (dfd, dname):
        os.rename(sname, dname, src_dir_fd=sfd, dst_dir_fd=dfd)


def _copy_fd(in_fd: int, out_fd: int) -> int:
    total = 0
    try:
        while True:  # in-kernel copy: data never enters user space
            n = os.copy_file_range(in_fd, out_fd, 1 << 20)
            if n == 0:
                return total
            total += n
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            raise
    while True:  # fallback: read(2)/write(2), continuing from the current offsets
        chunk = os.read(in_fd, 1 << 20)
        if not chunk:
            return total
        view = memoryview(chunk)
        while view:
            view = view[os.write(out_fd, view):]
        total += len(chunk)


def copy_file(src: str, dst: str) -> int:
    """Copy file contents (and permission bits) between approved paths."""
    in_fd = open_file(src, os.O_RDONLY)
    try:
        mode = os.fstat(in_fd).st_mode & 0o777
        out_fd = open_file(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        try:
            total = _copy_fd(in_fd, out_fd)
            os.fsync(out_fd)
            return total
        finally:
            os.close(out_fd)
    finally:
        os.close(in_fd)
