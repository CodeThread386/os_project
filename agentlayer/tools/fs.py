"""File tools, implemented directly on the file-related system calls.

open(2)/read(2)/write(2)/fsync(2)/lseek(2) for content, stat(2)/lstat(2)
for metadata, getdents64(2) (through ``os.scandir``) for directories, and
rename(2)/unlink(2)/rmdir(2)/mkdir(2) for namespace changes. Files are
opened by their canonical path with ``O_NOFOLLOW``, so a symlink swapped in
after the broker's check makes the call fail instead of escaping.
"""

from __future__ import annotations

import difflib
import errno
import fnmatch
import grp
import os
import re
import shutil
import stat
import time

from agentlayer import procfs, safeio
from agentlayer.broker import Requirement
from agentlayer.tools.base import (
    Tool, ToolArgError, ToolContext, ToolOutput, fmt_bytes, fmt_time, param, register,
)

MAX_LIST = 200
MAX_WRITE = 1024 * 1024
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}


def _kind(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        return "device"
    return "other"


def _read_req(args: dict, ctx: ToolContext, key: str = "path", op: str = "read") -> list[Requirement]:
    return [Requirement("fs.read", path=ctx.real(args[key]), lexical=ctx.lexical(args[key]), op=op)]


# ------------------------------------------------------------ list_directory


def _list_directory(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    path = decision.approved().path
    rows, hidden = [], 0
    dfd = safeio.open_dir(path)
    try:
        for entry in os.scandir(dfd):  # getdents64(2) on the directory fd
            if not args["show_hidden"] and entry.name.startswith("."):
                hidden += 1
                continue
            st = entry.stat(follow_symlinks=False)  # fstatat(2) relative to dfd
            rows.append((_kind(st.st_mode), entry.name, st))
    finally:
        os.close(dfd)
    rows.sort(key=lambda r: (r[0] != "dir", r[1].lower()))
    total = len(rows)
    lines = []
    for kind, name, st in rows[:MAX_LIST]:
        suffix = "/" if kind == "dir" else ("@" if kind == "symlink" else "")
        size = "-" if kind == "dir" else fmt_bytes(st.st_size)
        lines.append(f"{stat.filemode(st.st_mode)} {size:>9}  {fmt_time(st.st_mtime)}  {name}{suffix}")
    listing = "\n".join(lines) or "(empty)"
    data = {"path": ctx.show(path), "entries": total, "listing": listing}
    if hidden:
        data["hidden_entries_not_shown"] = hidden
    if total > MAX_LIST:
        data["truncated"] = f"showing {MAX_LIST} of {total}"
    return ToolOutput(True, data, f"{total} entries in {ctx.show(path)}", listing)


register(Tool(
    name="list_directory",
    description="List a directory (like ls -l). Relative paths are inside the workspace.",
    rights=("fs.read",),
    params={
        "path": param("string", "Directory path", default="."),
        "show_hidden": param("boolean", "Include dotfiles", default=False),
    },
    requires=lambda a, c: _read_req(a, c, op="list"),
    run=_list_directory,
))


# ----------------------------------------------------------------- read_file


def _read_file(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    path = decision.approved().path
    limit = min(args["max_bytes"], ctx.policy.max_read_bytes)
    fd = safeio.open_file(path, os.O_RDONLY | os.O_NONBLOCK)  # O_NONBLOCK: a FIFO cannot hang us
    try:
        st = os.fstat(fd)
        if stat.S_ISDIR(st.st_mode):
            return ToolOutput.fail(f"{ctx.show(path)} is a directory; use list_directory")
        if not stat.S_ISREG(st.st_mode):
            return ToolOutput.fail(f"{ctx.show(path)} is a {_kind(st.st_mode)}, not a regular file")
        if args["offset"]:
            os.lseek(fd, args["offset"], os.SEEK_SET)
        chunks, remaining = [], limit + 1
        while remaining > 0:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    truncated = len(raw) > limit
    raw = raw[:limit]
    data: dict = {"path": ctx.show(path), "size": st.st_size, "offset": args["offset"]}
    if b"\0" in raw[:8192]:
        data.update(binary=True, first_bytes_hex=raw[:64].hex(" "))
        return ToolOutput(True, data, f"{ctx.show(path)} is binary ({fmt_bytes(st.st_size)})")
    text = raw.decode("utf-8", errors="replace")
    data["content"] = text
    if truncated:
        data["truncated"] = True
        data["next_offset"] = args["offset"] + len(raw)
    note = " (truncated; continue with next_offset)" if truncated else ""
    return ToolOutput(True, data, f"read {fmt_bytes(len(raw))} from {ctx.show(path)}{note}", text)


register(Tool(
    name="read_file",
    description="Read a text file. Large files are returned in chunks; pass next_offset to continue.",
    rights=("fs.read",),
    params={
        "path": param("string", "File path"),
        "offset": param("integer", "Byte offset to start at", default=0, minimum=0),
        "max_bytes": param("integer", "Maximum bytes to return", default=6000, minimum=1),
    },
    required=("path",),
    requires=_read_req,
    run=_read_file,
))


# ----------------------------------------------------------------- file_info


def _file_info(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    lex = ctx.lexical(args["path"])
    path = decision.approved().path
    lst = os.lstat(lex) if os.path.lexists(lex) else os.lstat(path)
    fd = safeio.open_file(path, os.O_PATH)
    try:
        st = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        group = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        group = str(st.st_gid)
    data = {
        "path": ctx.show(path),
        "type": _kind(st.st_mode),
        "size": f"{fmt_bytes(st.st_size)} ({st.st_size} bytes)",
        "permissions": f"{stat.filemode(st.st_mode)} ({stat.S_IMODE(st.st_mode):04o})",
        "owner": f"{procfs.username(st.st_uid)} ({st.st_uid})",
        "group": f"{group} ({st.st_gid})",
        "inode": st.st_ino,
        "hard_links": st.st_nlink,
        "device": f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}",
        "blocks_allocated": st.st_blocks,
        "modified": fmt_time(st.st_mtime),
        "accessed": fmt_time(st.st_atime),
        "status_changed": fmt_time(st.st_ctime),
        "you_can": "".join(c if os.access(path, m) else "-" for c, m in (("r", os.R_OK), ("w", os.W_OK), ("x", os.X_OK))),
    }
    if stat.S_ISLNK(lst.st_mode):
        data["symlink_to"] = os.readlink(lex)
    return ToolOutput(True, data, f"{data['type']} {ctx.show(path)}, {data['size']}, {data['permissions']}")


register(Tool(
    name="file_info",
    description="Show stat(2) metadata for a path: type, size, permissions, owner, inode, timestamps.",
    rights=("fs.read",),
    params={"path": param("string", "Path to inspect")},
    required=("path",),
    requires=lambda a, c: _read_req(a, c, op="stat"),
    run=_file_info,
))


# -------------------------------------------------------------- search_files


def _search_files(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    root = decision.approved().path
    if not os.path.isdir(root):
        return ToolOutput.fail(f"{ctx.show(root)} is not a directory")
    try:
        pattern = re.compile(args["contains"]) if args.get("contains") else None
    except re.error as exc:
        return ToolOutput.fail(f"invalid regular expression: {exc}")
    limit = args["max_results"]
    results: list[str] = []
    scanned = skipped = 0
    deadline = time.monotonic() + 8
    stopped = ""
    for dirpath, dirnames, filenames, dirfd in os.fwalk(root, follow_symlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and (args["include_hidden"] or not d.startswith(".")))
        for name in sorted(filenames):
            if not args["include_hidden"] and name.startswith("."):
                continue
            if not fnmatch.fnmatch(name, args["name"]):
                continue
            full = os.path.join(dirpath, name)
            try:
                st = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):  # skip symlinks, FIFOs, sockets, devices
                continue
            if not ctx.broker.readable(full, full):
                skipped += 1
                continue
            scanned += 1
            rel = os.path.relpath(full, root)
            if pattern is None:
                results.append(f"{rel} ({fmt_bytes(st.st_size)})")
            else:
                hits = _grep(name, dirfd, pattern)
                results.extend(f"{rel}:{n}: {line}" for n, line in hits)
            if len(results) >= limit:
                stopped = f"stopped at {limit} results"
                break
            if time.monotonic() > deadline or scanned > 20000:
                stopped = "stopped early: the tree is very large"
                break
        if stopped:
            break
    data = {"root": ctx.show(root), "matches": "\n".join(results[:limit]) or "(no matches)", "count": min(len(results), limit)}
    if stopped:
        data["note"] = stopped
    if skipped:
        data["skipped_by_policy"] = skipped
    what = "matching lines" if pattern else "files"
    return ToolOutput(True, data, f"{data['count']} {what} under {ctx.show(root)}", data["matches"])


def _grep(name: str, dirfd: int, pattern: re.Pattern, max_hits: int = 5) -> list[tuple[int, str]]:
    hits = []
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dirfd)
        with open(fd, "rb") as f:
            head = f.read(4096)
            if b"\0" in head:
                return []
            f.seek(0)
            for n, raw in enumerate(f, start=1):
                if n > 200_000:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if pattern.search(line):
                    hits.append((n, line.strip()[:200]))
                    if len(hits) >= max_hits:
                        break
    except OSError:
        pass
    return hits


register(Tool(
    name="search_files",
    description="Find files by name glob and/or search their contents with a regular expression (like find + grep).",
    rights=("fs.read",),
    params={
        "path": param("string", "Directory to search", default="."),
        "name": param("string", "File name glob, e.g. '*.py'", default="*"),
        "contains": param("string", "Optional regex to search inside files"),
        "include_hidden": param("boolean", "Also search dotfiles and dot-directories", default=False),
        "max_results": param("integer", "Maximum results", default=40, minimum=1, maximum=200),
    },
    requires=lambda a, c: _read_req(a, c, op="list"),
    run=_search_files,
))


# ---------------------------------------------------------------- write_file


def _write_target(args: dict, ctx: ToolContext) -> tuple[str, bool]:
    path = ctx.real(args["path"])
    if os.path.isdir(path):
        raise ToolArgError(f"{ctx.show(path)} is a directory")
    if len(args["content"].encode()) > MAX_WRITE:
        raise ToolArgError(f"content is larger than {fmt_bytes(MAX_WRITE)}")
    return path, os.path.isfile(path)


def _write_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    path, exists = _write_target(args, ctx)
    overwrite = exists and args["mode"] == "overwrite" and os.path.getsize(path) > 0
    return [Requirement("fs.write", path=path, lexical=ctx.lexical(args["path"]),
                        op=args["mode"], overwrite=overwrite)]


def _old_text(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            raw = f.read(MAX_WRITE + 1)
    except OSError:
        return None
    if b"\0" in raw[:8192] or len(raw) > MAX_WRITE:
        return None
    return raw.decode("utf-8", errors="replace")


def _diff(old: str, new: str, name: str, limit: int = 80) -> str:
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"a/{name}", f"b/{name}", n=2, lineterm=""))
    if len(lines) > limit:
        lines = lines[:limit] + [f"… {len(lines) - limit} more diff lines"]
    return "\n".join(lines) or "(no changes)"


def _write_preview(args: dict, ctx: ToolContext) -> str | None:
    path, exists = _write_target(args, ctx)
    if args["mode"] == "append":
        return f"append {len(args['content'].splitlines())} line(s) to {ctx.show(path)}"
    old = _old_text(path) if exists else None
    if old is None:
        return f"new file, {len(args['content'].splitlines())} line(s)" if not exists else "binary or very large file will be replaced"
    return _diff(old, args["content"], os.path.basename(path))


def _write_file(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    path = decision.approved().path
    existed = os.path.isfile(path)
    old = _old_text(path) if existed and args["mode"] == "overwrite" else None
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_APPEND if args["mode"] == "append" else os.O_TRUNC
    payload = args["content"].encode()
    pending = ctx.journal.prepare_write(path, args["mode"]) if ctx.journal else None  # backup first
    fd = safeio.open_file(path, flags, 0o644, create_parents=True)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)  # durable before we report success
    finally:
        os.close(fd)
    data = {"path": ctx.show(path), "bytes_written": len(payload), "created": not existed, "mode": args["mode"]}
    if pending is not None:
        entry = ctx.journal.commit(pending)
        data["undo_id"] = entry.id
        if entry.note:
            data["note"] = entry.note
    display = None
    if old is not None:
        diff = list(difflib.unified_diff(old.splitlines(), args["content"].splitlines(), lineterm="", n=0))
        data["lines_added"] = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
        data["lines_removed"] = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
        display = _diff(old, args["content"], os.path.basename(path), 40)
    verb = "created" if not existed else ("appended to" if args["mode"] == "append" else "overwrote")
    return ToolOutput(True, data, f"{verb} {ctx.show(path)} ({fmt_bytes(len(payload))})", display)


register(Tool(
    name="write_file",
    description="Create or overwrite a text file (or append to it). Parent directories are created as needed.",
    rights=("fs.write",),
    params={
        "path": param("string", "File path"),
        "content": param("string", "Full text to write"),
        "mode": param("string", "overwrite or append", enum=["overwrite", "append"], default="overwrite"),
    },
    required=("path", "content"),
    requires=_write_requires,
    run=_write_file,
    preview=_write_preview,
))


# ------------------------------------------------------------ make_directory


def _mkdir(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    path = decision.approved().path
    created = safeio.make_dirs(path)
    data = {"path": ctx.show(path), "created": created}
    if created and ctx.journal:
        data["undo_id"] = ctx.journal.record("mkdir", path, summary=f"created folder {path}").id
    verb = "created" if created else "already existed"
    return ToolOutput(True, data, f"{ctx.show(path)} {verb}")


register(Tool(
    name="make_directory",
    description="Create a directory (and any missing parents).",
    rights=("fs.write",),
    params={"path": param("string", "Directory path")},
    required=("path",),
    requires=lambda a, c: [Requirement("fs.write", path=c.real(a["path"]), lexical=c.lexical(a["path"]), op="mkdir")],
    run=_mkdir,
))


# --------------------------------------------------------- move / copy paths


def _destination(args: dict, ctx: ToolContext, source: str) -> tuple[str, bool]:
    dest = ctx.real(args["destination"])
    if os.path.isdir(dest) and not os.path.islink(dest):
        dest = os.path.join(dest, os.path.basename(source))
    exists = os.path.lexists(dest)
    if exists and not args["overwrite"]:
        raise ToolArgError(f"{ctx.show(dest)} already exists; set overwrite=true to replace it")
    if dest == source:
        raise ToolArgError("source and destination are the same")
    return dest, exists


def _move_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    src = ctx.real(args["source"], follow=False)
    if not os.path.lexists(src):
        raise ToolArgError(f"{ctx.show(src)} does not exist")
    dest, exists = _destination(args, ctx, src)
    return [
        Requirement("fs.write", path=src, lexical=ctx.lexical(args["source"]), op="move-from"),
        Requirement("fs.write", path=dest, lexical=ctx.lexical(args["destination"]), op="move-to", overwrite=exists),
    ]


def _move(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    src, dest = decision.approved(0).path, decision.approved(1).path
    try:
        safeio.rename(src, dest)  # rename(2): atomic within one filesystem
    except OSError as exc:
        if exc.errno != errno.EXDEV:  # crosses filesystems -> copy + unlink
            raise
        shutil.move(src, dest)
    data = {"from": ctx.show(src), "to": ctx.show(dest)}
    if ctx.journal:
        data["undo_id"] = ctx.journal.record("move", src, dest, f"moved {src} to {dest}").id
    return ToolOutput(True, data, f"moved {ctx.show(src)} -> {ctx.show(dest)}")


register(Tool(
    name="move_path",
    description="Move or rename a file or directory. If the destination is a directory, the source moves into it.",
    rights=("fs.write",),
    params={
        "source": param("string", "Existing path"),
        "destination": param("string", "New path or target directory"),
        "overwrite": param("boolean", "Replace an existing destination", default=False),
    },
    required=("source", "destination"),
    requires=_move_requires,
    run=_move,
    preview=lambda a, c: f"{c.show(c.real(a['source'], follow=False))} -> {c.show(c.real(a['destination']))}",
))


def _copy_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    src = ctx.real(args["source"])
    if not os.path.exists(src):
        raise ToolArgError(f"{ctx.show(src)} does not exist")
    dest, exists = _destination(args, ctx, src)
    return [
        Requirement("fs.read", path=src, lexical=ctx.lexical(args["source"]), op="read"),
        Requirement("fs.write", path=dest, lexical=ctx.lexical(args["destination"]), op="copy-to", overwrite=exists),
    ]


def _copy(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    src, dest = decision.approved(0).path, decision.approved(1).path
    exists = os.path.lexists(dest)
    if os.path.isdir(src):
        if any(not ctx.broker.readable(os.path.realpath(os.path.join(d, f)), os.path.join(d, f))
               for d, _, files in os.walk(src) for f in files):
            return ToolOutput.fail("the directory contains files you may not read; copy them individually")
        shutil.copytree(src, dest, symlinks=True, dirs_exist_ok=exists)
    else:
        safeio.copy_file(src, dest)  # copy_file_range(2): the kernel copies the data
    data = {"from": ctx.show(src), "to": ctx.show(dest)}
    if ctx.journal and not exists:
        data["undo_id"] = ctx.journal.record("copy", src, dest, f"copied {src} to {dest}").id
    return ToolOutput(True, data, f"copied {ctx.show(src)} -> {ctx.show(dest)}")


register(Tool(
    name="copy_path",
    description="Copy a file or a directory tree.",
    rights=("fs.read", "fs.write"),
    params={
        "source": param("string", "Existing path"),
        "destination": param("string", "New path or target directory"),
        "overwrite": param("boolean", "Replace an existing destination", default=False),
    },
    required=("source", "destination"),
    requires=_copy_requires,
    run=_copy,
))


# --------------------------------------------------------------- delete_path


def _delete_target(args: dict, ctx: ToolContext) -> tuple[str, bool]:
    path = ctx.real(args["path"], follow=False)
    if not os.path.lexists(path):
        raise ToolArgError(f"{ctx.show(path)} does not exist")
    is_dir = os.path.isdir(path) and not os.path.islink(path)
    if is_dir and not args["recursive"] and os.listdir(path):
        raise ToolArgError(f"{ctx.show(path)} is a non-empty directory; set recursive=true to delete it and its contents")
    return path, is_dir


def _delete_requires(args: dict, ctx: ToolContext) -> list[Requirement]:
    path, is_dir = _delete_target(args, ctx)
    return [Requirement("fs.delete", path=path, lexical=ctx.lexical(args["path"]), op="delete", recursive=is_dir)]


def _tree_summary(path: str) -> tuple[int, int, int, list[str]]:
    files = dirs = size = 0
    sample: list[str] = []
    for dirpath, dirnames, filenames in os.walk(path):
        dirs += len(dirnames)
        for name in filenames:
            files += 1
            full = os.path.join(dirpath, name)
            try:
                size += os.lstat(full).st_size
            except OSError:
                pass
            if len(sample) < 8:
                sample.append(os.path.relpath(full, path))
    return files, dirs, size, sample


def _delete_preview(args: dict, ctx: ToolContext) -> str | None:
    path, is_dir = _delete_target(args, ctx)
    fate = ("will be PERMANENTLY deleted" if args.get("permanent") or not ctx.journal
            else "will be moved to the Trash (undo brings it back)")
    if not is_dir:
        kind = "symlink" if os.path.islink(path) else "file"
        return f"{kind} {ctx.show(path)} ({fmt_bytes(os.lstat(path).st_size)}) {fate}"
    files, dirs, size, sample = _tree_summary(path)
    listing = "\n".join(f"  {s}" for s in sample) + ("\n  …" if files > len(sample) else "")
    return f"directory {ctx.show(path)} with {files} file(s) in {dirs} subdirectories ({fmt_bytes(size)}) " \
           f"{fate}:\n{listing}"


def _delete(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    path = decision.approved().path
    if ctx.journal and not args.get("permanent"):
        is_dir = stat.S_ISDIR(os.lstat(path).st_mode)
        files = _tree_summary(path)[0] if is_dir else 1
        entry = ctx.journal.trash(path)  # rename(2) into ~/.local/share/Trash
        data = {"trashed": ctx.show(path), "undo_id": entry.id, "files": files,
                "note": "moved to the Trash; undo (or the file manager's Trash) restores it"}
        return ToolOutput(True, data, f"moved {ctx.show(path)} to the Trash (undo id {entry.id})")
    if not stat.S_ISDIR(os.lstat(path).st_mode):
        safeio.unlink(path)  # unlink(2) removes a symlink itself, never its target
        return ToolOutput(True, {"deleted": ctx.show(path)}, f"deleted {ctx.show(path)}")
    files, dirs, size, _ = _tree_summary(path)
    if args["recursive"]:
        safeio.rmtree(path)
    else:
        safeio.rmdir(path)
    data = {"deleted": ctx.show(path), "files_removed": files, "directories_removed": dirs + 1, "bytes_freed": size}
    return ToolOutput(True, data, f"deleted directory {ctx.show(path)} ({files} files)")


register(Tool(
    name="delete_path",
    description="Delete a file, symlink or directory (it goes to the Trash, so undo can restore it). "
                "Non-empty directories need recursive=true.",
    rights=("fs.delete",),
    params={
        "path": param("string", "Path to delete"),
        "recursive": param("boolean", "Delete a directory and everything in it", default=False),
        "permanent": param("boolean", "Skip the Trash (cannot be undone); only if the user asks", default=False),
    },
    required=("path",),
    requires=_delete_requires,
    run=_delete,
    preview=_delete_preview,
    destructive=True,
))


# ---------------------------------------------------------------- disk_usage


def _disk_usage(args: dict, ctx: ToolContext, decision) -> ToolOutput:
    path = decision.approved().path
    vfs = os.statvfs(path)
    total = vfs.f_blocks * vfs.f_frsize
    free = vfs.f_bavail * vfs.f_frsize
    used = total - vfs.f_bfree * vfs.f_frsize
    data: dict = {
        "path": ctx.show(path),
        "filesystem": f"{fmt_bytes(used)} used of {fmt_bytes(total)} ({used / total * 100 if total else 0:.0f}%), "
                      f"{fmt_bytes(free)} available",
        "inodes": f"{vfs.f_files - vfs.f_ffree} used of {vfs.f_files}",
    }
    if os.path.isdir(path):
        sizes = []
        budget = [50_000]

        def du(p: str) -> int:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(p):
                for name in filenames:
                    budget[0] -= 1
                    if budget[0] < 0:
                        return total_size
                    try:
                        total_size += os.lstat(os.path.join(dirpath, name)).st_blocks * 512
                    except OSError:
                        pass
            return total_size

        with os.scandir(path) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False):
                    sizes.append((du(entry.path), entry.name + "/"))
                else:
                    sizes.append((entry.stat(follow_symlinks=False).st_blocks * 512, entry.name))
        sizes.sort(reverse=True)
        data["largest"] = "\n".join(f"{fmt_bytes(s):>10}  {n}" for s, n in sizes[: args["top"]]) or "(empty)"
        data["total"] = fmt_bytes(sum(s for s, _ in sizes))
        if budget[0] < 0:
            data["note"] = "stopped counting after 50,000 files; totals are lower bounds"
    return ToolOutput(True, data, f"{ctx.show(path)}: {data.get('total', '')} — filesystem {data['filesystem']}",
                      data.get("largest"))


register(Tool(
    name="disk_usage",
    description="Filesystem space (statvfs) and the largest entries inside a directory (like df + du).",
    rights=("fs.read",),
    params={
        "path": param("string", "Directory or file", default="."),
        "top": param("integer", "How many of the largest entries to list", default=10, minimum=1, maximum=50),
    },
    requires=lambda a, c: _read_req(a, c, op="stat"),
    run=_disk_usage,
))
