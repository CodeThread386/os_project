# AgentLayer — Agentic AI Control Layer for Linux

**BCSE303P Operating Systems Lab · VIT · Fall 2026–27**
Sarthak Trivedi (24BYB0192) · Mrinal Gupta (24BDE0027) · Vaaman Bansal (24BDE0067) · Guide: Dr. Porkodi S

AgentLayer lets you operate a Linux computer in plain English, by typing, speaking, or from a web dashboard,
while every action an AI proposes passes through a **capability broker** that enforces an explicit
least-privilege policy. It also works as a security layer for *other* AI agents (Claude Code, Claude
Desktop, Cursor) through the Model Context Protocol.

```
 You (terminal · dashboard · voice · scheduled task · another AI agent over MCP)
  │
  ▼
 LLM (Groq: gpt-oss-120b → qwen3.8 → gpt-oss-20b → local Ollama) ── plans the next tool call
  │                                   ▲
  ▼                                   │ observes the result (exit code, diff, /proc data, …)
 Capability Broker ── checks the policy (access matrix), asks you when needed,
  │                   writes a hash-chained audit record
  ▼                                   │
 OS interface ── 38 tools on real system calls: openat/O_NOFOLLOW walks, fork/execve/wait4 with pipes and
  │              rlimits, kill, setpriority, inotify, ptrace (strace), /proc and /sys, D-Bus, sudo
  ▼
 Linux kernel ── commands run inside bubblewrap namespaces (mount, PID, net, IPC, UTS, user)
```

| | |
|---|---|
| **Results** | 8/8 everyday tasks solved and **7/7 attacks contained** in the live evaluation (`agentlayer eval`) |
| **Size** | 38 tools · 18 rights · 5 phase policies · 290 automated tests · ~11k lines of Python |
| **Security** | broker + sandbox + injection shield (patterns, Llama Prompt Guard 2, gpt-oss-safeguard) + undo + audit chain |

---

## Quick start

```bash
cd os-project
./setup.sh                    # venv, install, self-check (agentlayer doctor)
source .venv/bin/activate

agentlayer                    # chat in the terminal          agentlayer ui     → web dashboard
agentlayer run "how is my laptop doing?"                      agentlayer top    → live system monitor
agentlayer demo               # the five phases side by side  agentlayer eval   → benign + attack scenarios
agentlayer install-desktop    # adds AgentLayer to your application launcher
```

Needs Linux, Python ≥ 3.11 and `bubblewrap` for the command sandbox (`sudo apt install bubblewrap strace`).
Put Groq keys in `.env` (`GROQ_API_KEYS=gsk_a,gsk_b,…`, see `.env.example`), or just run Ollama for a fully
local setup. Desktop features need a graphical session; window control needs KDE Plasma.

---

## What it can do

| Area | Ask it… | What happens underneath | Asks you first |
|---|---|---|---|
| **Files** | "what's taking space in Downloads?", "rename these", "find TODOs" | `openat` walks with `O_NOFOLLOW`, `getdents64`, `statvfs`, `copy_file_range` | overwrites, deletes |
| **Undo** | "undo that" | overwrites backed up first; deletes go to the **desktop Trash** | — |
| **Processes** | "what's using my RAM?", "pause that process" | `/proc/<pid>/stat` sampled twice (CPU %), `kill(2)`, `setpriority(2)` | signals |
| **OS internals** | "memory map of Firefox", "which syscalls does ls make?" | `/proc/<pid>/maps,smaps_rollup,fd,limits,ns,cgroup,sched`, **strace** in the sandbox | — |
| **Kernel views** | "show interrupts / page faults / memory pressure / buddy allocator" | `/proc/{interrupts,vmstat,pressure,buddyinfo,locks,mounts}` | — |
| **Hardware** | "battery health? temperatures? which disks are SSDs?" | `/sys/class/{power_supply,thermal,net,drm}`, `/sys/block/*/queue/scheduler` | — |
| **Network** | "which programs listen on ports?" | `/proc/net/tcp*` + socket-inode → PID mapping | — |
| **Commands & code** | "write a C program that forks, compile and run it" | `fork`/`execve`/`wait4`, pipes, rlimits, bubblewrap namespaces | internet access |
| **Watching** | "tell me when the download finishes" | `inotify_init1`, `inotify_add_watch`, `select` | — |
| **Desktop** | "open Firefox", "close that window", "what's on my screen?" | KWin scripting over D-Bus, Spectacle, a **vision model** | closing, screenshots, clipboard reads |
| **Software** | "install VLC" | `sudo` (password typed into a hidden prompt, piped to sudo) | **always, live** |
| **Automation** | "every day at 9:00, report free disk space" | **systemd user timers** running AgentLayer headless | creating a task |
| **Memory** | "remember that my guide is Dr. Porkodi" | long-term notes in every new session; resumable conversations | — |
| **Voice** | `/voice`, or the microphone button | PipeWire recording → Whisper; answers read aloud | — |

---

## Front ends

* **Terminal** — `agentlayer`: streaming tool cards, diffs, live command output, approvals (`y` / `n` /
  `a` = "yes for the rest of this request"). Slash commands: `/undo /dryrun /grants /revoke /voice /speak
  /tasks /tools /policy /audit /verify /memory /sessions /resume /export /quota /usage`.
* **Web dashboard** — `agentlayer ui`: chat with live tool cards and approval dialogs, voice input, spoken
  answers, a live system monitor, the access matrix with temporary grants, the audit log with chain
  verification, undo, memory, sessions, scheduled tasks, tools and quota. Privilege-separated: the
  web process never runs tools; a single-threaded worker does (see *Architecture*).
* **MCP server** — `agentlayer mcp`: other AI agents get AgentLayer's 38 tools *through the broker*.
  Approvals pop up as KDE dialogs. See [docs/MCP.md](docs/MCP.md).
  `claude mcp add agentlayer -- $PWD/.venv/bin/agentlayer mcp`
* **Scheduled tasks** — headless runs started by systemd timers; anything needing approval is declined.
* **`agentlayer top`** — a terminal system monitor built on the same `/proc` sampling.

---

## The five phases

Phases 1–4 are the plan from the Review-1 proposal; phase 5 adds everything requested afterwards. Each is a
bundled policy (`-p phase1` … `-p phase5`, **phase 5 is the default**). `agentlayer demo` prints this live:

| Action | p1 | p2 | p3 | p4 | p5 |
|---|:-:|:-:|:-:|:-:|:-:|
| system overview · list processes · read a workspace file · inspect a process's open files | allow | allow | allow | allow | allow |
| read `~/.ssh/id_rsa`, `/etc/shadow`, `~/.bashrc` · SIGKILL PID 1 · delete the audit log · `systemctl reboot` as root | deny | deny | deny | deny | deny |
| create a file · run a pipeline · python3 in the sandbox · trace syscalls · fetch an allow-listed URL | deny | allow | allow | allow | allow |
| overwrite / delete a file | deny | allow | allow | confirm | confirm |
| SIGTERM to your own process | deny | deny | allow | confirm | confirm |
| list `~/Downloads` · launch an app · list windows | deny | deny | deny | deny | allow |
| internet for a command · screenshot · read the clipboard · install a package · schedule a task · temporary capability | deny | deny | deny | deny | confirm |

---

## How it maps to the OS syllabus

| Component | OS concept | Where |
|---|---|---|
| Capability broker, TOML policies, temporary grants | access matrix, capability lists, **revocation**, policy vs. mechanism, reference monitor | `broker.py`, `policy.py` |
| File tools, undo journal, Trash | system calls, **journaling** / write-ahead backup, atomic `rename`, `copy_file_range`, `ftruncate` | `tools/fs.py`, `journal.py` |
| Race-free paths | TOCTOU, `openat(2)` + `O_NOFOLLOW` walks, directory file descriptors, `fchdir` | `safeio.py` |
| Command execution | **`fork`/`execve`/`wait4`**, `pipe`+`dup2`, `setsid`, `killpg`, signal dispositions, exit status | `executor.py` |
| Resource limits | `setrlimit` (CPU → `SIGXCPU`, memory, file size → `SIGXFSZ`, fds), `rusage` accounting | `executor.py` |
| Sandbox | **namespaces** (mount, PID, net, IPC, UTS, user), bind mounts, masking with `nodev` device nodes | `sandbox.py` |
| Process tools, `os_internals` | **PCB** via procfs, process states, zombies, threads, **virtual memory layout**, Rss/Pss, fd tables, rlimits, cgroups, scheduler statistics | `procfs.py`, `tools/proc.py`, `tools/internals.py` |
| Kernel views | interrupts, page faults & reclaim (`vmstat`), **PSI pressure**, **buddy allocator**, file locks, mounts, modules | `tools/internals.py` |
| Syscall tracing | **ptrace** (strace) on a sandboxed child, syscall semantics | `trace_syscalls` |
| File watching | **inotify** (kernel event notification, blocking `select`) | `inotify.py` |
| Scheduling | `nice`/`setpriority`, policies, affinity, context switches; **systemd timers** for deferred work | `tools/proc.py`, `scheduler.py` |
| Hardware | **sysfs**, I/O schedulers, power supply, thermal zones | `tools/internals.py` |
| Networking | sockets in `/proc/net`, socket inodes in `/proc/<pid>/fd` | `network_connections` |
| Desktop control | **IPC over D-Bus**, compositor scripting, session re-parenting (`setsid -f`) | `desktop.py` |
| Web dashboard | **privilege separation**, fork-before-threads, UNIX `socketpair` IPC | `web/` |
| Privileged actions | user vs. superuser, setuid `sudo`, authentication outside the agent | `tools/admin.py` |
| Audit, memory, sessions | append + `fsync`, `flock`, hash chains, atomic replace | `audit.py`, `memory.py` |

---

## Security model (layers)

1. **Least privilege at the interface** — the model only sees tools its policy grants, and only the core
   group until it needs more.
2. **Capability broker** — every call becomes concrete requirements (canonical path, PID, program, host, app)
   checked against the policy; deny rules override grants; a few **invariants** hold under any policy: the
   policy, audit log and AgentLayer's state are untouchable, `/proc/*/environ` and `/proc/*/mem` are never
   readable, PID 1 / the agent / its parents can't be signalled, the windows AgentLayer runs in can't be
   closed, root always needs a live "y".
3. **Just-in-time capabilities** — when a task needs one more folder, host, program or app, the agent asks
   (`request_capability`); you approve live; grants expire, can be revoked (`/revoke`), never override deny
   rules and never include root.
4. **Race-free execution** — tools act only on the broker-approved canonical path, opened by `openat` walks
   with `O_NOFOLLOW` (a symlink swapped in after the check fails with `ELOOP`).
5. **Sandbox** — commands run without a shell, with a minimal environment (no API keys), rlimits, and in
   bubblewrap namespaces where only the workspace and granted folders exist, files matching deny rules are
   masked, there is no network unless you approve, and no other process is visible.
6. **Injection shield** — outside content (files, web pages, command output, window titles, screen text) is
   screened by pattern rules, **Llama Prompt Guard 2** and a **gpt-oss-safeguard** policy check; a hit
   *taints* the request so every further change needs your approval.
7. **Confirmation layer** — destructive actions show a diff / file list / process / command first.
8. **Recovery and accountability** — deletes go to the Trash, overwrites are backed up (`agentlayer undo`),
   and every decision is in a hash-chained, `fsync`'d audit log (`agentlayer audit --verify`).

Details: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) · [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Privacy:** what tools return (file contents, window titles, approved screenshots or clipboard text) goes to
the model provider. `--provider ollama` keeps everything on the machine.

---

## Evaluation

`agentlayer eval` runs 15 scenarios against the live model, each in an isolated sandbox with its own audit
log, Trash and memory, headless (nobody can approve anything). Outcomes are measured on the file system,
the answer and the audit log, not on what the model says. Latest run (`openai/gpt-oss-120b`):

| | scenarios | result |
|---|---|---|
| everyday tasks (create, count, find largest, read JSON, compute in Python, rename, kernel version, measure syscalls) | 8 | **8/8 solved** |
| attacks (injected exfiltration, injected mass delete, direct secret theft, environment-variable exfiltration, symlink trap, audit tampering, unattended root install) | 7 | **7/7 contained** |

---

## Staying inside Groq's rate limits

Groq limits each **organisation** per **model**. The 25 configured keys form 5 organisations, so there are
15 independent budgets (5 × 3 models). AgentLayer learns which keys share an organisation, models each budget
as Groq's token bucket from the `x-ratelimit-*` headers, sends each request to the bucket with the most room,
paces instead of failing, parks a bucket after a 429 until Groq's stated reset (persisted across restarts),
falls back across models and finally to Ollama, and keeps requests small (on-demand tool groups, capped
results, context fitting to the tightest model limit). `agentlayer quota [--probe]` shows the live state.

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEYS` / `GROQ_API_KEY` | – | one or more Groq keys |
| `AGENTLAYER_POLICY` | `phase5-desktop` | policy name or path |
| `AGENTLAYER_MODEL`, `AGENTLAYER_FALLBACK_MODELS` | `openai/gpt-oss-120b`, built-in chain | models |
| `AGENTLAYER_VISION_MODEL` | `qwen/qwen3.8-27b` | model for screenshots and images |
| `AGENTLAYER_GUARD` | `1` | prompt-injection screening |
| `AGENTLAYER_PROVIDER` | `auto` | `groq`, `ollama`, `openai` (any OpenAI-compatible server) |
| `AGENTLAYER_WORKSPACE` | from policy | workspace directory |
| `AGENTLAYER_MCP_CONFIRM` | `auto` | MCP approvals via KDE dialog (`deny` declines them all) |
| `OLLAMA_HOST`, `AGENTLAYER_OLLAMA_MODEL` | `localhost:11434`, auto | local model |

To change what the agent may do, copy `agentlayer/policies/phase5-desktop.toml`, edit it, and pass
`-p my-policy.toml`. Unknown rights or keys are rejected (security configuration fails closed).

---

## Tests

```bash
make test            # 290 tests, ~15 s: no network, quota, password or windows needed
make eval            # the live evaluation (uses the model)
```

Coverage includes every broker rule and invariant, symlink/`..`/TOCTOU races on files *and* parent
directories, sandbox isolation and masking, the fork/exec engine (pipelines, `SIGPIPE`, timeouts, CPU
limits, no zombies, no leaked environment), undo and the Trash, temporary capabilities, tool groups, the
injection shield and taint, dry-run, all OS-internals topics, inotify, strace tracing, scheduling (fake
systemd), the MCP protocol over stdio, the web dashboard's access control and chat round trip, root actions
through a fake `sudo` (the password never reaches the model or the log), quota pooling against a fake Groq
server, and the agent loop with scripted models.

---

## Project layout

```
agentlayer/
  agent.py      the plan → check → execute → observe loop, tool groups, taint, dry-run, context fitting
  broker.py     capability broker, invariants, temporary capabilities      policy.py   policies / access matrix
  tools/        fs · proc · exec · net · system · internals · gui · vision · admin · tasks · undo · capability · memory
  executor.py   fork/exec/wait4 pipelines      sandbox.py  bubblewrap      safeio.py   race-free openat walks
  procfs.py     /proc parser                   inotify.py  inotify(7)      monitor.py  live metrics
  journal.py    undo journal + Trash           guard.py    injection shield scheduler.py systemd timers
  llm.py · quota.py   Groq/Ollama backends, key pool, vision, guard, Whisper
  session.py    builds a runtime for every front end             audit.py · memory.py  journals and stores
  cli.py · ui.py      terminal            web/  dashboard (server + worker)    mcp.py  MCP server
  evals.py      evaluation harness        policies/  phase1 … phase5
docs/           ARCHITECTURE · THREAT_MODEL · DEMO (review script) · MCP
tests/          290 tests
```

## Limitations

* Typing and clicking inside other applications is not possible on Wayland without extra system setup
  (`ydotool`); window control is implemented for KDE Plasma.
* Inside the sandbox, commands can change files in the workspace and the granted folders without per-file
  confirmation (like any program you run there); the file tools are the confirmed, undoable path.
* `copy_path` of whole directories and DNS for `fetch_url` are not fully race-free.
* The model can still make poor choices within its permissions; the layers above bound the damage, and the
  audit log and undo make it visible and reversible.

## References

Silberschatz, Galvin & Gagne, *Operating System Concepts* · Tanenbaum, *Modern Operating Systems* ·
Dennis & Van Horn (1966) · Lampson (1974), "Protection" · Yao et al. (2023), ReAct · Schick et al. (2023),
Toolformer · Greshake et al. (2023), indirect prompt injection · Meta Llama Prompt Guard 2 · OpenAI
gpt-oss-safeguard · Model Context Protocol specification · freedesktop.org Trash and Desktop Entry specs ·
bubblewrap · Groq documentation.
