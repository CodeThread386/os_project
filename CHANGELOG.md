# Changelog

## 2.0.0

**New front ends**
* Web dashboard (`agentlayer ui`): chat with live tool cards and approval dialogs, voice input, spoken answers,
  live system monitor, access matrix, audit log with chain verification, undo, memory, sessions, tasks, tools,
  quota. Privilege-separated web process and agent worker.
* MCP server (`agentlayer mcp`) so other AI agents use AgentLayer's tools through the broker.
* `agentlayer top` (terminal system monitor), `agentlayer eval` (live evaluation), `agentlayer undo`,
  `agentlayer tasks`, `agentlayer install-desktop`.

**New capabilities**
* OS internals: memory maps, open files, limits, namespaces, cgroups, threads, scheduler stats, I/O, interrupts,
  vmstat, PSI pressure, buddy allocator, mounts, modules, locks, CPUs.
* Syscall tracing with strace inside the sandbox; inotify file watching; sockets and listening ports;
  hardware (battery, thermal, disks, network, USB, GPU, CPU); systemd services and journal logs.
* Vision: `look_at_screen` and `describe_image` (qwen3.8-27b vision).
* Scheduled tasks on systemd user timers (headless, never root, never self-scheduling).
* Voice: `/voice` (PipeWire + Whisper) and `/speak`.

**Safety**
* Undo journal and the freedesktop Trash for every file change.
* Just-in-time temporary capabilities (`request_capability`, `/grants`, `/revoke`).
* Prompt-injection shield (patterns, Llama Prompt Guard 2, gpt-oss-safeguard) with request tainting.
* Dry-run mode; human-only confirmations for root, scheduling and capability requests.

**Engineering**
* On-demand tool groups keep prompts at ~3K tokens; 2,048-token answers; Groq tool-call validation handled
  locally; live command output; shared runtime builder for every front end; 290 tests.

## 1.0.0

Review-1 prototype: capability broker, 26 tools, five phase policies, bubblewrap sandbox, hash-chained audit
log, Groq key pool with pacing and fallback, memory and sessions, desktop control, sudo with confirmation.
