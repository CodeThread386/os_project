# Review demo script (about 12 minutes)

Run `agentlayer doctor` beforehand: every line should be green. Keep a terminal and the dashboard open.

## 1. The idea (1 min)

"An AI plans OS actions; a capability broker decides; tools perform approved actions through system calls;
everything is audited." Show the diagram at the top of the README.

## 2. Least privilege, phase by phase (2 min)

```bash
agentlayer demo
agentlayer policy -p phase1     # the access matrix of the read-only phase
```

Point out: the same 30 actions judged by five policies; deny rules beat grants; invariants (PID 1, audit log).

## 3. The agent at work (3 min) — dashboard (`agentlayer ui`)

* "How is my laptop doing? Battery, temperature, memory and the busiest processes."
* "Which system calls does `ls -la` make, and what do they do?" → a real strace table from the sandbox.
* "Write a C program that forks a child, compile it and run it." → PIDs 2 and 3: the sandbox's PID namespace.
* Open **Monitor** (live /proc sampling) and **Access matrix**.

## 4. Safety layers (4 min)

* "Show me my SSH private key." → denied by a deny rule (Audit log tab shows it).
* Put an injected instruction in a file (see `docs/THREAT_MODEL.md`), ask "Summarise notes.txt" → the shield
  flags it; any change afterwards asks you.
* "Delete demo/old.txt" → confirmation with a preview; approve; then `agentlayer undo` (it comes back from the
  Trash; show it in Dolphin's Trash too).
* "Create report.md in ~/some-folder" → denied, the agent asks for a **temporary capability**; approve;
  `/grants` shows the countdown; `/revoke`.
* `/dryrun on` → "clean up my workspace" → every change is simulated.
* "Install cowsay" → red ROOT prompt and a hidden sudo password prompt.

## 5. Automation, voice, other agents (1 min)

* "In 2 minutes, notify me that the scheduler works" → a systemd user timer; the notification arrives.
* Microphone button in the dashboard (Whisper), **Speak** toggle.
* `claude mcp add agentlayer -- $PWD/.venv/bin/agentlayer mcp` → Claude Code uses AgentLayer's tools through
  the same broker; approvals pop up as KDE dialogs.

## 6. Evidence (1 min)

```bash
agentlayer eval          # 8/8 tasks, 7/7 attacks contained
agentlayer audit --verify
make test                # 290 tests
```

## Likely questions

* **Why not just give the model a shell?** No shell means no `;`, `&&`, `$()`, redirection to abuse; every
  program is allow-listed; arguments are checked; commands are sandboxed.
* **What if the model is tricked?** The broker does not trust the model: denials are mechanical. The shield
  narrows what a trick can attempt; the sandbox and confirmations bound what it can achieve; undo and the
  audit log recover and explain.
* **Where is the OS content?** fork/execve/wait4, pipes, rlimits, signals, namespaces, procfs/sysfs, inotify,
  ptrace, openat/O_NOFOLLOW against TOCTOU, the access matrix and capability revocation, journaling, systemd
  timers, D-Bus IPC, privilege separation.
* **What does it cost?** About 3K tokens per model call; 25 keys in 5 organisations give 15 budgets; the
  quota pool paces requests and falls back to a local model.
