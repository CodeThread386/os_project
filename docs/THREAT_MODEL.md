# Threat model

## Assets

* the user's files, credentials and secrets (`~/.ssh`, `.env` files, keyrings, browser profiles, API keys);
* the integrity of the system (system files, other users' processes, running services);
* the user's privacy (screen contents, clipboard, window titles);
* AgentLayer's own trusted computing base: policy files, audit log, memory, sessions, undo journal.

## Adversaries and trust

| Who | Can | Trusted? |
|---|---|---|
| The user | approve or decline actions, edit the policy | yes |
| The language model | propose any tool call with any arguments | **no**: treated as possibly confused or hijacked |
| Content the agent reads (web pages, files, command output, window titles, screen text) | carry instructions (indirect prompt injection) | **no** |
| Another AI agent connected over MCP | propose tool calls | **no**: same broker, same approvals |
| Other local users / processes | connect to 127.0.0.1, race the file system | **no** |
| The kernel, bubblewrap, systemd, sudo | enforce their own rules | yes (out of scope) |

## Trust boundaries

1. Model → agent: everything the model emits is validated (`prepare_args`) and authorised (broker).
2. Agent → OS: only tool implementations make system calls, only on broker-approved objects.
3. Commands → system: a bubblewrap sandbox with minimal environment and rlimits.
4. Outside content → model: the injection shield and taint.
5. Browser → dashboard: token, Host check, CSRF header, CSP; the web process cannot run tools.
6. Agent → root: sudo with a person's live approval and password.

## STRIDE

| Threat | Example | Mitigations |
|---|---|---|
| **S**poofing | a web page pretends to be the user ("the user approved this") | approvals only come from the UI; the shield flags false-authority text |
| | a local process calls the dashboard API | random token, SameSite cookie, CSRF header, loopback-only Host check |
| **T**ampering | the agent edits its own policy or deletes the audit log | trusted-computing-base invariant; audit hash chain detects edits |
| | a symlink swapped in between check and use | canonical paths, `openat` walks with `O_NOFOLLOW` |
| **R**epudiation | "the agent did that on its own" | every decision and outcome in the audit log, with approvals recorded |
| **I**nformation disclosure | read `~/.ssh/id_rsa`, `/proc/self/environ`, `.env` files | deny rules, never-readable `/proc` files, sandbox masks, minimal child environment |
| | exfiltrate data over the network | GET-only fetches, private-address refusal, allow-listed hosts, network off in the sandbox unless approved |
| | screenshots/clipboard sent to the model | confirmations; `--provider ollama` keeps data local |
| **D**enial of service | fork bomb, infinite loop, giant output | rlimits, deadlines, killing the process group, PID namespace, output caps |
| **E**levation of privilege | the model runs `sudo` or tricks the user | no sudo in commands; root only through `run_as_root`/`packages` with a live approval and the password piped to sudo; `systemctl reboot` and similar denied |
| | the model widens its own rights | temporary grants need a live approval, expire, never cover root or deny rules |

## Prompt injection specifically

1. Patterns catch well-known phrasings instantly.
2. Llama Prompt Guard 2 catches explicit jailbreak techniques (0.9996 on "ignore your previous
   instructions…").
3. gpt-oss-safeguard checks content that addresses an AI or moves secrets against a written policy, and catches
   politely worded social engineering that Prompt Guard misses (0.004 on our sample, flagged by safeguard).
4. A flag *taints* the request: every further state-changing action needs the user, even if the policy
   would allow it.
5. Regardless of 1–4, the broker and sandbox limit what any instruction can achieve (`agentlayer eval`
   attacks: 7/7 contained).

## Residual risks

* An injection that evades all detectors can still make the model misuse *allowed* actions (for example
  writing a misleading file in the workspace); undo and the audit log make that visible and reversible.
* Inside the sandbox, a program may change files in the workspace and granted folders without per-file
  confirmation.
* DNS rebinding for `fetch_url` and directory-tree copies are not fully race-free.
* A user who approves everything without reading defeats the confirmation layer; human-only prompts (root,
  temporary grants, scheduling) cannot be auto-approved for that reason.
