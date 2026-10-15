# Using AgentLayer from other AI agents (MCP)

`agentlayer mcp` speaks the Model Context Protocol over stdio. Any MCP client gets AgentLayer's tools, and
every call is checked by the same capability broker, sandbox, injection shield, undo journal and audit log
as AgentLayer's own agent. The client brings the language model; AgentLayer brings the safety.

## Claude Code

```bash
claude mcp add agentlayer -- /home/sarthak/sarthak/repos/os-project/.venv/bin/agentlayer mcp
claude mcp list
```

## Claude Desktop / Cursor

```json
{
  "mcpServers": {
    "agentlayer": {
      "command": "/home/sarthak/sarthak/repos/os-project/.venv/bin/agentlayer",
      "args": ["-p", "phase5-desktop", "mcp"]
    }
  }
}
```

## Approvals

When a call needs the user's approval, a KDE dialog (`kdialog`) appears on the desktop with the tool, its
arguments, the broker's reason and a preview. Root actions also ask for the sudo password in a dialog; the
password goes straight to sudo. Without a graphical session (or with `AGENTLAYER_MCP_CONFIRM=deny` /
`--confirm deny`), such calls are declined and the client receives an error.

## What the client sees

* `tools/list` — all tools the policy grants, with JSON schemas; each description ends with what the tool
  does underneath (`[OS: openat walk …]`).
* `tools/call` — the tool's JSON result, or `isError: true` with the broker's reason ("Permission denied by
  the capability broker: …"). Denials are final.
* The session appears in `agentlayer audit` with `frontend: mcp` and the client's name.
