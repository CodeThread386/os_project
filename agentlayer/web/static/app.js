"use strict";
// AgentLayer dashboard. No third-party code: everything here is served by the local AgentLayer process.

const TOKEN = document.querySelector('meta[name="agentlayer-token"]').content;
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtBytes = (n) => { n = Number(n) || 0; const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0; while (n >= 1024 && i < 4) { n /= 1024; i++; } return `${n.toFixed(i ? 1 : 0)} ${u[i]}`; };
const fmtRate = (n) => `${fmtBytes(n)}/s`;
const fmtDur = (s) => { s = Math.floor(s); const d = Math.floor(s / 86400), h = Math.floor(s % 86400 / 3600), m = Math.floor(s % 3600 / 60); return d ? `${d}d ${h}h` : h ? `${h}h ${m}m` : `${m}m`; };
const time = (t) => new Date(t * 1000).toLocaleTimeString();

async function api(path, body, raw) {
  const opts = { headers: { "X-AgentLayer-Token": TOKEN } };
  if (body !== undefined || raw !== undefined) {
    opts.method = "POST";
    opts.headers["Content-Type"] = raw ? (raw.type || "application/octet-stream") : "application/json";
    opts.body = raw || JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}

function toast(text, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = text;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

// ---------------------------------------------------------------- markdown

function md(src) {
  const blocks = [];
  let text = esc(src).replace(/```[\w-]*\n([\s\S]*?)```/g, (_, code) => { blocks.push(`<pre><code>${code}</code></pre>`); return `\u0000${blocks.length - 1}\u0000`; });
  const inline = (s) => s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1<i>$2</i>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  let html = "", list = null, para = [], table = [];
  const flushPara = () => { if (para.length) { html += `<p>${para.map(inline).join("<br>")}</p>`; para = []; } };
  const flushList = () => { if (list) { html += `<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`; list = null; } };
  const flushTable = () => {
    if (!table.length) return;
    const rows = table.filter((r) => !/^\s*\|?\s*:?-{2,}/.test(r)).map((r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim())));
    html += "<table>" + rows.map((r, i) => `<tr>${r.map((c) => i ? `<td>${c}</td>` : `<th>${c}</th>`).join("")}</tr>`).join("") + "</table>";
    table = [];
  };
  for (const line of text.split("\n")) {
    const block = line.match(/^\u0000(\d+)\u0000$/);
    if (block) { flushPara(); flushList(); flushTable(); html += blocks[+block[1]]; continue; }
    if (/^\s*\|.*\|\s*$/.test(line)) { flushPara(); flushList(); table.push(line); continue; }
    flushTable();
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { flushPara(); flushList(); html += `<h3>${inline(h[2])}</h3>`; continue; }
    const ul = line.match(/^\s*[-*•]\s+(.*)$/), ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const tag = ul ? "ul" : "ol";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push((ul || ol)[1]);
      continue;
    }
    if (!line.trim()) { flushPara(); flushList(); continue; }
    flushList();
    para.push(line);
  }
  flushPara(); flushList(); flushTable();
  return html.replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[+i]);
}

const plain = (s) => String(s).replace(/```[\s\S]*?```/g, " ").replace(/[`*_#>|]/g, "").replace(/\s+/g, " ").trim();

// -------------------------------------------------------------------- chat

const state = { status: {}, busy: false, card: null, replaying: false, tokens: 0, calls: 0, speak: false, pending: null };

function scrollDown() { const m = $("#messages"); m.scrollTop = m.scrollHeight; }
function hideWelcome() { $("#welcome")?.remove(); }

function addMessage(role, html, meta = "") {
  hideWelcome();
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  wrap.innerHTML = `<div class="bubble">${html}${meta ? `<div class="meta">${esc(meta)}</div>` : ""}</div>`;
  $("#messages").appendChild(wrap);
  scrollDown();
}

function addNote(text, kind = "") {
  hideWelcome();
  const el = document.createElement("div");
  el.className = `note ${kind}`;
  el.textContent = text;
  $("#messages").appendChild(el);
  scrollDown();
}

function toolCard(ev) {
  hideWelcome();
  const el = document.createElement("div");
  el.className = "tool";
  const args = Object.entries(ev.args || {}).map(([k, v]) => `${k}=${typeof v === "string" ? JSON.stringify(v.length > 80 ? v.slice(0, 77) + "…" : v) : JSON.stringify(v)}`).join("  ");
  el.innerHTML = `<div class="head"><span class="name">${esc(ev.name)}</span><span class="args">${esc(args)}</span></div>`;
  $("#messages").appendChild(el);
  state.card = el;
  scrollDown();
}

function onDecision(ev) {
  const el = state.card;
  if (!el) return;
  el.classList.add(ev.verdict);
  el.querySelector(".head").insertAdjacentHTML("beforeend", `<span class="badge ${esc(ev.verdict)}">${esc(ev.verdict)}</span>`);
  el.insertAdjacentHTML("beforeend", `<div class="reason">${esc(ev.reason)}</div>`);
}

function onOutput(ev) {
  const el = state.card;
  if (!el) return;
  let live = el.querySelector(".live");
  if (!live) { live = document.createElement("div"); live.className = "live"; el.appendChild(live); }
  live.textContent = (live.textContent + ev.text).slice(-8000);
  live.scrollTop = live.scrollHeight;
  scrollDown();
}

function onResult(ev) {
  const el = state.card;
  if (!el) return;
  const dry = (ev.summary || "").startsWith("dry run");
  if (dry) el.querySelector(".head").insertAdjacentHTML("beforeend", '<span class="badge dry">simulated</span>');
  el.insertAdjacentHTML("beforeend", `<div class="summary ${ev.ok ? "" : "fail"}">↳ ${esc(ev.summary)}</div>`);
  if (ev.flagged) el.insertAdjacentHTML("beforeend", '<div class="note bad">⚠ flagged as a possible prompt injection</div>');
  if (ev.display && !el.querySelector(".live")) {
    const lines = ev.display.split("\n");
    const isDiff = /^(---|\+\+\+|@@)/m.test(ev.display);
    el.insertAdjacentHTML("beforeend", `<details ${lines.length <= 14 ? "open" : ""}><summary>${lines.length} line(s) of output</summary><pre class="${isDiff ? "diff" : ""}">${isDiff ? diffHtml(ev.display) : esc(ev.display)}</pre></details>`);
  }
  scrollDown();
}

function diffHtml(text) {
  return text.split("\n").map((l) => {
    const cls = l.startsWith("@@") ? "hunk" : l.startsWith("+") && !l.startsWith("+++") ? "add" : l.startsWith("-") && !l.startsWith("---") ? "del" : "";
    return cls ? `<span class="${cls}">${esc(l)}</span>` : esc(l);
  }).join("\n");
}

function setBusy(on) {
  state.busy = on;
  $("#send").disabled = on;
  $("#stop").classList.toggle("hidden", !on);
  if (!on) $("#thinking").classList.add("hidden");
}

function speak(text) {
  if (!state.speak || !("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(plain(text).slice(0, 700));
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

// ------------------------------------------------------------- dialogs

function showModal(html, cls = "") {
  const card = $("#modal-card");
  card.className = `modal-card ${cls}`;
  card.innerHTML = html;
  $("#modal").classList.remove("hidden");
}
function hideModal() { $("#modal").classList.add("hidden"); state.pending = null; }

function showConfirm(ev) {
  state.pending = ev;
  const preview = ev.preview ? (/^(---|\+\+\+|@@)/m.test(ev.preview) ? `<pre class="diff">${diffHtml(ev.preview)}</pre>` : `<pre>${esc(ev.preview)}</pre>`) : "";
  const title = ev.root ? "Run as ROOT?" : ev.human ? "Approve? (asked every time)" : "Approve this action?";
  showModal(`<h3>${title}</h3>
    <div class="mono">${esc(ev.tool)} ${esc(JSON.stringify(ev.args))}</div>
    <div class="reason">${esc(ev.reason)}</div>${preview}
    <div class="actions">
      <button class="btn" data-answer="no">Deny</button>
      ${ev.human ? "" : '<button class="btn" data-answer="always">Approve for this request</button>'}
      <button class="btn ${ev.root ? "danger" : "primary"}" data-answer="yes">Approve</button>
    </div>`, ev.root ? "root" : "");
  $$("#modal-card [data-answer]").forEach((b) => b.addEventListener("click", async () => {
    hideModal();
    await api("/api/answer", { request: ev.rid, value: b.dataset.answer }).catch((e) => toast(e.message, "bad"));
  }));
}

function showSecret(ev) {
  state.pending = ev;
  showModal(`<h3>Password needed</h3><div class="muted">${esc(ev.prompt)} — it goes straight to sudo; the AI and the logs never see it.</div>
    <form id="secret-form"><input type="password" id="secret" autocomplete="current-password" autofocus>
    <div class="actions"><button type="button" class="btn" id="secret-cancel">Cancel</button><button class="btn primary">Continue</button></div></form>`, "root");
  const send = async (value) => { hideModal(); await api("/api/answer", { request: ev.rid, value }).catch((e) => toast(e.message, "bad")); };
  $("#secret-form").addEventListener("submit", (e) => { e.preventDefault(); const v = $("#secret").value; $("#secret").value = ""; send(v); });
  $("#secret-cancel").addEventListener("click", () => send(""));
  setTimeout(() => $("#secret")?.focus(), 50);
}

// ------------------------------------------------------------- events

function renderStatus(s) {
  if (!s || !s.policy) return;
  state.status = s;
  const chips = [
    `<span class="chip"><b>policy</b> ${esc(s.policy)}</span>`,
    `<span class="chip"><b>model</b> ${esc((s.model || "").split("\n")[0])}</span>`,
    `<span class="chip"><b>sandbox</b> ${esc((s.sandbox || "").split(" — ")[0])}</span>`,
    ...(s.facts || []).filter(([k]) => k === "shield").map(() => '<span class="chip">🛡 injection shield</span>'),
    `<span class="chip">${esc(s.tools)} tools</span>`,
  ];
  if (s.grants && s.grants.length) chips.push(`<span class="chip">${s.grants.length} temporary grant(s)</span>`);
  if (s.dry_run) chips.unshift('<span class="chip dry">DRY RUN</span>');
  $("#chips").innerHTML = chips.join("");
  $("#dryrun").checked = !!s.dry_run;
}

function handle(ev) {
  switch (ev.ev) {
    case "hello":
      renderStatus(ev.status);
      state.replaying = true;
      (ev.history || []).forEach(handle);
      state.replaying = false;
      setBusy(!!ev.busy);
      if (ev.pending) (ev.pending.ev === "secret" ? showSecret : showConfirm)(ev.pending);
      break;
    case "ready": case "status": renderStatus(ev.status); break;
    case "user": addMessage("user", esc(ev.text).replace(/\n/g, "<br>")); break;
    case "thinking": $("#thinking").classList.toggle("hidden", !ev.on); break;
    case "busy": setBusy(ev.on); break;
    case "tool_call": toolCard(ev); break;
    case "decision": onDecision(ev); break;
    case "output": onOutput(ev); break;
    case "result": onResult(ev); break;
    case "confirm": if (!state.replaying) showConfirm(ev); break;
    case "secret": if (!state.replaying) showSecret(ev); break;
    case "notice": addNote(`· ${ev.text}`, "info"); break;
    case "warning": addNote(`⚠ ${ev.text}`, "bad"); if (!state.replaying) toast(ev.text, "bad"); break;
    case "answer":
      addMessage("agent", md(ev.text), `${ev.dry_run ? "dry run · " : ""}${ev.seconds ? ev.seconds + " s" : ""}`);
      state.card = null;
      if (!state.replaying) speak(ev.text);
      break;
    case "error": addNote(ev.text, "bad"); break;
    case "fatal": addNote(`The agent could not run: ${ev.text}`, "bad"); $("#conn").className = "conn off"; break;
    case "cleared": $("#messages").innerHTML = ""; break;
    case "resumed":
      $("#messages").innerHTML = "";
      (ev.history || []).forEach((m) => m.role === "user" ? addMessage("user", esc(m.content)) : addMessage("agent", md(m.content)));
      addNote(`resumed conversation ${ev.session} (${ev.turns} earlier request(s))`, "info");
      break;
    case "usage":
      state.tokens += (ev.prompt_tokens || 0) + (ev.completion_tokens || 0);
      state.calls += 1;
      $("#usage").textContent = `${state.calls} model call(s) · ${state.tokens.toLocaleString()} tokens · ${ev.model}`;
      (ev.notes || []).forEach((n) => addNote(`· ${n}`, "info"));
      break;
  }
}

function connect() {
  const es = new EventSource("/api/events");
  es.onopen = () => { $("#conn").className = "conn on"; $("#conn").title = "connected"; };
  es.onerror = () => { $("#conn").className = "conn off"; $("#conn").title = "disconnected — retrying"; };
  es.onmessage = (e) => handle(JSON.parse(e.data));
}

async function send(text) {
  text = text.trim();
  if (!text || state.busy) return;
  $("#input").value = "";
  autosize();
  try { await api("/api/ask", { text }); } catch (e) { toast(e.message, "bad"); }
}

function autosize() { const t = $("#input"); t.style.height = "auto"; t.style.height = Math.min(200, t.scrollHeight) + "px"; }

// --------------------------------------------------------------- voice

let recorder = null;
async function toggleMic() {
  if (recorder) { recorder.stop(); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); } catch (e) { toast("microphone unavailable: " + e.message, "bad"); return; }
  const type = MediaRecorder.isTypeSupported("audio/webm;codecs=opus") ? "audio/webm;codecs=opus" : "";
  recorder = new MediaRecorder(stream, type ? { mimeType: type } : {});
  const chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    $("#mic").classList.remove("recording");
    recorder = null;
    const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
    try {
      toast("transcribing…");
      const { text } = await api("/api/voice", undefined, blob);
      if (text && text.replace(/[\s.]/g, "").length > 1) { $("#input").value = text; send(text); } else toast("I did not catch that");
    } catch (e) { toast(e.message, "bad"); }
  };
  recorder.start();
  $("#mic").classList.add("recording");
  toast("recording… click the microphone again to stop");
}

// --------------------------------------------------------------- views

const views = {
  async monitor() { await pollMonitor(); },
  async matrix() {
    const m = await api("/api/matrix");
    $("#matrix-policy").textContent = `policy ${m.policy}`;
    const cls = { "✔": "c-ok", "✔*": "c-ok", "ask": "c-ask", "✖": "c-deny", "temp": "c-temp" };
    $("#matrix").innerHTML = `<tr><th>object \\ right</th>${m.columns.map((c) => `<th title="${esc(c.right)}">${esc(c.label)}</th>`).join("")}</tr>` +
      m.rows.map((r) => `<tr><td>${esc(r.object)}</td>${m.columns.map((c) => { const v = r.cells[c.right]; return `<td>${v ? `<span class="cell ${cls[v]}">${esc(v)}</span>` : '<span class="c-none">·</span>'}</td>`; }).join("")}</tr>`).join("");
    $("#invariants").innerHTML = m.invariants.map((i) => `<li>${esc(i)}</li>`).join("");
    const grants = m.temporary || [];
    $("#grants").innerHTML = grants.length ? `<table class="table"><tr><th>id</th><th>right</th><th>covers</th><th>lasts</th><th>reason</th><th></th></tr>${grants.map((g) =>
      `<tr><td class="mono">${esc(g.id)}</td><td>${esc(g.right)}</td><td class="mono">${esc(g.target)}</td><td>${esc(g.lasts)}</td><td>${esc(g.reason)}</td><td><button class="btn small" data-revoke="${esc(g.id)}">Revoke</button></td></tr>`).join("")}</table>`
      : '<div class="muted">None. When an action is denied only because no grant covers it, the agent may ask you for a temporary, expiring capability.</div>';
    $$("[data-revoke]").forEach((b) => b.addEventListener("click", async () => { await api("/api/revoke", { id: b.dataset.revoke }); setTimeout(views.matrix, 400); }));
  },
  async audit() {
    const rows = (await api("/api/audit?n=250")).reverse();
    $("#audit").innerHTML = "<tr><th>#</th><th>time</th><th>event</th><th>tool</th><th>verdict</th><th>detail</th></tr>" + rows.map((e) => {
      let verdict = e.verdict || "";
      if (verdict === "confirm") verdict = e.confirmed === false ? "deny" : "confirm";
      const detail = e.summary || e.reason || e.text || e.result || e.model || (e.groups ? `groups ${e.groups.join(", ")}` : "") || "";
      return `<tr><td>${esc(e.seq)}</td><td>${esc((e.ts || "").slice(11, 19))}</td><td>${esc(e.event)}</td><td class="mono">${esc(e.tool || "")}</td><td>${verdict ? `<span class="badge ${esc(verdict)}">${esc(e.confirmed === false ? "declined" : e.verdict)}</span>` : ""}</td><td>${esc(String(detail).slice(0, 180))}</td></tr>`;
    }).join("");
  },
  async undo() {
    const rows = await api("/api/undo");
    $("#undo").innerHTML = "<tr><th>id</th><th>when</th><th>change</th><th></th></tr>" + (rows.map((e) =>
      `<tr><td class="mono">${esc(e.id)}</td><td>${esc(e.when)}</td><td>${esc(e.summary)}${e.note ? `<div class="muted">${esc(e.note)}</div>` : ""}</td><td>${e.undone ? '<span class="muted">undone</span>' : `<button class="btn small" data-undo="${esc(e.id)}">Undo</button>`}</td></tr>`).join("") || '<tr><td colspan="4" class="muted">No changes recorded yet.</td></tr>');
    $$("[data-undo]").forEach((b) => b.addEventListener("click", async () => {
      try { const r = await api("/api/undo", { id: b.dataset.undo }); toast(r.message, "ok"); } catch (e) { toast(e.message, "bad"); }
      views.undo();
    }));
  },
  async memory() {
    const rows = await api("/api/memory");
    $("#memory").innerHTML = "<tr><th>id</th><th>saved</th><th>note</th><th></th></tr>" + (rows.map((f) =>
      `<tr><td class="mono">${esc(f.id)}</td><td>${esc(f.when)}</td><td>${esc(f.text)}</td><td><button class="btn small" data-forget="${esc(f.id)}">Forget</button></td></tr>`).join("") || '<tr><td colspan="4" class="muted">Nothing remembered yet.</td></tr>');
    $$("[data-forget]").forEach((b) => b.addEventListener("click", async () => { await api("/api/memory/forget", { id: b.dataset.forget }); views.memory(); }));
  },
  async sessions() {
    const rows = await api("/api/sessions");
    $("#sessions").innerHTML = "<tr><th>last used</th><th>requests</th><th>first request</th><th></th></tr>" + (rows.map((s) =>
      `<tr><td>${esc(new Date(s.updated * 1000).toLocaleString())}</td><td>${esc(s.turns)}</td><td>${esc(s.title)}</td><td><button class="btn small" data-resume="${esc(s.id)}">Continue</button></td></tr>`).join("") || '<tr><td colspan="4" class="muted">No saved conversations.</td></tr>');
    $$("[data-resume]").forEach((b) => b.addEventListener("click", async () => { await api("/api/resume", { session: b.dataset.resume }); show("chat"); }));
  },
  async tasks() {
    const rows = await api("/api/tasks");
    $("#tasks").innerHTML = "<tr><th>name</th><th>next run</th><th>runs</th><th>last result</th><th>request</th><th></th></tr>" + (rows.map((t) =>
      `<tr><td class="mono">${esc(t.name)}</td><td>${esc(t.next_run)}</td><td>${esc(t.runs)}</td><td>${esc(t.last_status || "—")}</td><td>${esc(t.request)}</td><td><button class="btn small" data-cancel="${esc(t.name)}">Cancel</button></td></tr>`).join("") || '<tr><td colspan="6" class="muted">No scheduled tasks. Try: “every day at 9:00, tell me how much disk space is left”.</td></tr>');
    $$("[data-cancel]").forEach((b) => b.addEventListener("click", async () => { await api("/api/tasks/cancel", { name: b.dataset.cancel }); views.tasks(); }));
  },
  async tools() {
    const rows = await api("/api/tools");
    $("#tools").innerHTML = "<tr><th>tool</th><th>group</th><th>rights</th><th>ok</th><th>what it does underneath</th></tr>" + rows.map((t) =>
      `<tr><td class="mono"><b>${esc(t.name)}</b><div class="muted">${esc(t.description)}</div></td><td>${esc(t.group)}</td><td class="mono">${esc(t.rights.join(", "))}</td><td>${t.available ? "✔" : '<span class="muted">·</span>'}</td><td class="mono">${esc(t.syscalls)}</td></tr>`).join("");
  },
  async quota() {
    const q = await api("/api/quota");
    $("#quota-groups").innerHTML = q.keys ? `<b>${q.keys}</b> key(s) in <b>${q.groups.length}</b> organisation(s): ` + q.groups.map((g, i) => `<span class="chip">org ${i + 1}: ${g.length} keys</span>`).join(" ") : '<span class="muted">No Groq keys configured.</span>';
    $("#quota").innerHTML = "<tr><th>org</th><th>model</th><th>TPM available</th><th>RPD available</th><th>used today</th><th>status</th></tr>" + (q.rows || []).map((r) => {
      return `<tr><td class="mono">${esc(r.group.slice(0, 6))}</td><td class="mono">${esc(r.model)}</td><td>${r.tpm_available == null ? "untried" : `${r.tpm_available.toLocaleString()} / ${r.tpm_limit.toLocaleString()}`}</td><td>${r.rpd_available == null ? "untried" : `${r.rpd_available} / ${r.rpd_limit}`}</td><td>${r.tokens_today.toLocaleString()} tok · ${r.requests_today} req</td><td>${r.blocked_for ? `<span class="badge deny">blocked ${Math.round(r.blocked_for)} s</span>` : '<span class="badge allow">ready</span>'}</td></tr>`;
    }).join("");
  },
};

// ------------------------------------------------------------- monitor

const hist = { net: [], disk: [] };
let monitorTimer = null, monitorBusy = false;

function spark(canvas, series, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const max = Math.max(1, ...series.flat());
  series.forEach((data, idx) => {
    ctx.beginPath();
    data.forEach((v, i) => { const x = (i / Math.max(1, 59)) * w, y = h - (v / max) * (h - 4) - 2; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.strokeStyle = color[idx]; ctx.lineWidth = 1.6; ctx.stroke();
  });
}

function metric(label, value, sub, extra = "") {
  return `<div class="card metric"><div class="label">${esc(label)}</div><div class="value">${value}</div><div class="sub">${sub}</div>${extra}</div>`;
}

async function pollMonitor() {
  clearTimeout(monitorTimer); monitorTimer = null;
  const active = () => $("#view-monitor").classList.contains("active");
  if (monitorBusy || !active()) return;  // one polling chain at a time
  monitorBusy = true;
  try {
    const s = await api("/api/system");
    const m = s.memory, pct = (a, b) => b ? Math.round(a / b * 100) : 0;
    hist.net.push([s.net.rx, s.net.tx]); hist.disk.push([s.disk.read, s.disk.write]);
    hist.net = hist.net.slice(-60); hist.disk = hist.disk.slice(-60);
    const bar = (p) => `<div class="bar"><i style="width:${Math.min(100, p)}%"></i></div>`;
    const cores = `<div class="cores">${s.cpu.cores.map((c) => `<i style="height:${Math.max(4, c)}%" title="${c}%"></i>`).join("")}</div>`;
    $("#monitor-cards").innerHTML = [
      metric("CPU", `${s.cpu.total}%`, `${s.cpu.cores.length} logical CPUs · load ${s.load.join(" / ")}`, cores),
      metric("Memory", `${pct(m.used, m.total)}%`, `${fmtBytes(m.used)} of ${fmtBytes(m.total)} · cache ${fmtBytes(m.cached)}`, bar(pct(m.used, m.total))),
      metric("Swap", `${pct(m.swap_used, m.swap_total)}%`, `${fmtBytes(m.swap_used)} of ${fmtBytes(m.swap_total)}`, bar(pct(m.swap_used, m.swap_total))),
      metric("Network", `↓ ${fmtRate(s.net.rx)}`, `↑ ${fmtRate(s.net.tx)}`, '<canvas class="spark" id="spark-net"></canvas>'),
      metric("Disk I/O", `R ${fmtRate(s.disk.read)}`, `W ${fmtRate(s.disk.write)}`, '<canvas class="spark" id="spark-disk"></canvas>'),
      metric("Processes", s.processes.total, `${s.processes.running} running · ${s.processes.blocked} blocked · ${s.processes.zombie} zombie · ${s.processes.stopped} stopped`),
      metric("Temperature", s.temperature == null ? "—" : `${s.temperature.toFixed(0)} °C`, "CPU package / hottest zone"),
      metric("Battery", s.battery ? `${s.battery.level}%` : "—", s.battery ? esc(s.battery.status) : "no battery", s.battery ? bar(s.battery.level) : ""),
      metric("Uptime", fmtDur(s.uptime), new Date(Date.now() - s.uptime * 1000).toLocaleString()),
    ].join("");
    spark($("#spark-net"), [hist.net.map((x) => x[0]), hist.net.map((x) => x[1])], ["#22d3ee", "#818cf8"]);
    spark($("#spark-disk"), [hist.disk.map((x) => x[0]), hist.disk.map((x) => x[1])], ["#34d399", "#fbbf24"]);
    $("#top-cpu").innerHTML = "<tr><th>pid</th><th>name</th><th>CPU</th><th>RSS</th></tr>" + s.top_cpu.map((p) => `<tr><td class="mono">${p.pid}</td><td>${esc(p.name)}</td><td>${p.cpu}%</td><td>${fmtBytes(p.rss)}</td></tr>`).join("");
    $("#top-mem").innerHTML = "<tr><th>pid</th><th>name</th><th>RSS</th><th>state</th></tr>" + s.top_memory.map((p) => `<tr><td class="mono">${p.pid}</td><td>${esc(p.name)}</td><td>${fmtBytes(p.rss)}</td><td>${esc(p.state)}</td></tr>`).join("");
  } catch (e) { console.warn("monitor:", e); }  // keep polling
  finally { monitorBusy = false; }
  if (active()) monitorTimer = setTimeout(pollMonitor, 2000);
}

function show(name) {
  $$("#nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  if (views[name]) views[name]().catch((e) => toast(e.message, "bad"));
  if (name === "chat") $("#input").focus();
}

// ----------------------------------------------------------------- init

const SUGGESTIONS = [
  "How is my laptop doing? Battery, temperature, memory and the busiest processes.",
  "What is taking up space in my Downloads folder?",
  "Show the memory map of the process using the most RAM",
  "Which system calls does `ls -la` make, and what do they do?",
  "Write a C program that forks a child, compile it and run it",
  "What's on my screen right now?",
  "Which programs are listening on network ports?",
  "Every day at 9:00, tell me how much disk space is left",
];

document.addEventListener("DOMContentLoaded", () => {
  $("#suggestions").innerHTML = SUGGESTIONS.map((s) => `<button class="suggestion">${esc(s)}</button>`).join("");
  $$(".suggestion").forEach((b) => b.addEventListener("click", () => send(b.textContent)));
  $$("#nav button").forEach((b) => b.addEventListener("click", () => { history.replaceState(null, "", `#/${b.dataset.view}`); show(b.dataset.view); }));
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); send($("#input").value); });
  $("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send($("#input").value); } });
  $("#input").addEventListener("input", autosize);
  $("#stop").addEventListener("click", () => api("/api/stop", {}).catch(() => {}));
  $("#mic").addEventListener("click", toggleMic);
  $("#dryrun").addEventListener("change", (e) => api("/api/dryrun", { on: e.target.checked }));
  $("#speak").addEventListener("change", (e) => { state.speak = e.target.checked; if (!state.speak && "speechSynthesis" in window) speechSynthesis.cancel(); });
  $("#verify").addEventListener("click", async () => { const r = await api("/api/audit/verify"); $("#verify-result").textContent = (r.ok ? "✔ " : "✖ ") + r.message; });
  $("#memory-form").addEventListener("submit", async (e) => { e.preventDefault(); const t = $("#memory-text").value.trim(); if (t) { await api("/api/memory/add", { text: t }); $("#memory-text").value = ""; views.memory(); } });
  $("#new-chat").addEventListener("click", async () => { await api("/api/reset", {}); show("chat"); });
  connect();
  const start = location.hash.replace(/^#\/?/, "");  // #/monitor deep links
  if (start && document.getElementById(`view-${start}`)) show(start); else $("#input").focus();
});
