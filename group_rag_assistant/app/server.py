from __future__ import annotations

import asyncio
import json
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, Optional, TypedDict, Iterable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import SETTINGS
from app.rag.query import ask

app = FastAPI(title="Group RAG Assistant", version="0.4")

# Put your images here:
#   app/static/logo_day.png
#   app/static/logo_night.png
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -----------------------------
# Simple in-memory chat memory
# -----------------------------
DEFAULT_CONTEXT_TURNS = 10   # remember last 10 turns per session
_MAX_TURNS_PER_SESSION = 80  # hard cap


class Turn(TypedDict):
    q: str
    a: str


_MEM_LOCK = Lock()
_SESSION_TURNS: Dict[str, Deque[Turn]] = {}


def index_ready() -> bool:
    return SETTINGS.vectorstore_dir.exists() and any(SETTINGS.vectorstore_dir.iterdir())


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _get_history(session_id: str) -> Deque[Turn]:
    with _MEM_LOCK:
        if session_id not in _SESSION_TURNS:
            _SESSION_TURNS[session_id] = deque(maxlen=_MAX_TURNS_PER_SESSION)
        return _SESSION_TURNS[session_id]


def _build_context(session_id: str, depth: int) -> str:
    depth = _clamp_int(depth, 0, 10)
    if depth <= 0:
        return ""

    hist = _get_history(session_id)
    if not hist:
        return ""

    turns = list(hist)[-depth:]
    lines = []
    for t in turns:
        lines.append(f"User: {t['q']}\nAssistant: {t['a']}")
    return "\n\n".join(lines)


def _store_turn(session_id: str, q: str, a: str) -> None:
    hist = _get_history(session_id)
    with _MEM_LOCK:
        hist.append({"q": q, "a": a})


def _chunk_text(s: str, size: int = 120) -> Iterable[str]:
    if not s:
        return []
    return (s[i: i + size] for i in range(0, len(s), size))


class AskRequest(BaseModel):
    agent: str  # thesis/python/reviewer (intern)
    question: str
    k: int = 4
    session_id: Optional[str] = None


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Local Research Assistant</title>
  <style>
    :root{
      --bg: #ffffff;
      --panel: #ffffff;
      --text: #111111;
      --muted: #6b7280;
      --border: #e5e7eb;
      --shadow: 0 10px 30px rgba(0,0,0,0.06);
      --btn: #111111;
      --btnText: #ffffff;
      --pillBg: #f3f4f6;
      --pillText: #111111;
      --pillActiveBg: #111111;
      --pillActiveText: #ffffff;
      --inputBg: #ffffff;

      --contentMax: 920px;
      --chatMax: 880px;
    }
    [data-theme="dark"]{
      --bg: #0b0f19;
      --panel: #0f1629;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --border: rgba(255,255,255,0.10);
      --shadow: 0 10px 30px rgba(0,0,0,0.35);
      --btn: #e5e7eb;
      --btnText: #0b0f19;
      --pillBg: rgba(255,255,255,0.08);
      --pillText: #e5e7eb;
      --pillActiveBg: #e5e7eb;
      --pillActiveText: #0b0f19;
      --inputBg: rgba(255,255,255,0.06);
    }
    *{ box-sizing: border-box; }
    body{
      margin:0;
      font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .wrap{
      min-height: 100vh;
      display:flex;
      align-items:flex-start;
      justify-content:center;
      padding: 28px 18px;
    }
    .shell{
      width: 100%;
      max-width: var(--contentMax);
      display:flex;
      flex-direction:column;
      gap: 14px;
    }
    .topbar{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap: 12px;
    }
    .brand{
      display:flex;
      align-items:center;
      gap: 10px;
    }
    .brand img{
      width: 34px;
      height: 34px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--panel);
      object-fit: cover;
    }
    .title{
      font-size: 14px;
      font-weight: 650;
      letter-spacing: 0.2px;
    }
    .subtitle{
      font-size: 12px;
      color: var(--muted);
      margin-top: 2px;
      line-height: 1.35;
    }
    .toggles{
      display:flex;
      align-items:center;
      gap: 10px;
    }
    .toggleBtn{
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      border-radius: 12px;
      padding: 8px 10px;
      cursor:pointer;
      box-shadow: var(--shadow);
      font-weight: 600;
    }
    .panel{
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 16px;
    }
    .centerHero{
      display:flex;
      flex-direction:column;
      align-items:center;
      text-align:center;
      gap: 10px;
      padding: 6px 0 2px 0;
    }
    .heroLogo{
      width: 64px;
      height: 64px;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: var(--panel);
      object-fit: cover;
      box-shadow: var(--shadow);
    }
    .heroH{
      font-size: 20px;
      font-weight: 800;
      margin: 0;
      letter-spacing: 0.2px;
    }
    .heroP{
      font-size: 13px;
      color: var(--muted);
      margin: 0;
      max-width: 760px;
      line-height: 1.5;
    }
    .metaRow{
      display:flex;
      align-items:center;
      justify-content:center;
      gap: 10px;
      margin-top: 6px;
      flex-wrap: wrap;
    }
    .pill{
      display:flex;
      gap: 6px;
      padding: 6px;
      border-radius: 999px;
      background: var(--pillBg);
      border: 1px solid var(--border);
    }
    .pill button{
      border: 0;
      background: transparent;
      color: var(--pillText);
      padding: 8px 12px;
      border-radius: 999px;
      cursor:pointer;
      font-size: 13px;
      font-weight: 650;
    }
    .pill button.active{
      background: var(--pillActiveBg);
      color: var(--pillActiveText);
    }

    /* Chatbereich: scrollbar, damit Eingabe immer sichtbar bleibt */
    .chatWrap{
      margin: 14px auto 10px auto;
      width: 100%;
      max-width: var(--chatMax);
      display:flex;
      flex-direction:column;
      gap: 10px;

      max-height: 56vh;
      overflow-y: auto;
      padding: 6px 8px 6px 0;
      scroll-behavior: smooth;
    }

    .msg{ display:flex; width:100%; }
    .bubble{
      max-width: 100%;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px 12px;
      box-shadow: var(--shadow);
      line-height: 1.55;
      font-size: 14px;
      overflow: hidden;
    }
    .msg.user{ justify-content:flex-end; }
    .msg.user .bubble{ background: var(--pillBg); }
    .msg.assistant{ justify-content:flex-start; }
    .msg.assistant .bubble{ background: var(--inputBg); }

    /* Markdown typography */
    .md p{ margin: 0 0 10px 0; }
    .md p:last-child{ margin-bottom:0; }
    .md h1,.md h2,.md h3{ margin: 8px 0 8px 0; line-height: 1.2; }
    .md h1{ font-size: 18px; }
    .md h2{ font-size: 16px; }
    .md h3{ font-size: 15px; }
    .md ul{ margin: 0 0 10px 18px; padding:0; }
    .md li{ margin: 4px 0; }

    .md code.inline{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 0.95em;
      padding: 2px 6px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: rgba(127,127,127,0.10);
    }

    /* Code blocks */
    pre.code{
      margin: 10px 0 0 0;
      padding: 12px 12px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(127,127,127,0.10);
      overflow-x: auto;
    }
    pre.code code{
      white-space: pre;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 13px;
      line-height: 1.5;
      display:block;
    }

    details.sourcesBox{
      margin-top: 10px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: var(--panel);
      padding: 8px 10px;
    }
    details.sourcesBox summary{
      cursor:pointer;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      user-select: none;
    }
    .sourcesList{
      margin: 8px 0 0 0;
      padding: 0 0 0 16px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }
    .sourcesList li{ margin: 6px 0; }
    .sourcesList a{
      color: inherit;
      text-decoration: underline;
      text-underline-offset: 2px;
      word-break: break-word;
    }

    textarea{
      width: 100%;
      min-height: 120px;
      /* Auto-Resize: keine manuelle Resize-Handle, JS macht die Höhe */
      resize: none;

      border-radius: 16px;
      border: 1px solid var(--border);
      background: var(--inputBg);
      color: var(--text);
      padding: 14px 14px;
      font-size: 14px;
      outline: none;
      line-height: 1.5;

      max-width: var(--chatMax);
      display:block;
      margin: 0 auto;

      /* Auto-Resize Limits */
      overflow-y: hidden; /* JS schaltet ggf. auf auto */
      max-height: 35vh;
    }

    .actions{
      display:flex;
      gap: 10px;
      align-items:center;
      justify-content:space-between;
      flex-wrap: wrap;
      margin-top: 10px;
      max-width: var(--chatMax);
      margin-left:auto;
      margin-right:auto;
    }
    .btnRow{
      display:flex;
      gap: 10px;
      align-items:center;
      flex-wrap: wrap;
    }
    .sendBtn{
      border: 0;
      background: var(--btn);
      color: var(--btnText);
      padding: 10px 14px;
      border-radius: 14px;
      cursor:pointer;
      font-weight: 750;
    }
    .copyBtn{
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 14px;
      cursor:pointer;
      font-weight: 700;
      box-shadow: var(--shadow);
    }
    .copyBtn:disabled{
      opacity: 0.5;
      cursor: not-allowed;
      box-shadow: none;
    }
    .status{
      font-size: 12px;
      color: var(--muted);
      max-width: var(--chatMax);
    }
    .foot{
      text-align:center;
      color: var(--muted);
      font-size: 12px;
      padding-top: 2px;
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="shell">

    <div class="topbar">
      <div class="brand">
        <img id="logoTop" src="/static/logo_day.png" onerror="this.style.display='none';" alt="logo"/>
        <div>
          <div class="title">Local Research Assistant</div>
          <div class="subtitle">Ollama + RAG · KB managed via PyCharm (no UI upload/reindex)</div>
        </div>
      </div>
      <div class="toggles">
        <button class="toggleBtn" id="themeBtn" onclick="toggleTheme()">Dark mode</button>
      </div>
    </div>

    <div class="panel">
      <div class="centerHero">
        <img id="logoHero" class="heroLogo" src="/static/logo_day.png" onerror="this.style.display='none';" alt="logo"/>
        <div class="heroH">Ask your local assistant</div>
        <p class="heroP">
          Writer / Python / Reviewer. Chat UI with Markdown + code blocks + streaming effect.
          Update <b>kb/</b> and run <code class="inline">python main.py ingest</code> when needed.
        </p>

        <div class="metaRow">
          <div class="pill" role="tablist" aria-label="agent selector">
            <button id="btn-thesis" class="active" onclick="setAgent('thesis')">Writer</button>
            <button id="btn-python" onclick="setAgent('python')">Python</button>
            <button id="btn-reviewer" onclick="setAgent('reviewer')">Reviewer</button>
          </div>
          <button class="copyBtn" onclick="newChat()" title="Start a new session (clears memory for this browser).">New chat</button>
          <button class="copyBtn" onclick="clearChatUI()" title="Clears the visible chat (does not clear server memory).">Clear UI</button>
        </div>
      </div>

      <div class="chatWrap" id="chat"></div>

      <textarea id="q" placeholder="Writer mode: e.g., Write a Methods paragraph / Rewrite in academic English / Draft an abstract..."></textarea>

      <div class="actions">
        <div class="btnRow">
          <button class="sendBtn" onclick="send()">Send</button>
          <button class="copyBtn" id="copyLastBtn" onclick="copyLastAssistant()" disabled>Copy last</button>
        </div>
        <div class="status" id="status"></div>
      </div>
    </div>

    <div class="foot">
      Local-only. No uploads in UI. Update kb/ + run <code class="inline">python main.py ingest</code> when needed.
    </div>

  </div>
</div>

<script>
let agent = "thesis";
let lastAssistantRaw = "";

// placeholders
const placeholders = {
  thesis: "Writer mode: e.g., Write a Methods paragraph / Rewrite in academic English / Draft an abstract...",
  python: "Python mode: e.g., Write code to load CSV, clean data, plot results... (code-only)",
  reviewer: "Reviewer mode: e.g., Review this paragraph for clarity, claims, missing citations, logic..."
};

// session id (per browser) – enables server-side memory
let sessionId = localStorage.getItem("session_id");
if(!sessionId){
  sessionId = (crypto && crypto.randomUUID)
    ? crypto.randomUUID()
    : (Date.now().toString(36) + Math.random().toString(36).slice(2));
  localStorage.setItem("session_id", sessionId);
}

/* ---------- Scroll: nur wenn User unten ist ---------- */
function isChatNearBottom(thresholdPx = 140){
  const chat = document.getElementById("chat");
  if(!chat) return true;
  const distance = chat.scrollHeight - (chat.scrollTop + chat.clientHeight);
  return distance <= thresholdPx;
}
function maybeScrollToBottom(force = false){
  const chat = document.getElementById("chat");
  if(!chat) return;
  if(force || isChatNearBottom()){
    chat.scrollTop = chat.scrollHeight;
  }
}

/* ---------- Textarea Auto-Resize ---------- */
function autoResizeTextarea(el){
  if(!el) return;
  el.style.height = "auto";

  const maxPx = Math.round(window.innerHeight * 0.35); // match CSS max-height: 35vh
  const next = Math.min(el.scrollHeight, maxPx);

  el.style.height = next + "px";
  el.style.overflowY = (el.scrollHeight > maxPx) ? "auto" : "hidden";
}
(function bindTextareaAutosize(){
  const ta = document.getElementById("q");
  if(!ta) return;
  const handler = () => autoResizeTextarea(ta);
  ta.addEventListener("input", handler);
  window.addEventListener("resize", handler);
  handler(); // initial
})();

function setAgent(a){
  agent = a;
  document.getElementById("btn-thesis").classList.toggle("active", a==="thesis");
  document.getElementById("btn-python").classList.toggle("active", a==="python");
  document.getElementById("btn-reviewer").classList.toggle("active", a==="reviewer");
  const ta = document.getElementById("q");
  if (ta) ta.placeholder = placeholders[a] || "Type your request here...";
}

function setLogoForTheme(t){
  const src = (t === "dark") ? "/static/logo_night.png" : "/static/logo_day.png";
  const top = document.getElementById("logoTop");
  const hero = document.getElementById("logoHero");
  if (top) top.src = src;
  if (hero) hero.src = src;
}

function applyTheme(t){
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem("theme", t);
  const btn = document.getElementById("themeBtn");
  if (btn) btn.textContent = (t === "dark") ? "Light mode" : "Dark mode";
  setLogoForTheme(t);
}

function toggleTheme(){
  const cur = localStorage.getItem("theme");
  applyTheme(cur === "dark" ? "light" : "dark");
}

(function initTheme(){
  const saved = localStorage.getItem("theme");
  if(saved){
    applyTheme(saved);
  }else{
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(prefersDark ? "dark" : "light");
  }
  setAgent(agent);
})();

function newChat(){
  sessionId = (crypto && crypto.randomUUID)
    ? crypto.randomUUID()
    : (Date.now().toString(36) + Math.random().toString(36).slice(2));
  localStorage.setItem("session_id", sessionId);
  clearChatUI();
  const ta = document.getElementById("q");
  if(ta){
    ta.value = "";
    autoResizeTextarea(ta);
    ta.focus();
  }
  setStatus("New chat started (new session_id).");
}

function clearChatUI(){
  const chat = document.getElementById("chat");
  if(chat) chat.innerHTML = "";
  lastAssistantRaw = "";
  document.getElementById("copyLastBtn").disabled = true;
  setStatus("Chat UI cleared.");
}

function setStatus(s){
  const status = document.getElementById("status");
  if(status) status.textContent = s || "";
}

function escapeHtml(s){
  return (s || "")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

/* Minimal safe Markdown renderer */
function mdToHtml(md){
  const text = md || "";
  const parts = [];
  let i = 0;

  while(i < text.length){
    const start = text.indexOf("```", i);
    if(start === -1){
      parts.push({type:"text", value: text.slice(i)});
      break;
    }
    if(start > i) parts.push({type:"text", value: text.slice(i, start)});

    const langEnd = text.indexOf("\n", start + 3);
    if(langEnd === -1){
      parts.push({type:"text", value: text.slice(start)});
      break;
    }
    const lang = text.slice(start + 3, langEnd).trim();
    const end = text.indexOf("```", langEnd + 1);
    if(end === -1){
      parts.push({type:"text", value: text.slice(start)});
      break;
    }
    const code = text.slice(langEnd + 1, end);
    parts.push({type:"code", lang: lang || "", value: code});
    i = end + 3;
  }

  const htmlParts = [];

  function inlineFormat(s){
    let x = escapeHtml(s);
    x = x.replace(/`([^`]+)`/g, (m, g1) => `<code class="inline">${g1}</code>`);
    x = x.replace(/\*\*([^*]+)\*\*/g, (m, g1) => `<strong>${g1}</strong>`);
    return x;
  }

  function renderTextBlock(block){
    const lines = block.split(/\r?\n/);
    let html = "";
    let inList = false;

    function closeList(){
      if(inList){ html += "</ul>"; inList = false; }
    }

    for(const rawLine of lines){
      const line = rawLine.replace(/\s+$/,"");
      if(!line.trim()){
        closeList();
        continue;
      }

      if(line.startsWith("### ")){ closeList(); html += `<h3>${inlineFormat(line.slice(4))}</h3>`; continue; }
      if(line.startsWith("## ")){  closeList(); html += `<h2>${inlineFormat(line.slice(3))}</h2>`; continue; }
      if(line.startsWith("# ")){   closeList(); html += `<h1>${inlineFormat(line.slice(2))}</h1>`; continue; }

      if(line.startsWith("- ") || line.startsWith("* ")){
        if(!inList){ html += "<ul>"; inList = true; }
        html += `<li>${inlineFormat(line.slice(2))}</li>`;
        continue;
      }

      closeList();
      html += `<p>${inlineFormat(line)}</p>`;
    }

    closeList();
    return html;
  }

  for(const p of parts){
    if(p.type === "code"){
      const langClass = p.lang ? `language-${escapeHtml(p.lang)}` : "";
      htmlParts.push(
        `<pre class="code"><code class="${langClass}">${escapeHtml(p.value)}</code></pre>`
      );
    }else{
      htmlParts.push(renderTextBlock(p.value));
    }
  }

  return `<div class="md">${htmlParts.join("")}</div>`;
}

function renderAssistant(agentName, rawText){
  if(agentName === "python"){
    return `<pre class="code"><code>${escapeHtml(rawText || "")}</code></pre>`;
  }
  return mdToHtml(rawText || "");
}

function safeRenderAssistant(agentName, rawText){
  try{
    return renderAssistant(agentName, rawText);
  }catch(e){
    console.error("Render failed:", e);
    return `<pre class="code"><code>${escapeHtml(rawText || "")}</code></pre>`;
  }
}

/* Sources as friendly list with links */
function sourcesToHtml(sourcesRaw){
  const s = (sourcesRaw || "").trim();
  if(!s) return "";

  try{
    const j = JSON.parse(s);
    if(Array.isArray(j)){
      const lis = j.map(item => {
        const t = (typeof item === "string") ? item : JSON.stringify(item);
        return sourceItemToLi(t);
      }).join("");
      return `<ul class="sourcesList">${lis}</ul>`;
    }
  }catch(e){}

  const lines = s.split(/\r?\n/).map(x => x.trim()).filter(Boolean);
  const lis = lines.map(sourceItemToLi).join("");
  return `<ul class="sourcesList">${lis}</ul>`;
}

function sourceItemToLi(line){
  const urlMatch = line.match(/https?:\/\/[^\s)]+/i);
  if(urlMatch){
    const url = urlMatch[0];
    return `<li><a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(line)}</a></li>`;
  }
  return `<li>${escapeHtml(line)}</li>`;
}

function appendMessage(role, html, forceScroll=false){
  const chat = document.getElementById("chat");
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = html;
  wrap.appendChild(bubble);
  chat.appendChild(wrap);

  maybeScrollToBottom(forceScroll);
  return bubble;
}

/* Copy helper */
async function copyToClipboard(text){
  try{
    await navigator.clipboard.writeText(text);
    return true;
  }catch(e){
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try{
      document.execCommand("copy");
      document.body.removeChild(ta);
      return true;
    }catch(err){
      document.body.removeChild(ta);
      return false;
    }
  }
}

async function copyLastAssistant(){
  if(!lastAssistantRaw) return;
  const ok = await copyToClipboard(lastAssistantRaw);
  setStatus(ok ? "Copied last assistant message." : "Copy failed (browser permission).");
}

/* Streaming via NDJSON fetch */
async function send(){
  const ta = document.getElementById("q");
  const q = (ta ? ta.value.trim() : "");
  if(!q){ setStatus("Please enter a question."); return; }

  // Beim Senden: einmal hart nach unten scrollen, damit die neue Antwort sichtbar ist
  appendMessage("user", `<div class="md"><p>${escapeHtml(q)}</p></div>`, true);
  const assistantBubble = appendMessage("assistant", `<div class="md"><p><em>…</em></p></div>`, true);

  setStatus("Thinking...");
  document.getElementById("copyLastBtn").disabled = true;

  // Eingabe sofort leeren + Auto-Resize aktualisieren
  if(ta){
    ta.value = "";
    autoResizeTextarea(ta);
    ta.focus();
  }

  try{
    const res = await fetch("/api/ask_stream", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        agent: agent,
        question: q,
        k: 4,
        session_id: sessionId
      })
    });

    if(!res.ok){
      const t = await res.text();
      assistantBubble.innerHTML = `<div class="md"><p><strong>Error:</strong> ${escapeHtml(t)}</p></div>`;
      setStatus("Error.");
      maybeScrollToBottom(false);
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");

    let buffer = "";
    let rawAnswer = "";
    let sourcesRaw = "";
    let lastRenderAt = 0;

    while(true){
      const {value, done} = await reader.read();
      if(done) break;
      buffer += decoder.decode(value, {stream:true});

      let idx;
      while((idx = buffer.indexOf("\n")) >= 0){
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 1);
        if(!line) continue;

        let msg;
        try{ msg = JSON.parse(line); }catch(e){ continue; }

        if(msg.type === "delta"){
          rawAnswer += (msg.text || "");
          const now = Date.now();
          if(now - lastRenderAt > 45){
            assistantBubble.innerHTML = safeRenderAssistant(agent, rawAnswer);
            // nur scrollen, wenn User ohnehin unten ist
            maybeScrollToBottom(false);
            lastRenderAt = now;
          }
        }else if(msg.type === "done"){
          sourcesRaw = msg.sources || "";
          assistantBubble.innerHTML = safeRenderAssistant(agent, rawAnswer);

          const srcHtml = sourcesToHtml(sourcesRaw);
          if(srcHtml){
            const details = document.createElement("details");
            details.className = "sourcesBox";
            const summary = document.createElement("summary");
            summary.textContent = "Sources";
            details.appendChild(summary);

            const box = document.createElement("div");
            box.innerHTML = srcHtml;
            details.appendChild(box);

            assistantBubble.appendChild(details);
          }

          lastAssistantRaw = rawAnswer || "";
          document.getElementById("copyLastBtn").disabled = !lastAssistantRaw;

          setStatus("Done.");
          // auch hier: nur wenn unten
          maybeScrollToBottom(false);
        }else if(msg.type === "error"){
          assistantBubble.innerHTML = `<div class="md"><p><strong>Error:</strong> ${escapeHtml(msg.error || "unknown")}</p></div>`;
          setStatus("Error.");
          maybeScrollToBottom(false);
        }
      }
    }
  }catch(e){
    assistantBubble.innerHTML = `<div class="md"><p><strong>Error:</strong> ${escapeHtml(String(e))}</p></div>`;
    setStatus("Error.");
    maybeScrollToBottom(false);
  }
}

// Enter-to-send: Enter = send, Shift+Enter = newline
(function bindEnterSend(){
  const ta = document.getElementById("q");
  if(!ta) return;
  ta.addEventListener("keydown", (e) => {
    if(e.key === "Enter" && !e.shiftKey){
      e.preventDefault();
      send();
    }
  });
})();
</script>
</body>
</html>
"""
    )


@app.post("/api/ask")
def api_ask(req: AskRequest):
    """Non-stream fallback (not used by UI, but handy for debugging)."""
    try:
        if req.agent not in {"thesis", "python", "reviewer"}:
            raise HTTPException(status_code=400, detail="agent must be thesis/python/reviewer")

        if not index_ready():
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Vector index not found. Run: python main.py ingest (from project root), then reload the page."
                }
            )

        question_for_model = req.question
        if req.session_id:
            ctx = _build_context(req.session_id, DEFAULT_CONTEXT_TURNS)
            if ctx:
                question_for_model = (
                    "You are continuing an ongoing conversation. Use the context below to stay consistent.\n\n"
                    "=== Conversation context (most recent last) ===\n"
                    f"{ctx}\n"
                    "=== End context ===\n\n"
                    f"User: {req.question}"
                )

        out = ask(req.agent, question_for_model, k=req.k)

        if req.session_id:
            _store_turn(req.session_id, req.question, out.get("answer", ""))

        return JSONResponse({"ok": True, "answer": out.get("answer", ""), "sources": out.get("sources", "")})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.post("/api/ask_stream")
async def api_ask_stream(req: AskRequest):
    """
    Streaming effect via NDJSON.
    If your ask() supports true token streaming, replace chunking accordingly.
    """
    if req.agent not in {"thesis", "python", "reviewer"}:
        raise HTTPException(status_code=400, detail="agent must be thesis/python/reviewer")

    if not index_ready():
        async def err_gen():
            yield json.dumps(
                {
                    "type": "error",
                    "error": "Vector index not found. Run: python main.py ingest (from project root), then reload the page."
                },
                ensure_ascii=False,
            ) + "\n"

        return StreamingResponse(err_gen(), media_type="application/x-ndjson")

    question_for_model = req.question
    if req.session_id:
        ctx = _build_context(req.session_id, DEFAULT_CONTEXT_TURNS)
        if ctx:
            question_for_model = (
                "You are continuing an ongoing conversation. Use the context below to stay consistent.\n\n"
                "=== Conversation context (most recent last) ===\n"
                f"{ctx}\n"
                "=== End context ===\n\n"
                f"User: {req.question}"
            )

    async def gen():
        try:
            out = await asyncio.to_thread(ask, req.agent, question_for_model, req.k)

            answer = out.get("answer", "") or ""
            sources = out.get("sources", "") or ""

            if req.session_id:
                _store_turn(req.session_id, req.question, answer)

            for ch in _chunk_text(answer, size=140):
                yield json.dumps({"type": "delta", "text": ch}, ensure_ascii=False) + "\n"
                await asyncio.sleep(0)

            yield json.dumps({"type": "done", "sources": sources}, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")
