"use client";

import { marked } from "marked";
import { useEffect, useMemo, useRef, useState } from "react";

type AgentMeta = { reports?: string[]; model?: string; provider?: string };
type ToolEvent = {
  id?: string;
  name?: string;
  status?: "start" | "done";
  ok?: boolean;
  preview?: unknown;
};
type ChatTurn = { role: "user" | "assistant"; content: string };
type ThreadState = { key: string; messages: ChatTurn[] };

const API_KEY_STORAGE = "tilebench.openrouter_api_key";
const THREAD_STORAGE_PREFIX = "tilebench.agent.thread";
const MAX_THREAD_MESSAGES = 20;
const MAX_THREAD_CHARS = 24_000;
const MARKDOWN_TAGS = new Set([
  "A",
  "B",
  "BLOCKQUOTE",
  "BR",
  "CODE",
  "DEL",
  "DIV",
  "EM",
  "H1",
  "H2",
  "H3",
  "H4",
  "HR",
  "I",
  "LI",
  "OL",
  "P",
  "PRE",
  "S",
  "SPAN",
  "STRONG",
  "TABLE",
  "TBODY",
  "TD",
  "TH",
  "THEAD",
  "TR",
  "UL",
]);

function sanitizeMarkdownHtml(html: string): string {
  if (typeof DOMParser === "undefined") return "";
  const doc = new DOMParser().parseFromString(html, "text/html");
  const nodes = Array.from(doc.body.querySelectorAll("*"));
  for (const node of nodes) {
    if (!MARKDOWN_TAGS.has(node.tagName)) {
      node.replaceWith(...Array.from(node.childNodes));
      continue;
    }

    const href = node.tagName === "A" ? node.getAttribute("href") ?? "" : "";
    for (const attr of Array.from(node.attributes)) {
      node.removeAttribute(attr.name);
    }

    if (node.tagName === "A") {
      if (/^https?:\/\//.test(href)) {
        node.setAttribute("href", href);
        node.setAttribute("target", "_blank");
        node.setAttribute("rel", "noreferrer");
      }
    }
  }
  return doc.body.innerHTML;
}

function markdownHtml(markdown: string): string {
  return sanitizeMarkdownHtml(
    marked.parse(markdown, {
      async: false,
      breaks: true,
      gfm: true,
    }) as string
  );
}

function makeThreadKey(op: string | null | undefined): string {
  return `${THREAD_STORAGE_PREFIX}.${op ?? "global"}`;
}

function trimThread(messages: ChatTurn[]): ChatTurn[] {
  const valid = messages
    .filter((m) => (m.role === "user" || m.role === "assistant") && m.content.trim())
    .slice(-MAX_THREAD_MESSAGES);
  const kept: ChatTurn[] = [];
  let chars = 0;
  for (let i = valid.length - 1; i >= 0; i--) {
    const msg = valid[i];
    const remaining = MAX_THREAD_CHARS - chars;
    if (remaining <= 0) break;
    const content = msg.content.length > remaining ? msg.content.slice(-remaining) : msg.content;
    kept.push({ role: msg.role, content });
    chars += content.length;
  }
  return kept.reverse();
}

function parseThread(raw: string | null): ChatTurn[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return trimThread(
      parsed.flatMap((item): ChatTurn[] => {
        if (!item || typeof item !== "object") return [];
        const row = item as Record<string, unknown>;
        if ((row.role !== "user" && row.role !== "assistant") || typeof row.content !== "string") {
          return [];
        }
        return [{ role: row.role, content: row.content }];
      })
    );
  } catch {
    return [];
  }
}

function MarkdownBlock({ content }: { content: string }) {
  const html = useMemo(() => markdownHtml(content), [content]);
  return <div className="answer md" dangerouslySetInnerHTML={{ __html: html }} />;
}

/** Streams /api/agent over SSE. Retrieval stays server-side so tool access is
 *  over the deployed corpus; a pasted key is used only for the model call. */
export default function Agent({ op, embedded = false }: { op?: string | null; embedded?: boolean }) {
  const [q, setQ] = useState("");
  const [apiKey, setApiKey] = useState("");
  const historyKey = useMemo(() => makeThreadKey(op), [op]);
  const [thread, setThread] = useState<ThreadState>({ key: "", messages: [] });
  const [meta, setMeta] = useState<AgentMeta | null>(null);
  const [tools, setTools] = useState<ToolEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const messages = thread.key === historyKey ? thread.messages : [];

  useEffect(() => {
    setApiKey(window.sessionStorage.getItem(API_KEY_STORAGE) ?? "");
  }, []);

  useEffect(() => {
    setThread({
      key: historyKey,
      messages: parseThread(window.sessionStorage.getItem(historyKey)),
    });
    setMeta(null);
    setTools([]);
    setErr(null);
    setQ("");
  }, [historyKey]);

  useEffect(() => {
    const key = apiKey.trim();
    if (key) window.sessionStorage.setItem(API_KEY_STORAGE, key);
    else window.sessionStorage.removeItem(API_KEY_STORAGE);
  }, [apiKey]);

  useEffect(() => {
    if (thread.key !== historyKey) return;
    if (thread.messages.length) {
      window.sessionStorage.setItem(historyKey, JSON.stringify(trimThread(thread.messages)));
    } else {
      window.sessionStorage.removeItem(historyKey);
    }
  }, [historyKey, thread]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight });
  }, [messages, tools]);

  async function ask() {
    if (!q.trim() || busy) return;
    const userText = q.trim();
    const requestMessages = trimThread([...messages, { role: "user", content: userText }]);
    let assistantText = "";
    setBusy(true);
    setMeta(null);
    setTools([]);
    setErr(null);
    setQ("");
    setThread({ key: historyKey, messages: [...requestMessages, { role: "assistant", content: "" }] });
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;

    try {
      const res = await fetch("/api/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: requestMessages,
          contextOp: op ?? undefined,
          apiKey: apiKey.trim() || undefined,
        }),
        signal: ctl.signal,
      });

      if (!res.ok || !res.body) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.hint ?? d.error ?? `request failed (${res.status})`);
      }

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const frames = buf.split("\n\n");
        buf = frames.pop() ?? "";
        for (const f of frames) {
          const ev = /^event: (.+)$/m.exec(f)?.[1];
          const dataLine = /^data: (.+)$/m.exec(f)?.[1];
          if (!ev || !dataLine) continue;
          const data = JSON.parse(dataLine);
          if (ev === "meta") setMeta(data);
          else if (ev === "tool") setTools((prev) => [...prev, data]);
          else if (ev === "delta") {
            const text = typeof data.text === "string" ? data.text : "";
            assistantText += text;
            setThread((prev) => {
              if (prev.key !== historyKey) return prev;
              const next = [...prev.messages];
              const last = next[next.length - 1];
              if (last?.role === "assistant") next[next.length - 1] = { role: "assistant", content: assistantText };
              else next.push({ role: "assistant", content: assistantText });
              return { key: prev.key, messages: next };
            });
          } else if (ev === "error") {
            setErr(data.message);
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") setErr(e instanceof Error ? e.message : "failed");
    } finally {
      if (!assistantText) {
        setThread((prev) => {
          if (prev.key !== historyKey) return prev;
          const next = [...prev.messages];
          if (next[next.length - 1]?.role === "assistant" && !next[next.length - 1].content) next.pop();
          return { key: prev.key, messages: next };
        });
      }
      setBusy(false);
    }
  }

  function clearThread() {
    abort.current?.abort();
    setThread({ key: historyKey, messages: [] });
    setMeta(null);
    setTools([]);
    setErr(null);
    setQ("");
    setBusy(false);
  }

  return (
    <section id={embedded ? undefined : "agent"} className={embedded ? "agentembed" : undefined}>
      {!embedded && (
        <div className="sechead">
          <div className="eyebrow">agent</div>
          <h2>Chat with the performance agent</h2>
        </div>
      )}

      <div className="panel agentpanel chatconsole">
        <div className="keybar">
          <input
            type="password"
            value={apiKey}
            placeholder="OpenRouter key"
            autoComplete="off"
            spellCheck={false}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <button
            className="iconbtn keyclear"
            type="button"
            onClick={() => setApiKey("")}
            disabled={!apiKey}
            aria-label="Clear API key"
            title="Clear API key"
          >
            x
          </button>
        </div>

        <div className="chatlog" ref={transcriptRef}>
          {messages.map((message, i) => (
            <div key={`${message.role}-${i}`} className={`chatmsg ${message.role}`}>
              <div className="chatrole mono">{message.role === "user" ? "you" : "agent"}</div>
              {message.role === "assistant" ? (
                message.content ? <MarkdownBlock content={message.content} /> : <div className="answer pending">thinking...</div>
              ) : (
                <div className="userbubble">{message.content}</div>
              )}
            </div>
          ))}

          {tools.length ? (
            <details className="toolbox">
              <summary>tool trace ({tools.length})</summary>
              <div className="tooltrace">
                {tools.map((tool, i) => (
                  <details key={`${tool.id ?? tool.name}-${i}`} className={tool.ok === false ? "bad" : ""}>
                    <summary>
                      <span className="mono">{tool.status}</span>
                      <span className="mono">{tool.name}</span>
                    </summary>
                    {tool.preview !== undefined ? (
                      <pre>{JSON.stringify(tool.preview, null, 2)}</pre>
                    ) : null}
                  </details>
                ))}
              </div>
            </details>
          ) : null}
        </div>

        <div className="composer">
          <button
            className="iconbtn"
            type="button"
            onClick={clearThread}
            disabled={busy || messages.length === 0}
            aria-label="New chat"
            title="New chat"
          >
            +
          </button>
          <textarea
            className="composerinput"
            rows={1}
            value={q}
            placeholder={op ? `Ask about ${op}` : "Ask about TileBench"}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
            }}
          />
          <button className="sendbtn" onClick={ask} disabled={busy || !q.trim()}>
            {busy ? "..." : "send"}
          </button>
        </div>

        {meta?.reports?.length || meta?.model || (!embedded && op) ? (
          <div className="chatmeta">
            {meta?.reports?.length ? (
              <span className="muted" style={{ fontSize: 12 }}>
                grounded in: {meta.reports.join(", ")}
              </span>
            ) : null}
            {meta?.model ? (
              <span className="muted" style={{ fontSize: 12 }}>
                {meta.provider ?? "model"}: {meta.model}
              </span>
            ) : null}
            {!embedded && op ? (
              <span className="muted" style={{ fontSize: 12 }}>
                kernel: <span className="mono">{op}</span>
              </span>
            ) : null}
          </div>
        ) : null}

        {err && <div className="warnbox">{err}</div>}
      </div>
    </section>
  );
}
