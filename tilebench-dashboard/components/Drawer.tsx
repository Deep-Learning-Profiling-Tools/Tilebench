"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { marked } from "marked";
import type { Note, Operator, ProfileEntry } from "@/lib/types";
import { num, when } from "./format";
import Agent from "./Agent";

interface Detail {
  operator: Operator;
  notes: Note[];
  profile: (ProfileEntry & { report_present: boolean }) | null;
  source: string[];
}

type Tab = "results" | "notes" | "profile" | "source" | "chat";

export default function Drawer({ op, onClose }: { op: string | null; onClose: () => void }) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [tab, setTab] = useState<Tab>("results");
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [warn, setWarn] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [file, setFile] = useState<string | null>(null);
  const [code, setCode] = useState<string | null>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!op) return;
    setDetail(null);
    setTab("results");
    setReport(null);
    setCode(null);
    setFile(null);
    setWarn(null);
    fetch(`/api/operators/${op}`)
      .then((r) => r.json())
      .then(setDetail)
      .catch(() => setWarn("could not load this operator"));
  }, [op]);

  useEffect(() => {
    if (!op) return;
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    closeRef.current?.focus();
    return () => window.removeEventListener("keydown", esc);
  }, [op, onClose]);

  useEffect(() => {
    if (tab !== "profile" || !op || report !== null) return;
    fetch(`/api/profile/${op}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setReport(d?.report ?? ""));
  }, [tab, op, report]);

  const openFile = useCallback(
    (f: string) => {
      if (!op) return;
      setFile(f);
      setCode(null);
      fetch(`/api/source/${op}/${f}`)
        .then((r) => r.text())
        .then(setCode);
    },
    [op]
  );

  async function addNote() {
    if (!op || !draft.trim()) return;
    setSaving(true);
    setWarn(null);
    try {
      const r = await fetch(`/api/notes/${op}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ body: draft }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error ?? "failed");
      setDetail((prev) => (prev ? { ...prev, notes: d.notes } : prev));
      setDraft("");
      if (!d.durable) {
        setWarn(
          "Saved to this server instance only. Notes are not durable until a KV store is attached — see the README."
        );
      }
    } catch (e) {
      setWarn(e instanceof Error ? e.message : "failed to save note");
    } finally {
      setSaving(false);
    }
  }

  const o = detail?.operator;
  const tabs: Tab[] = ["results", "notes", "profile", "source", "chat"];

  return (
    <>
      <div className={`scrim${op ? " open" : ""}`} onClick={onClose} />
      <aside className={`drawer${op ? " open" : ""}`} aria-hidden={!op}>
        <div className="dhead">
          <div>
            <div className="eyebrow">kernel</div>
            <h2 className="mono">{op ?? ""}</h2>
          </div>
          <button ref={closeRef} className="closex" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="tabs">
          {tabs.map((t) => (
            <button key={t} className={`tab${tab === t ? " on" : ""}`} onClick={() => setTab(t)}>
              {t}
              {t === "notes" && detail?.notes.length ? ` (${detail.notes.length})` : ""}
            </button>
          ))}
        </div>

        <div className={`dbody${tab === "chat" ? " chatbody" : ""}`}>
          {!detail && op && <p className="muted">loading…</p>}

          {tab === "results" && o && (
            <>
              {(["default", "autotune"] as const).map((m) => {
                const s = o[m];
                return (
                  <div key={m}>
                    <div className="eyebrow" style={{ marginBottom: 8 }}>
                      {m}
                    </div>
                    {s ? (
                      <dl className="kv">
                        <dt>triton</dt>
                        <dd>{num(s.triton)}×</dd>
                        <dt>cutile</dt>
                        <dd>{num(s.cutile)}×</dd>
                        <dt>tilelang</dt>
                        <dd>{num(s.tilelang)}×</dd>
                        <dt>tl ÷ triton</dt>
                        <dd>{num(s.tl_over_triton)}</dd>
                        <dt>tl ÷ cutile</dt>
                        <dd>{num(s.tl_over_cutile)}</dd>
                        <dt>cases</dt>
                        <dd>{s.cases}</dd>
                      </dl>
                    ) : (
                      <p className="muted">
                        excluded — {o.autotune_excluded_reason ?? "no data"}
                      </p>
                    )}
                  </div>
                );
              })}
              <p className="muted" style={{ fontSize: 12.5 }}>
                Backend figures are geomean speedup vs torch. default and autotune are separate
                measurements and are never blended.
              </p>
            </>
          )}

          {tab === "notes" && detail && (
            <>
              {warn && <div className="warnbox">{warn}</div>}
              {detail.notes.length === 0 && <p className="muted">No notes on this kernel yet.</p>}
              {detail.notes.map((n) => (
                <div className="note" key={n.ts}>
                  <time>{when(n.ts)}</time>
                  {n.body}
                </div>
              ))}
              <div>
                <textarea
                  rows={4}
                  value={draft}
                  placeholder="What did you find?"
                  onChange={(e) => setDraft(e.target.value)}
                />
                <div className="row" style={{ marginTop: 8 }}>
                  <button className="act" onClick={addNote} disabled={saving || !draft.trim()}>
                    {saving ? "saving…" : "add note"}
                  </button>
                </div>
              </div>
            </>
          )}

          {tab === "profile" && detail && (
            <>
              {!detail.profile && (
                <p className="muted">
                  No NCU run for this kernel yet. Profiled kernels expose their write-up and the
                  list of <code>.ncu-rep</code> artifacts here.
                </p>
              )}
              {detail.profile && (
                <>
                  <dl className="kv">
                    <dt>run</dt>
                    <dd>{detail.profile.run}</dd>
                    <dt>ncu-rep</dt>
                    <dd>{detail.profile.ncu_reports.length} files</dd>
                    <dt>repo path</dt>
                    <dd style={{ overflowWrap: "anywhere" }}>{detail.profile.repo_path}</dd>
                  </dl>
                  {report ? (
                    <div
                      className="md"
                      dangerouslySetInnerHTML={{ __html: marked.parse(report) as string }}
                    />
                  ) : (
                    <p className="muted">loading write-up…</p>
                  )}
                </>
              )}
            </>
          )}

          {tab === "source" && detail && (
            <>
              <div className="row">
                {detail.source.map((f) => (
                  <button
                    key={f}
                    className={file === f ? "act" : "ghost"}
                    onClick={() => openFile(f)}
                  >
                    {f}
                  </button>
                ))}
              </div>
              {file && (code === null ? <p className="muted">loading…</p> : <pre className="src">{code}</pre>)}
              {!file && <p className="muted">Pick a file to read its source.</p>}
            </>
          )}

          {tab === "chat" && detail && <Agent op={op} embedded />}
        </div>
      </aside>
    </>
  );
}
