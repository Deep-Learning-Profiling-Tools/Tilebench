"use client";

import { Fragment, useCallback, useEffect, useRef, useState } from "react";
import { marked } from "marked";
import type { Operator, Platform, ProfileEntry } from "@/lib/types";
import { leader } from "@/lib/metrics";
import { label, sourceUrl, times, versus } from "./format";
import Agent from "./Agent";

interface Detail {
  operator: Operator;
  profile: (ProfileEntry & { report_present: boolean }) | null;
  source: string[];
}

type Tab = "results" | "profile" | "source" | "chat";

export default function Drawer({
  op,
  platform,
  home,
  dataset,
  onClose,
}: {
  op: string | null;
  platform: Platform;
  /** NCU write-ups exist for the home platform only */
  home: boolean;
  dataset: string;
  onClose: () => void;
}) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [tab, setTab] = useState<Tab>("results");
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

  const o = platform.operators.find((x) => x.op === op);
  const reports = (op && platform.reports[op]) || [];
  const tabs: Tab[] = home
    ? ["results", "profile", "source", "chat"]
    : ["results", "source", "chat"];
  const shown = tabs.includes(tab) ? tab : "results";

  return (
    <>
      <div className={`scrim${op ? " open" : ""}`} onClick={onClose} />
      <aside className={`drawer${op ? " open" : ""}`} aria-hidden={!op}>
        <div className="dhead">
          <div>
            <div className="eyebrow">kernel · {platform.name}</div>
            <h2 className="mono">{op ?? ""}</h2>
          </div>
          <button ref={closeRef} className="closex" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="tabs">
          {tabs.map((t) => (
            <button key={t} className={`tab${shown === t ? " on" : ""}`} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </div>

        <div className={`dbody${shown === "chat" ? " chatbody" : ""}`}>
          {warn && <div className="warnbox">{warn}</div>}
          {!detail && !warn && op && shown !== "results" && <p className="muted">loading…</p>}

          {shown === "results" && o && (
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
                        {platform.backends.map((b) => (
                          <Fragment key={b}>
                            <dt>{label(b)}</dt>
                            <dd>
                              {times(s[b])} <span className="muted">· {versus(s[b])}</span>
                            </dd>
                          </Fragment>
                        ))}
                        <dt>fastest</dt>
                        <dd>
                          {label(leader(s).name)} · {times(leader(s).margin)} the next fastest
                        </dd>
                        <dt>cases</dt>
                        <dd>{s.cases}</dd>
                      </dl>
                    ) : (
                      <p className="muted">no data</p>
                    )}
                  </div>
                );
              })}
              <div>
                <div className="eyebrow" style={{ marginBottom: 8 }}>
                  {platform.profiler} reports
                </div>
                {reports.length ? (
                  <>
                    <div className="row">
                      {reports.map((r) => (
                        <span key={r} className="badge">
                          {r}
                        </span>
                      ))}
                    </div>
                    <p style={{ fontSize: 13, margin: "10px 0 0" }}>
                      <a
                        href={`https://huggingface.co/datasets/${dataset}/tree/main/${platform.hf_path}/${op}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        <img className="hf" src="/huggingface.svg" alt="" width={16} height={16} />
                        {reports.length} reports on Hugging Face ↗
                      </a>
                    </p>
                  </>
                ) : (
                  <p className="muted">No reports published for this kernel.</p>
                )}
              </div>
              <p className="muted" style={{ fontSize: 12.5 }}>
                Each number is PyTorch time ÷ backend time on {platform.name}, as a geometric mean over the
                kernel&apos;s cases. default and
                autotune are separate measurements and are never blended.
              </p>
            </>
          )}

          {shown === "profile" && detail && (
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

          {shown === "source" && detail && (
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
              {op && (
                <p style={{ fontSize: 13, margin: 0 }}>
                  <a href={sourceUrl(op, file ?? undefined)} target="_blank" rel="noreferrer">
                    <img className="hf" src="/github.svg" alt="" width={16} height={16} />
                    {file ? `${file} on GitHub ↗` : "This kernel on GitHub ↗"}
                  </a>
                </p>
              )}
              {file && (code === null ? <p className="muted">loading…</p> : <pre className="src">{code}</pre>)}
              {!file && <p className="muted">Pick a file to read its source.</p>}
            </>
          )}

          {shown === "chat" && detail && <Agent op={op} platform={platform.id} embedded />}
        </div>
      </aside>
    </>
  );
}
