"use client";

import { Fragment, useMemo, useState } from "react";
import type { Backend, Mode, Platform, PlatformStats } from "@/lib/types";
import { fastestCounts, headToHead, leader, type Contender } from "@/lib/metrics";
import { label, times, versus } from "./format";
import Drawer from "./Drawer";

type Pooled = { default: PlatformStats; autotune: PlatformStats };
type BoardPlatform = Platform & { pooled: Pooled };
type SortKey = "op" | `${Mode}.${Backend}`;

function SummaryPanel({
  title,
  note,
  p,
  mode,
}: {
  title: string;
  note: string;
  p: BoardPlatform;
  mode: Mode;
}) {
  const s = p.pooled[mode];
  const who: Contender[] = [...p.backends, "torch"];
  const fastest = fastestCounts(p, mode);
  const kernels = p.operators.filter((o) => o[mode]).length;
  const best = leader(s).name;
  return (
    <div className="panel">
      <h3>
        {title} · {kernels} kernels · {s.cases.toLocaleString()} cases
      </h3>
      <p className="muted" style={{ fontSize: 12.5, margin: 0 }}>
        {note}
      </p>

      <table className="sum">
        <thead>
          <tr>
            <th>backend</th>
            <th title="Geometric mean over every case of PyTorch time ÷ backend time">
              speedup vs PyTorch
            </th>
            <th title="Kernels where this is the fastest of all, PyTorch included">fastest on</th>
            <th title="Kernels where this backend is faster than PyTorch">beats PyTorch on</th>
          </tr>
        </thead>
        <tbody>
          {who.map((w) => (
            <tr key={w}>
              <td>{label(w)}</td>
              <td>
                {w === "torch" ? (
                  <span className="muted">1.00× · baseline</span>
                ) : (
                  <span className={best === w ? "v lead" : "v"}>{times(s[w])}</span>
                )}
              </td>
              <td>
                {fastest[w]} <span className="muted">/ {kernels}</span>
              </td>
              <td>
                {w === "torch" ? (
                  <span className="muted">—</span>
                ) : (
                  <>
                    {headToHead(p, mode, w, "torch").wins} <span className="muted">/ {kernels}</span>
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div>
        <div className="sumcap">
          Head to head · number of kernels (out of {kernels}) where the row is faster than the
          column
        </div>
        <table className="sum h2h">
          <thead>
            <tr>
              <th>row faster than column</th>
              {who.map((c) => (
                <th key={c}>{label(c)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {who.map((r) => (
              <tr key={r}>
                <td>{label(r)}</td>
                {who.map((c) => {
                  if (r === c) {
                    return (
                      <td key={c} className="muted">
                        —
                      </td>
                    );
                  }
                  const h = headToHead(p, mode, r, c);
                  return (
                    <td
                      key={c}
                      className={h.wins > h.losses ? "ahead" : undefined}
                      title={`${label(r)} is faster than ${label(c)} on ${h.wins} kernels and slower on ${h.losses}${h.ties ? `, tied on ${h.ties}` : ""}`}
                    >
                      {h.wins} <span className="of">/ {kernels}</span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ModeCells({ s, backends }: { s: PlatformStats | null; backends: Backend[] }) {
  if (!s) {
    return (
      <>
        {[...backends, "fastest"].map((b, i) => (
          <td key={b} className={i === 0 ? "sep muted" : "muted"}>
            —
          </td>
        ))}
      </>
    );
  }
  const lead = leader(s).name;
  return (
    <>
      {backends.map((b, i) => {
        const v = s[b];
        const cls = lead === b ? "v lead" : v != null && v < 1 ? "v slow" : "v";
        return (
          <td key={b} className={i === 0 ? "sep" : undefined} title={`${label(b)}: ${versus(v)}`}>
            <span className={cls}>{times(v)}</span>
          </td>
        );
      })}
      <td className={`best${lead === "torch" ? " torch" : ""}`}>{label(lead)}</td>
    </>
  );
}

export default function Board({
  platforms,
  dataset,
  writeups,
}: {
  platforms: BoardPlatform[];
  dataset: string;
  writeups: string[];
}) {
  const [pid, setPid] = useState(platforms[0].id);
  const [sort, setSort] = useState<SortKey>("op");
  const [open, setOpen] = useState<string | null>(null);

  const p = platforms.find((x) => x.id === pid) ?? platforms[0];
  const isHome = p.id === platforms[0].id;

  const rows = useMemo(() => {
    const copy = [...p.operators];
    copy.sort((a, b) => {
      if (sort === "op") return a.op.localeCompare(b.op);
      const [m, k] = sort.split(".") as [Mode, Backend];
      const av = a[m]?.[k];
      const bv = b[m]?.[k];
      if (av == null && bv == null) return a.op.localeCompare(b.op);
      if (av == null) return 1;
      if (bv == null) return -1;
      return bv - av;
    });
    return copy;
  }, [p, sort]);

  const pick = (id: string) => {
    const next = platforms.find((x) => x.id === id);
    if (next && sort !== "op" && !next.backends.includes(sort.split(".")[1] as Backend)) {
      setSort("op");
    }
    setPid(id);
  };

  const th = (key: SortKey, label: string) => (
    <button
      className="sortbtn"
      aria-sort={sort === key ? "descending" : undefined}
      onClick={() => setSort(key)}
    >
      {label}
    </button>
  );

  const span = p.backends.length + 1;

  const hardware = (
    <div className="hwbar">
      <span className="eyebrow">current hardware</span>
      <div className="seg" role="tablist">
        {platforms.map((x) => (
          <button
            key={x.id}
            role="tab"
            aria-selected={x.id === p.id}
            className={`segbtn${x.id === p.id ? " on" : ""}`}
            onClick={() => pick(x.id)}
          >
            {x.name}
          </button>
        ))}
      </div>
      <span className="hwmeta">
        {p.note} · {p.backends.map(label).join(", ")} ·{" "}
        {Object.values(p.reports).reduce((a, r) => a + r.length, 0)} {p.profiler} reports
      </span>
    </div>
  );

  return (
    <>
      <section>
        <div className="sechead">
          <h2>Kernel List</h2>
          <p className="dek">
            Every number is a speedup over PyTorch on {p.name}: PyTorch&apos;s time divided by that
            backend&apos;s time, as a geometric mean over the kernel&apos;s input sizes and dtypes.
            Above 1× the backend beats PyTorch, below 1× it is slower. Click a column to sort, or a
            row for details, source and chat.
          </p>
          <div className="legend">
            <span>
              <span className="v lead">2.39×</span> fastest of the backends, and faster than
              PyTorch
            </span>
            <span>
              <span className="v">1.88×</span> faster than PyTorch
            </span>
            <span>
              <span className="v slow">0.306×</span> slower than PyTorch (here 3.27× slower)
            </span>
            {isHome && (
              <span>
                <span className="badge">ncu</span> has a written profiling analysis
              </span>
            )}
          </div>
        </div>
        {hardware}
        <div className="tablewrap">
          <table style={{ minWidth: 300 + span * 2 * 96 }}>
            <thead>
              <tr>
                <th rowSpan={2}>{th("op", "operator")}</th>
                <th className="grp" colSpan={span}>
                  default · speedup vs PyTorch
                </th>
                <th className="grp" colSpan={span}>
                  autotune · speedup vs PyTorch
                </th>
              </tr>
              <tr>
                {(["default", "autotune"] as const).map((m) => (
                  <Fragment key={m}>
                    {p.backends.map((b, i) => (
                      <th key={b} className={i === 0 ? "sep" : undefined}>
                        {th(`${m}.${b}`, b)}
                      </th>
                    ))}
                    <th>fastest</th>
                  </Fragment>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.op} className={open === r.op ? "sel" : ""} onClick={() => setOpen(r.op)}>
                  <td>
                    <span className="opname">
                      <span className="mono">{r.op}</span>
                      {isHome && writeups.includes(r.op) && <span className="badge">ncu</span>}
                    </span>
                  </td>
                  <ModeCells s={r.default} backends={p.backends} />
                  <ModeCells s={r.autotune} backends={p.backends} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <div className="sechead">
          <h2>Summary Metrics</h2>
        </div>
        {hardware}
        <div className="pooled">
          <SummaryPanel
            title="default"
            note="Every backend at the same hyperparameters when applicable."
            p={p}
            mode="default"
          />
          <SummaryPanel
            title="autotune"
            note="Every backend at the configuration that autotune selected."
            p={p}
            mode="autotune"
          />
        </div>
      </section>

      <Drawer
        op={open}
        platform={p}
        home={isHome}
        dataset={dataset}
        onClose={() => setOpen(null)}
      />
    </>
  );
}
