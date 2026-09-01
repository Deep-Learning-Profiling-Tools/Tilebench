"use client";

import { useMemo, useState } from "react";
import type { Operator, PooledStats } from "@/lib/types";
import { num, band } from "./format";
import Drawer from "./Drawer";

type Row = Operator & { has_profile: boolean; has_source: boolean };
type SortKey = "autotune" | "default" | "op";

function Cell({ v }: { v: number | null | undefined }) {
  const b = band(v);
  if (b === "none") return <span className="muted">—</span>;
  return <span className={`v ${b}`}>{num(v)}</span>;
}

function Pooled({ title, s, note }: { title: string; s: PooledStats; note: string }) {
  const max = Math.max(s.triton, s.cutile, s.tilelang, 1);
  const rows: Array<[string, number, boolean]> = [
    ["triton", s.triton, false],
    ["cutile", s.cutile, false],
    ["tilelang", s.tilelang, true],
  ];
  return (
    <div className="panel">
      <h3>
        {title} · {s.cases.toLocaleString()} cases
      </h3>
      <div className="bars">
        {rows.map(([label, v, tl]) => (
          <div className="bar" key={label}>
            <span className="mono">{label}</span>
            <span className="track">
              <span className={`fill${tl ? " tl" : ""}`} style={{ width: `${(v / max) * 100}%` }} />
            </span>
            <span className="val">{num(v, 3)}×</span>
          </div>
        ))}
      </div>
      <div className="ratio">
        <span>
          TileLang ÷ Triton <b>{num(s.tl_over_triton)}</b>
        </span>
        <span>
          ÷ cuTile <b>{num(s.tl_over_cutile)}</b>
        </span>
      </div>
      <p className="muted" style={{ fontSize: 12.5, margin: 0 }}>
        {note}
      </p>
    </div>
  );
}

export default function Board({
  operators,
  pooled,
}: {
  operators: Row[];
  pooled: { default: PooledStats; autotune: PooledStats };
}) {
  const [sort, setSort] = useState<SortKey>("autotune");
  const [open, setOpen] = useState<string | null>(null);

  const rows = useMemo(() => {
    const copy = [...operators];
    copy.sort((a, b) => {
      if (sort === "op") return a.op.localeCompare(b.op);
      const m = sort === "default" ? "default" : "autotune";
      const av = a[m]?.tl_over_triton;
      const bv = b[m]?.tl_over_triton;
      if (av == null && bv == null) return a.op.localeCompare(b.op);
      if (av == null) return 1;
      if (bv == null) return -1;
      return bv - av;
    });
    return copy;
  }, [operators, sort]);

  const th = (key: SortKey, label: string) => (
    <button
      className="sortbtn"
      aria-sort={sort === key ? "descending" : undefined}
      onClick={() => setSort(key)}
    >
      {label}
    </button>
  );

  return (
    <>
      <section>
        <div className="sechead">
          <div className="eyebrow">headline</div>
          <h2>Pooled across every comparable case</h2>
        </div>
        <div className="pooled">
          <Pooled
            title="default"
            s={pooled.default}
            note="Every backend at its own out-of-the-box configuration."
          />
          <Pooled
            title="autotune"
            s={pooled.autotune}
            note="Every backend at the winner its own tuner picked."
          />
        </div>
      </section>

      <section>
        <div className="sechead">
          <div className="eyebrow">per kernel</div>
          <h2>The {operators.length} you can compare</h2>
          <p className="dek">
            Sorted by TileLang ÷ Triton in autotune. Click any row for results, notes, the NCU
            write-up, source and kernel-scoped chat. Hatched rows are missing one mode.
          </p>
        </div>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th rowSpan={2}>{th("op", "operator")}</th>
                <th className="grp" colSpan={5}>
                  default
                </th>
                <th className="grp" colSpan={5}>
                  autotune
                </th>
              </tr>
              <tr>
                <th className="grp">tri</th>
                <th>cut</th>
                <th>tl</th>
                <th>{th("default", "tl÷tri")}</th>
                <th>tl÷cut</th>
                <th className="grp">tri</th>
                <th>cut</th>
                <th>tl</th>
                <th>{th("autotune", "tl÷tri")}</th>
                <th>tl÷cut</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.op}
                  className={`${r.tier === "partial" ? "partial " : ""}${open === r.op ? "sel" : ""}`}
                  onClick={() => setOpen(r.op)}
                >
                  <td>
                    <span className="opname">
                      {r.target && <span className="star" title="profiling target">★</span>}
                      <span className="mono">{r.op}</span>
                      {r.has_profile && <span className="badge">ncu</span>}
                    </span>
                  </td>
                  <td className="sep">{num(r.default?.triton)}</td>
                  <td>{num(r.default?.cutile)}</td>
                  <td>{num(r.default?.tilelang)}</td>
                  <td>
                    <Cell v={r.default?.tl_over_triton} />
                  </td>
                  <td>
                    <Cell v={r.default?.tl_over_cutile} />
                  </td>
                  <td className="sep">{num(r.autotune?.triton)}</td>
                  <td>{num(r.autotune?.cutile)}</td>
                  <td>{num(r.autotune?.tilelang)}</td>
                  <td>
                    <Cell v={r.autotune?.tl_over_triton} />
                  </td>
                  <td>
                    <Cell v={r.autotune?.tl_over_cutile} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <Drawer op={open} onClose={() => setOpen(null)} />
    </>
  );
}
