import type {
  Backend,
  ModeStats,
  Operator,
  PooledStats,
  Mode,
  Platform,
  PlatformStats,
} from "./types";

/**
 * Pool per-operator geomeans into one geomean, weighted by case count.
 *
 *   exp( sum(n_i * ln g_i) / sum(n_i) )
 *
 * This is exact: a geomean of geomeans weighted by population size equals the
 * geomean over the pooled population. It is why the board never needs to hold
 * all 4,100 individual cases in memory to report a headline number.
 */
export function pooledGeomean(values: Array<[number, number]>): number {
  let num = 0;
  let den = 0;
  for (const [g, n] of values) {
    if (!(g > 0) || !(n > 0)) continue;
    num += n * Math.log(g);
    den += n;
  }
  return den === 0 ? NaN : Math.exp(num / den);
}

const KEYS = ["triton", "cutile", "tilelang", "tl_over_triton", "tl_over_cutile"] as const;

export function poolMode(ops: Operator[], mode: Mode): PooledStats {
  const rows = ops.map((o) => o[mode]).filter((m): m is ModeStats => m !== null);
  const out: Record<string, number> = {};
  for (const k of KEYS) {
    out[k] = pooledGeomean(rows.map((r) => [r[k], r.cases] as [number, number]));
  }
  return {
    ...(out as unknown as Omit<PooledStats, "cases" | "operators">),
    cases: rows.reduce((a, r) => a + r.cases, 0),
    operators: rows.length,
  };
}

export const BACKENDS = ["triton", "cutile", "tilelang"] as const;

/**
 * Fastest backend in one mode. Figures are speedups over torch, so torch sits at
 * 1.0 and wins when nothing beats it. `margin` is the lead over the runner-up.
 */
export function leader(s: Partial<Record<Backend, number>>): {
  name: Backend | "torch";
  margin: number;
} {
  const ranked = [
    ...BACKENDS.map((b) => [b, s[b] ?? NaN] as [Backend | "torch", number]),
    ["torch", 1] as [Backend | "torch", number],
  ]
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]);
  return { name: ranked[0][0], margin: ranked.length > 1 ? ranked[0][1] / ranked[1][1] : NaN };
}

/** poolMode for one platform: only the backends that hardware was measured with. */
export function poolPlatform(p: Platform, mode: Mode): PlatformStats {
  const rows = p.operators.map((o) => o[mode]).filter((m): m is PlatformStats => m !== null);
  const out: PlatformStats = { cases: rows.reduce((a, r) => a + r.cases, 0) };
  for (const b of p.backends) {
    out[b] = pooledGeomean(rows.map((r) => [r[b] ?? NaN, r.cases] as [number, number]));
  }
  return out;
}

/** The backends of a comparison plus the PyTorch baseline, which sits at 1.0 by definition. */
export type Contender = Backend | "torch";

const speedup = (s: PlatformStats, who: Contender) => (who === "torch" ? 1 : s[who]);

/**
 * Kernels won / lost / tied by `row` against `col` in one mode. A kernel is won when the
 * row's geomean speedup over PyTorch is the higher one, which is the same as the geomean of
 * the per-case latency ratio between the two favouring the row.
 */
export function headToHead(p: Platform, mode: Mode, row: Contender, col: Contender) {
  let wins = 0;
  let losses = 0;
  let ties = 0;
  for (const o of p.operators) {
    const s = o[mode];
    if (!s) continue;
    const a = speedup(s, row);
    const b = speedup(s, col);
    if (a == null || b == null) continue;
    if (a > b) wins++;
    else if (a < b) losses++;
    else ties++;
  }
  return { wins, losses, ties };
}

/** How many kernels each contender is the fastest on, in one mode. */
export function fastestCounts(p: Platform, mode: Mode): Record<Contender, number> {
  const out: Record<Contender, number> = { triton: 0, cutile: 0, tilelang: 0, torch: 0 };
  for (const o of p.operators) {
    const s = o[mode];
    if (s) out[leader(s).name]++;
  }
  return out;
}

/** Verdict banding, shared by the table and the API so they never disagree. */
export type Band = "win" | "parity" | "behind" | "loss";

export function band(ratio: number): Band {
  if (ratio >= 1.05) return "win";
  if (ratio >= 0.95) return "parity";
  if (ratio >= 0.7) return "behind";
  return "loss";
}
