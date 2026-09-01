import type { ModeStats, Operator, PooledStats, Mode } from "./types";

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

/** Verdict banding, shared by the table and the API so they never disagree. */
export type Band = "win" | "parity" | "behind" | "loss";

export function band(ratio: number): Band {
  if (ratio >= 1.05) return "win";
  if (ratio >= 0.95) return "parity";
  if (ratio >= 0.7) return "behind";
  return "loss";
}
