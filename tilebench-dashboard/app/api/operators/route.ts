import { ok } from "@/lib/http";
import { getOperators, getProfiles, getSourceIndex } from "@/lib/data";
import { poolMode } from "@/lib/metrics";
import type { Mode } from "@/lib/types";

/**
 * GET /api/operators
 *   ?tier=both|partial      filter by comparability
 *   ?target=1               only profiling targets
 *   ?sort=autotune|default|op   (default: autotune ratio, descending)
 */
export async function GET(req: Request) {
  const q = new URL(req.url).searchParams;
  const [all, profiles, src] = await Promise.all([getOperators(), getProfiles(), getSourceIndex()]);
  const profiled = new Set(profiles.filter((p) => p.has_report).map((p) => p.op));

  let rows = all;
  const tier = q.get("tier");
  if (tier) rows = rows.filter((r) => r.tier === tier);
  if (q.get("target") === "1") rows = rows.filter((r) => r.target);

  const sort = q.get("sort") ?? "autotune";
  rows = [...rows].sort((a, b) => {
    if (sort === "op") return a.op.localeCompare(b.op);
    const m = (sort === "default" ? "default" : "autotune") as Mode;
    const av = a[m]?.tl_over_triton;
    const bv = b[m]?.tl_over_triton;
    if (av == null && bv == null) return a.op.localeCompare(b.op);
    if (av == null) return 1; // rows missing this mode sink to the bottom
    if (bv == null) return -1;
    return bv - av;
  });

  return ok({
    operators: rows.map((r) => ({
      ...r,
      has_profile: profiled.has(r.op),
      has_source: Boolean(src[r.op]),
    })),
    pooled: { default: poolMode(all, "default"), autotune: poolMode(all, "autotune") },
    counts: {
      total: all.length,
      both: all.filter((r) => r.tier === "both").length,
      partial: all.filter((r) => r.tier === "partial").length,
    },
  });
}
