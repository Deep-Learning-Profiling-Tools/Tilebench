import { getOperators, getProfiles, getSourceIndex } from "@/lib/data";
import { poolMode } from "@/lib/metrics";
import Board from "@/components/Board";

export const dynamic = "force-dynamic";

export default async function Page() {
  const [ops, profiles, src] = await Promise.all([
    getOperators(),
    getProfiles(),
    getSourceIndex(),
  ]);
  const profiled = new Set(profiles.filter((p) => p.has_report).map((p) => p.op));
  const rows = ops.map((o) => ({
    ...o,
    has_profile: profiled.has(o.op),
    has_source: Boolean(src[o.op]),
  }));

  const pooled = { default: poolMode(ops, "default"), autotune: poolMode(ops, "autotune") };
  const cases = pooled.default.cases + pooled.autotune.cases;
  const partial = ops.filter((o) => o.tier === "partial").length;

  return (
    <main className="wrap">
      <header>
        <div className="eyebrow">tilebench · b200 sm_100</div>
        <h1>TileLang against Triton, cuTile and torch</h1>
        <p className="dek">
          {ops.length} kernels measured in two modes on one B200. default and autotune are held
          apart everywhere — they answer different questions, and averaging them hides which one a
          number came from.
        </p>
        <div className="status">
          <span className="chip">
            <span className="dot" />
            {ops.length - partial} clean in both modes
          </span>
          <span className="chip warn">
            <span className="dot" />
            {partial} partial
          </span>
          <span className="chip">
            <span className="dot" />
            {cases.toLocaleString()} cases · 0 nan
          </span>
          <span className="chip info">
            <span className="dot" />
            {profiled.size} NCU write-ups
          </span>
        </div>
      </header>

      <Board operators={rows} pooled={pooled} />

      <footer className="foot">
        <p>
          Speedups are geometric means against torch on identical shapes. Pooled figures weight each
          operator by its case count, which is exact for a geomean of geomeans.
        </p>
        <p>
          Engineer notes are unverified observations recorded against a kernel, not established
          findings. NCU write-ups state separately what they established and what they did not.
        </p>
      </footer>
    </main>
  );
}
