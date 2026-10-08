import { getPlatforms, getProfiles } from "@/lib/data";
import { poolPlatform } from "@/lib/metrics";
import Board from "@/components/Board";
import { REPO } from "@/components/format";

export const dynamic = "force-dynamic";

export default async function Page() {
  const [data, profiles] = await Promise.all([getPlatforms(), getProfiles()]);
  const platforms = data.platforms.map((p) => ({
    ...p,
    pooled: { default: poolPlatform(p, "default"), autotune: poolPlatform(p, "autotune") },
  }));

  const kernels = new Set(platforms.flatMap((p) => p.operators.map((o) => o.op))).size;
  const cases = platforms.reduce((a, p) => a + p.pooled.default.cases + p.pooled.autotune.cases, 0);
  const reports = platforms.reduce(
    (a, p) => a + Object.values(p.reports).reduce((n, r) => n + r.length, 0),
    0
  );
  const names = platforms.map((p) => p.name);

  return (
    <main className="wrap">
      <header>
        <img
          className="logo"
          src="/tilebench-logo.png"
          alt="TileBench++"
          width={800}
          height={360}
        />
        <div className="eyebrow">tilebench++ · multi-architecture</div>
        <h1>
          TileBench<span className="pp">++</span>
        </h1>
        <p className="dek">
          {kernels} accelerator kernels spanning point-wise, reduction and normalization, matrix
          multiplication and attention, stencil and convolution, and data-layout workloads,
          benchmarked on {names.slice(0, -1).join(", ")} and {names[names.length - 1]}. Every
          kernel has default and autotuned results. Click a kernel row to ask questions about its
          performance.
        </p>
        <div className="status">
          <span className="chip">
            <span className="dot" />
            {platforms.length} platforms
          </span>
          <span className="chip">
            <span className="dot" />
            {kernels} kernels
          </span>
          <span className="chip">
            <span className="dot" />
            {cases.toLocaleString()} cases
          </span>
          <a className="chip" href={REPO} target="_blank" rel="noreferrer">
            <img className="hf" src="/github.svg" alt="" width={16} height={16} />
            Code on GitHub ↗
          </a>
          <a
            className="chip info"
            href={`https://huggingface.co/datasets/${data.reports_dataset}`}
            target="_blank"
            rel="noreferrer"
          >
            <img className="hf" src="/huggingface.svg" alt="" width={16} height={16} />
            {reports} profiling reports on Hugging Face ↗
          </a>
        </div>
      </header>

      <Board
        platforms={platforms}
        dataset={data.reports_dataset}
        writeups={profiles.filter((p) => p.has_report).map((p) => p.op)}
      />

      <footer className="foot">
        <p>
          Speedups are geometric means against PyTorch on identical shapes, measured on the same
          hardware as the backend. Pooled figures weight each operator by its case count, which is
          exact for a geomean of geomeans. Platforms are not compared on absolute latency.
        </p>
        <p>
          NCU write-ups cover the B200 only; the chat agent answers for every platform.
        </p>
      </footer>
    </main>
  );
}
