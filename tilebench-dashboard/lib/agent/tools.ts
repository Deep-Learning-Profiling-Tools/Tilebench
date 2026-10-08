import "server-only";
import fs from "node:fs/promises";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { getOperator, getOperators, getPlatforms, getSourceFile, getSourceIndex } from "@/lib/data";
import { poolMode, poolPlatform } from "@/lib/metrics";
import type { Mode } from "@/lib/types";

const DATA = path.join(process.cwd(), "data");
const DEFAULT_FILE_CHARS = 80_000;
const MAX_FILE_CHARS = 180_000;
const MAX_TOOL_RESULT_CHARS = 180_000;
const SAFE_OP = /^[a-z0-9_]+$/;
const SAFE_FILE = /^[a-zA-Z0-9_.-]+$/;
const BACKENDS = new Set(["triton", "cutile", "tilelang", "torch"]);
const AGENT_SOURCE_FILES = new Set([
  "impl_triton.py",
  "impl_cutile.py",
  "impl_tilelang.py",
  "impl_torch.py",
  "config.yaml",
]);

type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
type ToolResult = Record<string, unknown>;

/** [value] or [value, unit], as written by scripts/export_profiles.py. */
type MetricValue = [number | string] | [number | string, string];

interface ProfileKernel {
  name: string;
  launches: number;
  block_size?: string;
  grid_size?: string;
  compute_capability?: string;
  duration_ns_mean?: number;
  /** Nsight Compute's own curated, human-named metrics: section -> metric -> value */
  sections?: Record<string, Record<string, MetricValue>>;
  metrics: Record<string, MetricValue>;
  findings: Array<Record<string, unknown>>;
  hot_instructions: Array<Record<string, unknown>>;
}

interface ProfileReport {
  platform: string;
  op: string;
  backend: string;
  dtype: string;
  profiler: string;
  capture?: unknown;
  distinct_kernels: number;
  kernels: ProfileKernel[];
}

interface ProfileEntry {
  platform: string;
  op: string;
  backend: string;
  dtype: string;
  profiler: string;
  file: string;
  bytes: number;
  distinct_kernels: number;
  kernels: Array<{ name: string; metrics: number; findings: number; hot_instructions: number }>;
}

function asObj(input: unknown): Record<string, unknown> {
  return input && typeof input === "object" && !Array.isArray(input)
    ? (input as Record<string, unknown>)
    : {};
}

function asStr(v: unknown): string | undefined {
  return typeof v === "string" && v.trim() ? v.trim() : undefined;
}

function asInt(v: unknown, fallback: number, min: number, max: number): number {
  const n = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.trunc(n)));
}

function badInput(message: string): ToolResult {
  return { ok: false, error: message };
}

function preview(value: unknown, max = MAX_TOOL_RESULT_CHARS): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return text.length > max ? `${text.slice(0, max)}\n...[truncated ${text.length - max} chars]` : text;
}

function normDataPath(rel: string): string | null {
  const clean = rel.replace(/^\/+/, "");
  const full = path.resolve(DATA, clean);
  if (full !== DATA && !full.startsWith(DATA + path.sep)) return null;
  return full;
}

// ---------- exported profiling reports (data/profiles_json) ----------

let manifestCache: ProfileEntry[] | null = null;
async function profileManifest(): Promise<ProfileEntry[]> {
  if (!manifestCache) {
    try {
      const raw = await fs.readFile(path.join(DATA, "profiles_json", "manifest.json"), "utf8");
      manifestCache = (JSON.parse(raw) as { reports: ProfileEntry[] }).reports;
    } catch {
      manifestCache = [];
    }
  }
  return manifestCache;
}

let descriptionCache: Record<string, Record<string, string>> | null = null;
async function metricDescriptions(platform: string): Promise<Record<string, string>> {
  if (!descriptionCache) {
    try {
      descriptionCache = JSON.parse(
        await fs.readFile(path.join(DATA, "profiles_json", "metric_descriptions.json"), "utf8")
      );
    } catch {
      descriptionCache = {};
    }
  }
  return descriptionCache?.[platform] ?? {};
}

const reportCache = new Map<string, ProfileReport>();
async function loadProfile(entry: ProfileEntry): Promise<ProfileReport | null> {
  const hit = reportCache.get(entry.file);
  if (hit) return hit;
  const full = normDataPath(`profiles_json/${entry.file}`);
  if (!full) return null;
  try {
    const report = JSON.parse(gunzipSync(await fs.readFile(full)).toString("utf8")) as ProfileReport;
    if (reportCache.size >= 24) reportCache.delete(reportCache.keys().next().value as string);
    reportCache.set(entry.file, report);
    return report;
  } catch {
    return null;
  }
}

function publicProfile(r: ProfileEntry): ToolResult {
  return {
    platform: r.platform,
    op: r.op,
    backend: r.backend,
    dtype: r.dtype,
    profiler: r.profiler,
    kernels: r.kernels.map((k) => k.name),
    distinct_kernels: r.distinct_kernels,
  };
}

function profileFilter(args: Record<string, unknown>) {
  const platform = asStr(args.platform)?.toLowerCase();
  const op = asStr(args.op);
  const backend = asStr(args.backend)?.toLowerCase();
  const dtype = asStr(args.dtype)?.toLowerCase();
  return (r: ProfileEntry) =>
    (!platform || r.platform.toLowerCase() === platform) &&
    (!op || r.op === op) &&
    (!backend || r.backend === backend) &&
    (!dtype || r.dtype.toLowerCase() === dtype);
}

/** Resolve exactly one report from platform/op/backend/dtype, or explain what is ambiguous. */
async function oneProfile(
  args: Record<string, unknown>
): Promise<{ entry: ProfileEntry; report: ProfileReport } | { error: ToolResult }> {
  for (const key of ["platform", "op", "backend", "dtype"]) {
    if (!asStr(args[key])) {
      return { error: badInput("platform, op, backend and dtype are all required; call find_profiles to see what exists") };
    }
  }
  const matches = (await profileManifest()).filter(profileFilter(args));
  if (matches.length !== 1) {
    const near = (await profileManifest())
      .filter(profileFilter({ platform: args.platform, op: args.op }))
      .map((r) => `${r.backend}_${r.dtype}`);
    return { error: { ok: false, error: "no such profiling report", available_for_this_platform_and_op: near } };
  }
  const report = await loadProfile(matches[0]);
  if (!report) return { error: { ok: false, error: "report file could not be read", file: matches[0].file } };
  return { entry: matches[0], report };
}

function pickKernels(report: ProfileReport, kernel: string | undefined): ProfileKernel[] {
  if (!kernel) return report.kernels;
  const q = kernel.toLowerCase();
  return report.kernels.filter((k) => k.name.toLowerCase().includes(q));
}

const unpack = (m: MetricValue) => (m.length > 1 ? { value: m[0], unit: m[1] } : { value: m[0] });

/** The handful of metrics worth reading first, per profiler. */
const NCU_KEY_METRICS = [
  "gpu__time_duration.sum",
  "sm__throughput.avg.pct_of_peak_sustained_elapsed",
  "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
  "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
  "dram__throughput.avg.pct_of_peak_sustained_elapsed",
  "dram__bytes_read.sum",
  "dram__bytes_write.sum",
  "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
  "lts__throughput.avg.pct_of_peak_sustained_elapsed",
  "l1tex__t_sector_hit_rate.pct",
  "lts__t_sector_hit_rate.pct",
  "sm__warps_active.avg.pct_of_peak_sustained_active",
  "sm__warps_active.avg.per_cycle_active",
  "smsp__warps_eligible.avg.per_cycle_active",
  "smsp__issue_active.avg.per_cycle_active",
  "smsp__issue_inst0.avg.pct_of_peak_sustained_active",
  "sm__inst_executed.sum",
  "smsp__inst_executed.avg.per_cycle_active",
  "smsp__thread_inst_executed_per_inst_executed.ratio",
  "smsp__cycles_active.avg",
  "sm__inst_executed_pipe_tensor.sum",
  "launch__registers_per_thread",
  "launch__thread_count",
  "launch__occupancy_limit_registers",
  "launch__occupancy_limit_shared_mem",
  "launch__occupancy_limit_warps",
  "launch__occupancy_limit_blocks",
  "launch__waves_per_multiprocessor",
  "launch__shared_mem_per_block_static",
  "launch__shared_mem_per_block_dynamic",
  "sm__maximum_warps_per_active_cycle_pct",
  "sm__warps_active.avg.pct_of_peak_sustained_active",
];

function keyMetrics(report: ProfileReport, k: ProfileKernel): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (k.sections && Object.keys(k.sections).length) {
    for (const [section, metrics] of Object.entries(k.sections)) {
      out[section] = Object.fromEntries(
        Object.entries(metrics).map(([name, m]) => [name, m.length > 1 ? `${m[0]} ${m[1]}` : m[0]])
      );
    }
    return out;
  }
  if (report.profiler === "rocprof-compute") {
    for (const [name, m] of Object.entries(k.metrics)) {
      if (name.startsWith("System Speed-of-Light /") && m[0] !== 0) out[name] = unpack(m);
    }
  } else {
    for (const name of NCU_KEY_METRICS) if (k.metrics[name]) out[name] = unpack(k.metrics[name]);
  }
  return out;
}

async function findProfiles(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const op = asStr(args.op);
  if (op && !SAFE_OP.test(op)) return badInput("bad op");
  const all = await profileManifest();
  if (all.length === 0) return { ok: false, error: "no profiling reports are deployed" };
  const rows = all.filter(profileFilter(args));
  if (rows.length > 150) {
    const counts: Record<string, Record<string, number>> = {};
    for (const r of rows) {
      counts[r.platform] ??= {};
      counts[r.platform][r.backend] = (counts[r.platform][r.backend] ?? 0) + 1;
    }
    return {
      ok: true,
      total: rows.length,
      reports_per_platform_and_backend: counts,
      note: "Too many to list. Every report is one (platform, op, backend, dtype). Filter by op, or by platform and backend.",
    };
  }
  return { ok: true, total: rows.length, reports: rows.map(publicProfile) };
}

async function getProfileSummary(input: unknown): Promise<ToolResult> {
  const found = await oneProfile(asObj(input));
  if ("error" in found) return found.error;
  const { report } = found;
  const reduced =
    report.profiler === "Nsight Compute" &&
    !report.kernels.some((k) => k.sections?.["GPU Speed Of Light Throughput"]);
  return {
    ok: true,
    ...(reduced
      ? {
          reduced_capture:
            "This report was captured with a small custom metric set. It has no section summaries, no profiler findings and no instruction samples, so it cannot be compared metric-for-metric with the other backends' full captures.",
        }
      : {}),
    platform: report.platform,
    op: report.op,
    backend: report.backend,
    dtype: report.dtype,
    profiler: report.profiler,
    capture: report.capture ?? null,
    distinct_kernels: report.distinct_kernels,
    kernels: report.kernels.map((k) => ({
      name: k.name,
      launches: k.launches,
      block_size: k.block_size,
      grid_size: k.grid_size,
      duration_ns_mean: k.duration_ns_mean,
      metrics_available: Object.keys(allMetrics(k)).length,
      key_metrics: keyMetrics(report, k),
      profiler_findings: k.findings,
      hottest_instructions: k.hot_instructions.slice(0, 8),
    })),
    note: "When launches is above 1 the kernel ran several times and the metrics are from its first launch. key_metrics are the profiler's own named summary metrics, grouped by section; query them by name as 'Section / Metric'. Thousands of raw hardware counters are also available through query_profile_metrics, and the full instruction list through get_profile_hotspots.",
  };
}

/** Every metric of a kernel under one namespace: "Section / Metric" names first, then raw counters. */
function allMetrics(k: ProfileKernel): Record<string, MetricValue> {
  const named: Record<string, MetricValue> = {};
  for (const [section, metrics] of Object.entries(k.sections ?? {})) {
    for (const [name, m] of Object.entries(metrics)) named[`${section} / ${name}`] = m;
  }
  return { ...named, ...k.metrics };
}

function matchMetrics(k: ProfileKernel, names: string[], pattern: string | undefined, limit: number) {
  const out: Array<[string, MetricValue]> = [];
  let matched = 0;
  const metrics = allMetrics(k);
  if (names.length) {
    for (const n of names) if (metrics[n]) out.push([n, metrics[n]]);
    matched = out.length;
  } else if (pattern) {
    const terms = pattern.toLowerCase().split(/\s+/).filter(Boolean);
    for (const [name, m] of Object.entries(metrics)) {
      const hay = name.toLowerCase();
      if (m[0] === "no data" || !terms.every((t) => hay.includes(t))) continue;
      matched++;
      if (out.length < limit) out.push([name, m]);
    }
  }
  return { out, matched };
}

async function queryProfileMetrics(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const found = await oneProfile(args);
  if ("error" in found) return found.error;
  const names = Array.isArray(args.names) ? args.names.filter((n): n is string => typeof n === "string") : [];
  const pattern = asStr(args.pattern);
  const limit = asInt(args.limit, 60, 1, 200);
  if (!names.length && !pattern) return badInput("pass names (exact) or pattern (space-separated substrings, all must match)");
  const kernels = pickKernels(found.report, asStr(args.kernel));
  if (kernels.length === 0) {
    return { ok: false, error: "no kernel matches", kernels: found.report.kernels.map((k) => k.name) };
  }
  const describe = await metricDescriptions(found.report.platform);
  return {
    ok: true,
    platform: found.report.platform,
    op: found.report.op,
    backend: found.report.backend,
    dtype: found.report.dtype,
    kernels: kernels.map((k) => {
      const { out, matched } = matchMetrics(k, names, pattern, limit);
      return {
        name: k.name,
        matched,
        truncated: matched > out.length,
        metrics: out.map(([name, m]) => {
          const description = describe[name.replace(/ \[[^\]]*\]$/, "")];
          return { name, ...unpack(m), ...(description ? { description } : {}) };
        }),
      };
    }),
    ...(names.length
      ? { not_found: names.filter((n) => !kernels.some((k) => allMetrics(k)[n])) }
      : {}),
  };
}

async function getProfileHotspots(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const found = await oneProfile(args);
  if ("error" in found) return found.error;
  const limit = asInt(args.limit, 25, 1, 40);
  const kernels = pickKernels(found.report, asStr(args.kernel));
  const any = kernels.some((k) => k.hot_instructions.length);
  return {
    ok: true,
    platform: found.report.platform,
    op: found.report.op,
    backend: found.report.backend,
    dtype: found.report.dtype,
    kernels: kernels.map((k) => ({ name: k.name, instructions: k.hot_instructions.slice(0, limit) })),
    ...(any
      ? { note: "pct_of_samples is the share of this kernel's profiler samples that landed on the instruction." }
      : { note: "This report carries no per-instruction samples, so hotspots cannot be given for it." }),
  };
}

async function compareProfileMetrics(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const op = asStr(args.op);
  const dtype = asStr(args.dtype);
  if (!op || !SAFE_OP.test(op)) return badInput("op is required");
  const names = Array.isArray(args.names) ? args.names.filter((n): n is string => typeof n === "string") : [];
  const pattern = asStr(args.pattern);
  if (!names.length && !pattern) return badInput("pass names (exact) or pattern");
  const limit = asInt(args.limit, 25, 1, 60);
  const entries = (await profileManifest()).filter(
    profileFilter({ op, dtype, platform: args.platform, backend: args.backend })
  );
  if (entries.length === 0) return { ok: false, error: "no profiling reports match" };
  if (entries.length > 24) return badInput("too many reports; narrow with dtype, platform or backend");
  const rows = [];
  for (const entry of entries) {
    const report = await loadProfile(entry);
    if (!report) continue;
    for (const k of pickKernels(report, asStr(args.kernel))) {
      const { out } = matchMetrics(k, names, pattern, limit);
      rows.push({
        platform: report.platform,
        backend: report.backend,
        dtype: report.dtype,
        kernel: k.name,
        metrics: Object.fromEntries(out.map(([name, m]) => [name, unpack(m)])),
      });
    }
  }
  return {
    ok: true,
    op,
    rows,
    note: "Metric names differ between profilers and between GPU generations, so a name can be absent on one platform. Compare backends within a platform before comparing across platforms.",
  };
}

async function listOperators(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const tier = asStr(args.tier);
  const target = args.target === true;
  const sort = asStr(args.sort) ?? "autotune";
  const all = await getOperators();
  let rows = all;
  if (tier) rows = rows.filter((o) => o.tier === tier);
  if (target) rows = rows.filter((o) => o.target);
  rows = [...rows].sort((a, b) => {
    if (sort === "op") return a.op.localeCompare(b.op);
    const mode = (sort === "default" ? "default" : "autotune") as Mode;
    return (b[mode]?.tl_over_triton ?? -Infinity) - (a[mode]?.tl_over_triton ?? -Infinity);
  });
  return {
    ok: true,
    operators: rows.map((o) => ({
      op: o.op,
      tier: o.tier,
      target: o.target,
      default: o.default,
      autotune: o.autotune,
      autotune_excluded_reason: o.autotune_excluded_reason,
    })),
    pooled: { default: poolMode(all, "default"), autotune: poolMode(all, "autotune") },
  };
}

const SPEEDUP_UNITS =
  "speedup over PyTorch on the same hardware and inputs (PyTorch ms / backend ms), geometric mean over cases; above 1 is faster than PyTorch, below 1 is slower";

/** One operator on every measured platform, with the profiling reports published for it. */
async function operatorAcrossPlatforms(op: string) {
  const { platforms, reports_dataset } = await getPlatforms();
  return platforms.map((p) => {
    const row = p.operators.find((o) => o.op === op);
    return {
      platform: p.id,
      name: p.name,
      backends: p.backends,
      default: row?.default ?? null,
      autotune: row?.autotune ?? null,
      profiler: p.profiler,
      published_reports: p.reports[op] ?? [],
      reports_url: `https://huggingface.co/datasets/${reports_dataset}/tree/main/${p.hf_path}/${op}`,
    };
  });
}

async function getPlatformResults(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const id = asStr(args.platform);
  const op = asStr(args.op);
  if (op && !SAFE_OP.test(op)) return badInput("bad op");
  const { platforms } = await getPlatforms();
  if (op) return { ok: true, units: SPEEDUP_UNITS, op, platforms: await operatorAcrossPlatforms(op) };
  const summary = platforms.map((p) => ({
    platform: p.id,
    name: p.name,
    note: p.note,
    backends: p.backends,
    profiler: p.profiler,
    operators: p.operators.length,
    published_reports: Object.values(p.reports).reduce((n, r) => n + r.length, 0),
    pooled: { default: poolPlatform(p, "default"), autotune: poolPlatform(p, "autotune") },
  }));
  if (!id) return { ok: true, units: SPEEDUP_UNITS, platforms: summary };
  const p = platforms.find((x) => x.id.toLowerCase() === id.toLowerCase());
  if (!p) return { ok: false, error: "unknown platform", platforms: platforms.map((x) => x.id) };
  return {
    ok: true,
    units: SPEEDUP_UNITS,
    ...summary.find((s) => s.platform === p.id),
    per_operator: p.operators,
  };
}

async function getOperatorTool(input: unknown): Promise<ToolResult> {
  const op = asStr(asObj(input).op);
  if (!op || !SAFE_OP.test(op)) return badInput("op is required");
  const [operator, src, manifest] = await Promise.all([
    getOperator(op),
    getSourceIndex(),
    profileManifest(),
  ]);
  if (!operator) return { ok: false, error: "unknown operator", op };
  return {
    ok: true,
    units: SPEEDUP_UNITS,
    operator_b200: operator,
    platforms: await operatorAcrossPlatforms(op),
    source_files: src[op]
      ? Object.keys(src[op]).filter((file) => AGENT_SOURCE_FILES.has(file)).sort()
      : [],
    profiles: manifest.filter((r) => r.op === op).map(publicProfile),
  };
}


async function listFiles(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const op = asStr(args.op);
  if (op && !SAFE_OP.test(op)) return badInput("bad op");

  const files: Array<{ path: string; bytes?: number }> = [];
  {
    const src = await getSourceIndex();
    for (const [srcOp, byFile] of Object.entries(src)) {
      if (op && srcOp !== op) continue;
      for (const [file, bytes] of Object.entries(byFile)) {
        if (!AGENT_SOURCE_FILES.has(file)) continue;
        files.push({ path: `source/${srcOp}/${file}`, bytes });
      }
    }
  }
  return { ok: true, files: files.sort((a, b) => a.path.localeCompare(b.path)) };
}

async function readFileTool(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const rel = asStr(args.path);
  const offset = asInt(args.offset, 0, 0, Number.MAX_SAFE_INTEGER);
  const maxChars = asInt(args.max_chars, DEFAULT_FILE_CHARS, 1_000, MAX_FILE_CHARS);
  if (!rel) return badInput("path is required; call list_files first");
  const allowed = ["source/", "operators.json", "operators.csv"];
  if (!allowed.some((p) => rel === p || rel.startsWith(p))) return badInput("path is outside the agent corpus");
  if (rel.startsWith("source/")) {
    const sourceMatch = /^source\/([a-z0-9_]+)\/([^/]+)$/.exec(rel);
    if (!sourceMatch || !AGENT_SOURCE_FILES.has(sourceMatch[2])) {
      return badInput("only impl_<backend>.py and config.yaml are exposed under source/");
    }
  }
  const full = normDataPath(rel);
  if (!full) return badInput("bad path");
  try {
    const text = await fs.readFile(full, "utf8");
    return {
      ok: true,
      path: rel,
      chars: text.length,
      offset,
      max_chars: maxChars,
      next_offset: offset + maxChars < text.length ? offset + maxChars : null,
      truncated: offset + maxChars < text.length,
      content: text.slice(offset, offset + maxChars),
    };
  } catch {
    return { ok: false, error: "file not found", path: rel };
  }
}

async function readSource(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const op = asStr(args.op);
  const backend = asStr(args.backend);
  const file = asStr(args.file) ?? (backend ? `impl_${backend}.py` : "");
  const offset = asInt(args.offset, 0, 0, Number.MAX_SAFE_INTEGER);
  const maxChars = asInt(args.max_chars, DEFAULT_FILE_CHARS, 1_000, MAX_FILE_CHARS);
  if (!op || !SAFE_OP.test(op) || !SAFE_FILE.test(file)) {
    return badInput("op is required, plus backend or file");
  }
  if (!AGENT_SOURCE_FILES.has(file)) {
    return badInput("file must be impl_triton.py, impl_cutile.py, impl_tilelang.py, impl_torch.py or config.yaml");
  }
  const body = await getSourceFile(op, file);
  if (body === null) return { ok: false, error: "source file not found", op, file };
  return {
    ok: true,
    op,
    file,
    chars: body.length,
    offset,
    max_chars: maxChars,
    next_offset: offset + maxChars < body.length ? offset + maxChars : null,
    content: body.slice(offset, offset + maxChars),
    truncated: offset + maxChars < body.length,
  };
}

async function grepFiles(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const query = asStr(args.query);
  const op = asStr(args.op);
  const backend = asStr(args.backend);
  const maxResults = asInt(args.max_results, 40, 1, 200);
  const contextLines = asInt(args.context_lines, 2, 0, 8);
  if (!query) return badInput("query is required");
  if (op && !SAFE_OP.test(op)) return badInput("bad op");
  if (backend && !BACKENDS.has(backend)) return badInput("bad backend");

  const listed = await listFiles({ op });
  const files = Array.isArray(listed.files) ? listed.files as Array<{ path: string }> : [];
  const hits = [];
  const q = query.toLowerCase();
  for (const f of files) {
    if (backend && f.path.includes("/impl_") && !f.path.endsWith(`impl_${backend}.py`)) continue;
    const full = normDataPath(f.path);
    if (!full) continue;
    let text: string;
    try {
      text = await fs.readFile(full, "utf8");
    } catch {
      continue;
    }
    const lines = text.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      if (!lines[i].toLowerCase().includes(q)) continue;
      const start = Math.max(0, i - contextLines);
      const end = Math.min(lines.length, i + contextLines + 1);
      hits.push({
        path: f.path,
        line: i + 1,
        match: lines[i],
        context: lines.slice(start, end).map((line, j) => ({ line: start + j + 1, text: line })),
      });
      if (hits.length >= maxResults) return { ok: true, query, hits, truncated: true };
    }
  }
  return { ok: true, query, hits, truncated: false };
}

export const AGENT_TOOLS = [
  {
    name: "list_operators",
    description: "List TileBench++ operators and pooled board stats for the NVIDIA B200 only. For other hardware or cross-platform questions use get_platform_results.",
    input_schema: {
      type: "object",
      properties: {
        tier: { type: "string", enum: ["both", "partial", "excluded"] },
        target: { type: "boolean" },
        sort: { type: "string", enum: ["autotune", "default", "op"] },
      },
    },
  },
  {
    name: "get_platform_results",
    description: "Benchmark results per hardware platform (NVIDIA B200, NVIDIA GH200, AMD MI300X). No arguments: every platform's backends and pooled speedups. platform: that platform's per-operator table. op: one operator across all platforms.",
    input_schema: {
      type: "object",
      properties: {
        platform: { type: "string", enum: ["B200", "GH200", "MI300X"] },
        op: { type: "string" },
      },
    },
  },
  {
    name: "get_operator",
    description: "Get one operator's benchmark stats on every platform, its implementation source files for each backend, and the profiling reports that can be queried for it.",
    input_schema: { type: "object", properties: { op: { type: "string" } }, required: ["op"] },
  },
  {
    name: "list_files",
    description: "List the kernel source files in the corpus: impl_triton.py, impl_cutile.py, impl_tilelang.py, impl_torch.py and config.yaml per operator.",
    input_schema: { type: "object", properties: { op: { type: "string" } } },
  },
  {
    name: "read_file",
    description: "Read a corpus file by path (source/<op>/<file>, operators.csv, operators.json). Use offset/next_offset to page through large files.",
    input_schema: {
      type: "object",
      properties: {
        path: { type: "string" },
        offset: { type: "integer" },
        max_chars: { type: "integer" },
      },
      required: ["path"],
    },
  },
  {
    name: "read_source",
    description: "Read one backend's implementation of an operator (or its config.yaml). Pass backend, or file. Use offset/next_offset to page through large files.",
    input_schema: {
      type: "object",
      properties: {
        op: { type: "string" },
        backend: { type: "string", enum: ["triton", "cutile", "tilelang", "torch"] },
        file: { type: "string", enum: ["impl_triton.py", "impl_cutile.py", "impl_tilelang.py", "impl_torch.py", "config.yaml"] },
        offset: { type: "integer" },
        max_chars: { type: "integer" },
      },
      required: ["op"],
    },
  },
  {
    name: "grep_files",
    description: "Search kernel source across operators and backends for a substring.",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string" },
        op: { type: "string" },
        backend: { type: "string", enum: ["triton", "cutile", "tilelang", "torch"] },
        max_results: { type: "integer" },
        context_lines: { type: "integer" },
      },
      required: ["query"],
    },
  },
  {
    name: "find_profiles",
    description: "List the profiling reports that can be queried. One report per (platform, op, backend, dtype): Nsight Compute on B200 and GH200 for Triton, cuTile and TileLang; rocprof-compute on MI300X for Triton. Filter by any of the four.",
    input_schema: {
      type: "object",
      properties: {
        platform: { type: "string", enum: ["B200", "GH200", "MI300X"] },
        op: { type: "string" },
        backend: { type: "string", enum: ["triton", "cutile", "tilelang"] },
        dtype: { type: "string" },
      },
    },
  },
  {
    name: "get_profile_summary",
    description: "Start here for one profiling report: its kernels, launch configuration, a curated set of key metrics (duration, compute and memory throughput, occupancy, cache hit rates), the profiler's own bottleneck findings, and the hottest instructions.",
    input_schema: {
      type: "object",
      properties: {
        platform: { type: "string", enum: ["B200", "GH200", "MI300X"] },
        op: { type: "string" },
        backend: { type: "string" },
        dtype: { type: "string" },
      },
      required: ["platform", "op", "backend", "dtype"],
    },
  },
  {
    name: "query_profile_metrics",
    description: "Read measured metric values from one profiling report. pattern is space-separated substrings that must all appear in the metric name (e.g. 'Memory Throughput', 'Occupancy', 'dram throughput', 'Speed-of-Light'); names fetches exact metric names. Named summary metrics appear as 'Section / Metric'; raw hardware counters keep their profiler names. Each report holds thousands of metrics.",
    input_schema: {
      type: "object",
      properties: {
        platform: { type: "string", enum: ["B200", "GH200", "MI300X"] },
        op: { type: "string" },
        backend: { type: "string" },
        dtype: { type: "string" },
        kernel: { type: "string", description: "substring of a kernel name; default all kernels in the report" },
        pattern: { type: "string" },
        names: { type: "array", items: { type: "string" } },
        limit: { type: "integer" },
      },
      required: ["platform", "op", "backend", "dtype"],
    },
  },
  {
    name: "compare_profile_metrics",
    description: "Put the same metrics side by side across the reports for one operator, e.g. all three backends on one platform. Narrow with dtype, platform or backend. Pass exact names or a pattern.",
    input_schema: {
      type: "object",
      properties: {
        op: { type: "string" },
        dtype: { type: "string" },
        platform: { type: "string", enum: ["B200", "GH200", "MI300X"] },
        backend: { type: "string" },
        kernel: { type: "string" },
        pattern: { type: "string" },
        names: { type: "array", items: { type: "string" } },
        limit: { type: "integer" },
      },
      required: ["op"],
    },
  },
  {
    name: "get_profile_hotspots",
    description: "The instructions where one kernel spent its profiler samples: SASS assembly on B200 and GH200 (no source lines), ISA with the Python source line on MI300X. A few reports carry no samples.",
    input_schema: {
      type: "object",
      properties: {
        platform: { type: "string", enum: ["B200", "GH200", "MI300X"] },
        op: { type: "string" },
        backend: { type: "string" },
        dtype: { type: "string" },
        kernel: { type: "string" },
        limit: { type: "integer" },
      },
      required: ["platform", "op", "backend", "dtype"],
    },
  },
] as const;

const HANDLERS: Record<string, (input: unknown) => Promise<ToolResult>> = {
  list_operators: listOperators,
  get_platform_results: getPlatformResults,
  get_operator: getOperatorTool,
  list_files: listFiles,
  read_file: readFileTool,
  read_source: readSource,
  grep_files: grepFiles,
  find_profiles: findProfiles,
  get_profile_summary: getProfileSummary,
  query_profile_metrics: queryProfileMetrics,
  compare_profile_metrics: compareProfileMetrics,
  get_profile_hotspots: getProfileHotspots,
};

export async function runAgentTool(name: string, input: unknown): Promise<ToolResult> {
  const handler = HANDLERS[name];
  if (!handler) return { ok: false, error: `unknown tool ${name}` };
  try {
    const result = await handler(input);
    return JSON.parse(preview(result)) as ToolResult;
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "tool failed" };
  }
}

export function toolPreview(result: unknown): Json {
  const text = preview(result, 800);
  try {
    return JSON.parse(text) as Json;
  } catch {
    return text;
  }
}
