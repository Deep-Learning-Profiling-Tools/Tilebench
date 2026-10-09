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
  const any = (v: unknown) => {
    const text = asStr(v)?.toLowerCase();
    return text === "all" || text === "any" || text === "*" ? undefined : text;
  };
  const platform = any(args.platform);
  const op = asStr(args.op);
  const backend = any(args.backend);
  const dtype = any(args.dtype);
  return (r: { platform: string; op: string; backend: string; dtype: string }) =>
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
  const given = Array.isArray(args.names) ? args.names.filter((n): n is string => typeof n === "string" && n.trim() !== "") : [];
  const pattern = asStr(args.pattern);
  const defaulted = !given.length && !pattern;
  const names = defaulted ? [...NCU_KEY_METRICS] : given;
  const limit = asInt(args.limit, defaulted ? 60 : 25, 1, 60);
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
    ...(defaulted ? { metrics_shown: "the curated key metrics, because no names or pattern were given" } : {}),
    note: "Metric names differ between profilers and between GPU generations, so a name can be absent on one platform. Compare backends within a platform before comparing across platforms.",
  };
}

// ---------- compiler intermediate code (data/ir, written by scripts/export_ir.py) ----------

interface IrFile {
  kind: string;
  what: string;
  file: string;
  lines: number;
  chars: number;
}

interface IrEntry {
  platform: string;
  op: string;
  backend: string;
  dtype: string;
  kernel: string;
  shape?: Record<string, unknown>;
  config?: Record<string, unknown>;
  files: IrFile[];
  versions?: Record<string, unknown>;
  source_revision?: string | null;
  note?: string;
  verified?: string;
  profile_check?: string;
  profile_check_ok?: boolean | null;
  variant?: number;
  launch_variants?: number;
  config_note?: string;
  args?: string[];
  tensors?: string[];
  kw?: string[];
}

let irCache: IrEntry[] | null = null;
async function irManifest(): Promise<IrEntry[]> {
  if (!irCache) {
    try {
      const raw = await fs.readFile(path.join(DATA, "ir", "manifest.json"), "utf8");
      irCache = (JSON.parse(raw) as { entries: IrEntry[] }).entries;
    } catch {
      irCache = [];
    }
  }
  return irCache;
}

function publicIr(e: IrEntry): ToolResult {
  return {
    platform: e.platform,
    op: e.op,
    backend: e.backend,
    dtype: e.dtype,
    kernel: e.kernel,
    shape: e.shape,
    config: e.config,
    kinds: e.files.map((f) => ({ kind: f.kind, what: f.what, lines: f.lines })),
    ...(e.launch_variants && e.launch_variants > 1
      ? {
          variant: e.variant ?? 1,
          launch_variants: e.launch_variants,
          launch_arguments: e.args ?? [...(e.tensors ?? []), ...(e.kw ?? [])],
        }
      : {}),
    ...(e.config_note ? { config_note: e.config_note } : {}),
    ...(e.note ? { note: e.note } : {}),
    ...(e.verified ? { verified: e.verified } : {}),
    ...(e.profile_check ? { check_against_profile: e.profile_check } : {}),
    ...(e.profile_check_ok === false
      ? { warning: "This code did not match the profiled kernel. Do not use it to explain the profile." }
      : {}),
  };
}

async function findIr(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const op = asStr(args.op);
  if (op && !SAFE_OP.test(op)) return badInput("bad op");
  const all = await irManifest();
  if (all.length === 0) return { ok: false, error: "no intermediate code is deployed" };
  const rows = all.filter(profileFilter(args));
  return {
    ok: true,
    total: rows.length,
    operators_with_intermediate_code: [...new Set(all.map((e) => e.op))].sort(),
    entries: rows.slice(0, 120).map(publicIr),
    note:
      "Each entry is the code one compiler produced for one kernel at the profiled shape and that backend's autotuned configuration. An operator, backend or dtype that is not listed has no intermediate code here.",
  };
}

async function getIr(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  for (const key of ["platform", "op", "backend", "dtype"]) {
    if (!asStr(args[key])) return badInput("platform, op, backend and dtype are all required; call find_ir to see what exists");
  }
  const all = await irManifest();
  const matches = all.filter(profileFilter(args));
  if (matches.length === 0) {
    const near = all.filter(profileFilter({ op: args.op })).map((e) => `${e.platform}/${e.backend}_${e.dtype}`);
    return { ok: false, error: "no intermediate code for this report", available_for_this_op: near };
  }
  const wantedKernel = asStr(args.kernel)?.toLowerCase();
  const byKernel = wantedKernel ? matches.filter((e) => e.kernel.toLowerCase().includes(wantedKernel)) : matches;
  const exact = wantedKernel ? byKernel.filter((e) => e.kernel.toLowerCase() === wantedKernel) : [];
  const named = exact.length ? exact : byKernel;
  const wantedVariant = args.variant === undefined ? undefined : asInt(args.variant, 1, 1, 64);
  const narrowed = wantedVariant ? named.filter((e) => (e.variant ?? 1) === wantedVariant) : named;
  if (narrowed.length !== 1) {
    return {
      ok: true,
      kernels: (narrowed.length ? narrowed : matches).map(publicIr),
      note: "This report has intermediate code for more than one kernel, or for one kernel launched with different arguments. Pass kernel, and variant where launch_variants is above 1, to pick one.",
    };
  }
  const entry = narrowed[0];
  const kind = asStr(args.kind);
  const file = kind ? entry.files.find((f) => f.kind === kind) : entry.files.length === 1 ? entry.files[0] : undefined;
  if (!file) {
    return { ok: true, ...publicIr(entry), note: "Pass kind to read one of these." };
  }
  const full = normDataPath(`ir/${file.file}`);
  if (!full) return badInput("bad path");
  let text: string;
  try {
    text = await fs.readFile(full, "utf8");
  } catch {
    return { ok: false, error: "intermediate code file could not be read", file: file.file };
  }
  const lines = text.split("\n");
  const pattern = asStr(args.pattern)?.toLowerCase();
  if (pattern) {
    const hits = lines
      .map((line, i) => ({ line: i + 1, text: line.length > 400 ? `${line.slice(0, 400)}...` : line }))
      .filter((row) => row.text.toLowerCase().includes(pattern));
    return { ok: true, ...publicIr(entry), kind: file.kind, pattern, total_matches: hits.length, matches: hits.slice(0, 120) };
  }
  const start = asInt(args.start_line, 1, 1, Math.max(1, lines.length));
  const count = asInt(args.max_lines, 400, 1, 1500);
  const slice = lines.slice(start - 1, start - 1 + count);
  const end = start - 1 + slice.length;
  return {
    ok: true,
    ...publicIr(entry),
    kind: file.kind,
    what: file.what,
    versions: entry.versions,
    total_lines: lines.length,
    start_line: start,
    end_line: end,
    ...(end < lines.length ? { next_start_line: end + 1 } : {}),
    text: slice.join("\n"),
  };
}

// ---------- per-instruction listings (data/profile_listings, written by scripts/export_listings.py) ----------

interface ListingRow {
  offset: string;
  executed: number;
  samples: number;
  stalls: Record<string, number>;
  source: string | null;
  sass: string;
}

interface ListingKernel {
  name: string;
  samples: number;
  static_instructions?: number;
  stall_samples?: Record<string, number>;
  samples_by_source_line?: Array<Record<string, unknown>>;
  opcode_executions?: Record<string, number>;
  load_order?: Record<string, unknown>;
  instructions: ListingRow[];
  note?: string;
}

interface ListingReport {
  platform: string;
  op: string;
  backend: string;
  dtype: string;
  kernels: ListingKernel[];
}

interface ListingEntry {
  platform: string;
  op: string;
  backend: string;
  dtype: string;
  file: string;
}

let listingManifestCache: ListingEntry[] | null = null;
async function listingManifest(): Promise<ListingEntry[]> {
  if (!listingManifestCache) {
    try {
      const raw = await fs.readFile(path.join(DATA, "profile_listings", "manifest.json"), "utf8");
      listingManifestCache = (JSON.parse(raw) as { reports: ListingEntry[] }).reports;
    } catch {
      listingManifestCache = [];
    }
  }
  return listingManifestCache;
}

const listingCache = new Map<string, ListingReport>();
async function loadListing(entry: ListingEntry): Promise<ListingReport | null> {
  const hit = listingCache.get(entry.file);
  if (hit) return hit;
  const full = normDataPath(`profile_listings/${entry.file}`);
  if (!full) return null;
  try {
    const report = JSON.parse(gunzipSync(await fs.readFile(full)).toString("utf8")) as ListingReport;
    if (listingCache.size >= 12) listingCache.delete(listingCache.keys().next().value as string);
    listingCache.set(entry.file, report);
    return report;
  } catch {
    return null;
  }
}

function listingRow(r: ListingRow, total: number): ToolResult {
  return {
    offset: r.offset,
    sass: r.sass,
    executed: r.executed,
    samples: r.samples,
    ...(total ? { pct_of_samples: Math.round((1000 * r.samples) / total) / 10 } : {}),
    ...(Object.keys(r.stalls).length ? { stalls: r.stalls } : {}),
    ...(r.source ? { source: r.source } : {}),
  };
}

async function getProfileListing(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  for (const key of ["platform", "op", "backend", "dtype"]) {
    if (!asStr(args[key])) return badInput("platform, op, backend and dtype are all required");
  }
  const all = await listingManifest();
  const matches = all.filter(profileFilter(args));
  if (matches.length === 0) {
    return {
      ok: false,
      error: "no per-instruction listing for this report; use get_profile_hotspots for its hottest instructions",
      listings_for_this_op: all.filter(profileFilter({ op: args.op })).map((e) => `${e.platform}/${e.backend}_${e.dtype}`),
    };
  }
  const report = await loadListing(matches[0]);
  if (!report) return { ok: false, error: "listing file could not be read", file: matches[0].file };
  const wanted = asStr(args.kernel)?.toLowerCase();
  const kernels = wanted ? report.kernels.filter((k) => k.name.toLowerCase().includes(wanted)) : report.kernels;
  const head = { ok: true, platform: report.platform, op: report.op, backend: report.backend, dtype: report.dtype };
  const pattern = asStr(args.pattern)?.toLowerCase();
  const start = asStr(args.start_offset);
  if (!pattern && !start) {
    const shown = kernels.slice(0, 12);
    return {
      ...head,
      kernels_in_report: report.kernels.length,
      kernels: shown.map((k) => ({
        name: k.name,
        samples: k.samples,
        static_instructions: k.static_instructions ?? k.instructions.length,
        ...(k.note ? { note: k.note } : {}),
        stall_samples: k.stall_samples,
        samples_by_source_line: (k.samples_by_source_line ?? []).slice(0, 12),
        opcode_executions: k.opcode_executions,
        load_order: k.load_order,
        hottest: [...k.instructions].sort((a, b) => b.samples - a.samples).slice(0, 12).map((r) => listingRow(r, k.samples)),
      })),
      note:
        "Samples are taken at a fixed rate, so sample counts compare as time between kernels and backends. A stall is charged to the instruction that waits for a value, not the one that produced it: for long_scoreboard walk back to the LDG that loaded it. load_order lists global loads and load waits of the hot loop in address order; loads issued before the first wait overlap, a load issued after a wait is another memory round trip. Pass start_offset or pattern to read instructions in address order.",
    };
  }
  const limit = asInt(args.max_instructions, 120, 1, 400);
  return {
    ...head,
    kernels: kernels.slice(0, 4).map((k) => {
      let rows = k.instructions;
      if (pattern) rows = rows.filter((r) => r.sass.toLowerCase().includes(pattern) || (r.source ?? "").toLowerCase().includes(pattern));
      if (start) {
        const at = Number.parseInt(start, 16);
        rows = rows.filter((r) => Number.parseInt(r.offset, 16) >= at);
      }
      const slice = rows.slice(0, limit);
      return {
        name: k.name,
        samples: k.samples,
        matching_instructions: rows.length,
        ...(rows.length > slice.length ? { next_start_offset: rows[slice.length].offset } : {}),
        instructions: slice.map((r) => listingRow(r, k.samples)),
      };
    }),
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
  const [operator, src, manifest, ir, listings] = await Promise.all([
    getOperator(op),
    getSourceIndex(),
    profileManifest(),
    irManifest(),
    listingManifest(),
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
    intermediate_code: [...new Set(ir.filter((e) => e.op === op).map((e) => `${e.platform}/${e.backend}_${e.dtype}`))],
    instruction_listings: listings.filter((e) => e.op === op).map((e) => `${e.platform}/${e.backend}_${e.dtype}`),
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
  {
    name: "get_profile_listing",
    description: "Per-instruction evidence for one Nsight Compute report (B200, GH200), when a listing was exported for it. Without start_offset or pattern: per kernel, the stall reasons behind its samples, samples by source line, dynamic opcode totals, the order of global loads and load waits in the hot loop, and the hottest instructions with their stall reasons and source lines. With start_offset (hex) or pattern (substring of the SASS or source line): the instructions in address order with execution count, samples, stall reasons and source line. Use this to locate a mechanism; get_profile_hotspots is the fallback when no listing exists.",
    input_schema: {
      type: "object",
      properties: {
        platform: { type: "string", enum: ["B200", "GH200"] },
        op: { type: "string" },
        backend: { type: "string" },
        dtype: { type: "string" },
        kernel: { type: "string", description: "substring of a kernel name; default all kernels in the report" },
        start_offset: { type: "string", description: "hex offset to start reading from, e.g. 0x0300" },
        pattern: { type: "string", description: "e.g. LDG, BAR, STS, or a source file:line" },
        max_instructions: { type: "integer" },
      },
      required: ["platform", "op", "backend", "dtype"],
    },
  },
  {
    name: "find_ir",
    description: "List the compiler intermediate code that can be read: for one kernel at the profiled shape and that backend's autotuned configuration, what each compiler produced. Triton: ttir (before the GPU passes), ttgir (after them: layouts, shared memory, tensor path, pipelining) and ptx. TileLang: the generated CUDA (cu), and TIR before (tir) and after (lowered.tir) its passes. cuTile: the front-end Tile IR (tileir); its back end is closed. Filter by platform, op, backend or dtype. Coverage is partial: an operator that is not listed has none.",
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
    name: "get_ir",
    description: "Read compiler intermediate code for one (platform, op, backend, dtype). Without kind: what is available. With kind (ttir, ttgir, ptx, cu, tir, lowered.tir, tileir): the text, paged with start_line/max_lines, or only the lines containing pattern. This is what the compiler emitted for the profiled configuration; cite the lines you rely on.",
    input_schema: {
      type: "object",
      properties: {
        platform: { type: "string", enum: ["B200", "GH200", "MI300X"] },
        op: { type: "string" },
        backend: { type: "string", enum: ["triton", "cutile", "tilelang"] },
        dtype: { type: "string" },
        kernel: { type: "string", description: "substring of a kernel name, for operators that launch several kernels" },
        variant: { type: "integer", description: "which launch of the kernel, when it is launched with different arguments (launch_variants above 1)" },
        kind: { type: "string", enum: ["ttir", "ttgir", "ptx", "cu", "tir", "lowered.tir", "tileir"] },
        pattern: { type: "string" },
        start_line: { type: "integer" },
        max_lines: { type: "integer" },
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
  get_profile_listing: getProfileListing,
  find_ir: findIr,
  get_ir: getIr,
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
