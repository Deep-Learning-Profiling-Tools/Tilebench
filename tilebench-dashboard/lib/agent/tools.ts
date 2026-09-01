import "server-only";
import fs from "node:fs/promises";
import path from "node:path";
import { getOperator, getOperators, getSourceFile, getSourceIndex } from "@/lib/data";
import { poolMode } from "@/lib/metrics";
import type { Mode } from "@/lib/types";

const DATA = path.join(process.cwd(), "data");
const DEFAULT_FILE_CHARS = 80_000;
const MAX_FILE_CHARS = 180_000;
const MAX_TOOL_RESULT_CHARS = 180_000;
const SAFE_OP = /^[a-z0-9_]+$/;
const SAFE_FILE = /^[a-zA-Z0-9_.-]+$/;
const BACKENDS = new Set(["triton", "cutile", "tilelang", "torch"]);
const AGENT_SOURCE_FILES = new Set(["impl_tilelang.py"]);

type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
type ToolResult = Record<string, unknown>;

interface NcuMetric {
  name?: string;
  value?: unknown;
  unit?: string;
  description?: string;
}

interface NcuAction {
  action_index?: number;
  name?: string;
  metrics?: NcuMetric[];
  rule_results?: unknown[];
  source_files?: Record<string, string>;
  cli_details_page?: string;
  cli_source_page?: string;
  cli_rule_results?: unknown[];
  source_markers?: unknown[];
  timed_warp_samples?: unknown[];
  addresses?: unknown[];
}

interface NcuReportJson {
  source_report?: string;
  source_report_sha256?: string;
  op?: string;
  backend?: string | null;
  dtype?: string | null;
  num_actions?: number;
  actions?: NcuAction[];
}

interface NcuManifest {
  reports?: Array<{
    op?: string;
    backend?: string | null;
    dtype?: string | null;
    report?: string;
    json?: string;
    sha256?: string;
    bytes?: number;
    num_actions?: number;
  }>;
}

interface TileLangCompilerManifest {
  repo?: string;
  remote?: string;
  branch?: string;
  commit?: string;
  dirty?: boolean;
  source_roots?: string[];
  root_files?: string[];
  total_files?: number;
  total_bytes?: number;
  files?: Array<{
    path?: string;
    bytes?: number;
    sha256?: string;
  }>;
}

function publicNcuReport(r: NonNullable<NcuManifest["reports"]>[number]): ToolResult {
  return {
    op: r.op,
    backend: r.backend,
    dtype: r.dtype,
    json: r.json,
    sha256: r.sha256,
    bytes: r.bytes,
    num_actions: r.num_actions,
  };
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

function scrubNcuProvenance(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(scrubNcuProvenance);
  if (!value || typeof value !== "object") return value;
  const out: Record<string, unknown> = {};
  for (const [key, raw] of Object.entries(value)) {
    if ((key === "source_report" || key === "report") && typeof raw === "string" && /\.ncu-repz?$/.test(raw)) {
      continue;
    }
    out[key] = scrubNcuProvenance(raw);
  }
  return out;
}

async function readAgentTextFile(rel: string, full: string): Promise<string> {
  const text = await fs.readFile(full, "utf8");
  if (rel.startsWith("ncu_json/") && rel.endsWith(".json")) {
    try {
      return JSON.stringify(scrubNcuProvenance(JSON.parse(text)), null, 2);
    } catch {
      return text;
    }
  }
  return text;
}

function normDataPath(rel: string): string | null {
  const clean = rel.replace(/^\/+/, "");
  const full = path.resolve(DATA, clean);
  if (full !== DATA && !full.startsWith(DATA + path.sep)) return null;
  return full;
}

async function readJsonFile<T>(rel: string): Promise<T | null> {
  const full = normDataPath(rel);
  if (!full) return null;
  try {
    return JSON.parse(await fs.readFile(full, "utf8")) as T;
  } catch {
    return null;
  }
}

async function pathExists(rel: string): Promise<boolean> {
  const full = normDataPath(rel);
  if (!full) return false;
  try {
    await fs.access(full);
    return true;
  } catch {
    return false;
  }
}

async function walkFiles(rootRel: string): Promise<string[]> {
  const root = normDataPath(rootRel);
  if (!root) return [];
  const out: string[] = [];
  async function walk(dir: string) {
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) await walk(full);
      else if (entry.isFile()) out.push(path.relative(DATA, full).split(path.sep).join("/"));
    }
  }
  await walk(root);
  return out.sort();
}

async function ncuManifest(): Promise<NcuManifest | null> {
  return readJsonFile<NcuManifest>("ncu_json/manifest.json");
}

async function tileLangCompilerManifest(): Promise<TileLangCompilerManifest | null> {
  return readJsonFile<TileLangCompilerManifest>("tilelang_compiler/manifest.json");
}

async function tileLangCompilerFiles(): Promise<NonNullable<TileLangCompilerManifest["files"]>> {
  const manifest = await tileLangCompilerManifest();
  return manifest?.files ?? [];
}

async function isTileLangCompilerFile(rel: string): Promise<boolean> {
  if (rel === "tilelang_compiler/manifest.json") return true;
  if (!rel.startsWith("tilelang_compiler/")) return false;
  const file = rel.slice("tilelang_compiler/".length);
  return (await tileLangCompilerFiles()).some((entry) => entry.path === file);
}

async function readNcuReport(jsonPath: string): Promise<NcuReportJson | null> {
  const rel = jsonPath.startsWith("ncu_json/") ? jsonPath : `ncu_json/${jsonPath}`;
  return readJsonFile<NcuReportJson>(rel);
}

function selectActions(report: NcuReportJson, idx: number | undefined): NcuAction[] {
  const actions = report.actions ?? [];
  if (idx == null) return actions;
  return actions.filter((a, i) => (a.action_index ?? i) === idx);
}

function metricMatches(metric: NcuMetric, pattern: string | undefined): boolean {
  if (!pattern) return true;
  const hay = `${metric.name ?? ""}\n${metric.description ?? ""}`.toLowerCase();
  return hay.includes(pattern.toLowerCase());
}

function exactMetric(action: NcuAction, name: string): NcuMetric | undefined {
  return (action.metrics ?? []).find((m) => m.name === name);
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

async function getOperatorTool(input: unknown): Promise<ToolResult> {
  const op = asStr(asObj(input).op);
  if (!op || !SAFE_OP.test(op)) return badInput("op is required");
  const [operator, src, manifest] = await Promise.all([
    getOperator(op),
    getSourceIndex(),
    ncuManifest(),
  ]);
  if (!operator) return { ok: false, error: "unknown operator", op };
  return {
    ok: true,
    operator,
    source_files: src[op]
      ? Object.keys(src[op]).filter((file) => AGENT_SOURCE_FILES.has(file)).sort()
      : [],
    ncu_json_reports: (manifest?.reports ?? [])
      .filter((r) => r.op === op)
      .map(publicNcuReport),
  };
}

async function listFiles(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const op = asStr(args.op);
  const kind = asStr(args.kind) ?? "all";
  if (op && !SAFE_OP.test(op)) return badInput("bad op");

  const files: Array<{ path: string; bytes?: number }> = [];
  if (kind === "all" || kind === "source") {
    const src = await getSourceIndex();
    for (const [srcOp, byFile] of Object.entries(src)) {
      if (op && srcOp !== op) continue;
      for (const [file, bytes] of Object.entries(byFile)) {
        if (!AGENT_SOURCE_FILES.has(file)) continue;
        files.push({ path: `source/${srcOp}/${file}`, bytes });
      }
    }
  }
  if ((kind === "all" || kind === "ncu_json") && await pathExists("ncu_json")) {
    for (const file of await walkFiles("ncu_json")) files.push({ path: file });
  }
  if ((kind === "all" || kind === "ncu_source") && await pathExists("ncu_source")) {
    for (const file of await walkFiles("ncu_source")) files.push({ path: file });
  }
  if ((kind === "all" || kind === "tilelang_compiler") && await pathExists("tilelang_compiler")) {
    const manifest = await tileLangCompilerManifest();
    if (manifest) {
      files.push({ path: "tilelang_compiler/manifest.json" });
      for (const file of manifest.files ?? []) {
        if (file.path) files.push({ path: `tilelang_compiler/${file.path}`, bytes: file.bytes });
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
  const allowed = ["source/", "ncu_json/", "ncu_source/", "tilelang_compiler/", "operators.json"];
  if (!allowed.some((p) => rel === p || rel.startsWith(p))) return badInput("path is outside the agent corpus");
  if (rel.startsWith("source/")) {
    const sourceMatch = /^source\/([a-z0-9_]+)\/([^/]+)$/.exec(rel);
    if (!sourceMatch || !AGENT_SOURCE_FILES.has(sourceMatch[2])) {
      return badInput("only TileLang implementation source is exposed to the agent");
    }
  }
  if (rel.startsWith("tilelang_compiler/") && !(await isTileLangCompilerFile(rel))) {
    return badInput("path is outside the TileLang compiler snapshot");
  }
  const full = normDataPath(rel);
  if (!full) return badInput("bad path");
  try {
    const text = await readAgentTextFile(rel, full);
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
  const file = asStr(args.file) ?? "impl_tilelang.py";
  const offset = asInt(args.offset, 0, 0, Number.MAX_SAFE_INTEGER);
  const maxChars = asInt(args.max_chars, DEFAULT_FILE_CHARS, 1_000, MAX_FILE_CHARS);
  if (!op || !SAFE_OP.test(op) || !SAFE_FILE.test(file)) return badInput("op and file are required");
  if (!AGENT_SOURCE_FILES.has(file)) {
    return badInput("only impl_tilelang.py is exposed through read_source");
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
  const kind = asStr(args.kind) ?? "source";
  const maxResults = asInt(args.max_results, 40, 1, 200);
  const contextLines = asInt(args.context_lines, 2, 0, 8);
  if (!query) return badInput("query is required");
  if (op && !SAFE_OP.test(op)) return badInput("bad op");
  if (backend && !BACKENDS.has(backend)) return badInput("bad backend");

  const listed = await listFiles({ op, kind });
  const files = Array.isArray(listed.files) ? listed.files as Array<{ path: string }> : [];
  const hits = [];
  const q = query.toLowerCase();
  for (const f of files) {
    if (backend && f.path.includes("/impl_") && !f.path.endsWith(`impl_${backend}.py`)) continue;
    const full = normDataPath(f.path);
    if (!full) continue;
    let text: string;
    try {
      text = await readAgentTextFile(f.path, full);
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

async function listTileLangCompilerFiles(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const pathPrefix = asStr(args.path_prefix)?.replace(/^\/+/, "");
  const pattern = asStr(args.pattern)?.toLowerCase();
  const limit = asInt(args.limit, 80, 1, 500);
  const manifest = await tileLangCompilerManifest();
  if (!manifest) return { ok: false, error: "TileLang compiler snapshot is not deployed" };

  let files = manifest.files ?? [];
  if (pathPrefix) files = files.filter((f) => f.path?.startsWith(pathPrefix));
  if (pattern) files = files.filter((f) => f.path?.toLowerCase().includes(pattern));
  return {
    ok: true,
    snapshot: {
      repo: manifest.repo,
      branch: manifest.branch,
      commit: manifest.commit,
      dirty: manifest.dirty,
      source_roots: manifest.source_roots,
      root_files: manifest.root_files,
      total_files: manifest.total_files,
      total_bytes: manifest.total_bytes,
    },
    files: files.slice(0, limit),
    total_matches: files.length,
    truncated: files.length > limit,
  };
}

async function readTileLangCompilerFile(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const file = asStr(args.path)?.replace(/^\/+/, "");
  const offset = asInt(args.offset, 0, 0, Number.MAX_SAFE_INTEGER);
  const maxChars = asInt(args.max_chars, DEFAULT_FILE_CHARS, 1_000, MAX_FILE_CHARS);
  if (!file) return badInput("path is required; call list_tilelang_compiler_files first");
  const rel = `tilelang_compiler/${file}`;
  if (!(await isTileLangCompilerFile(rel))) return badInput("path is outside the TileLang compiler snapshot");
  const full = normDataPath(rel);
  if (!full) return badInput("bad path");
  try {
    const text = await fs.readFile(full, "utf8");
    return {
      ok: true,
      path: file,
      chars: text.length,
      offset,
      max_chars: maxChars,
      next_offset: offset + maxChars < text.length ? offset + maxChars : null,
      truncated: offset + maxChars < text.length,
      content: text.slice(offset, offset + maxChars),
    };
  } catch {
    return { ok: false, error: "compiler file not found", path: file };
  }
}

async function grepTileLangCompiler(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const query = asStr(args.query);
  const pathPrefix = asStr(args.path_prefix)?.replace(/^\/+/, "");
  const maxResults = asInt(args.max_results, 60, 1, 200);
  const contextLines = asInt(args.context_lines, 2, 0, 8);
  if (!query) return badInput("query is required");
  const manifest = await tileLangCompilerManifest();
  if (!manifest) return { ok: false, error: "TileLang compiler snapshot is not deployed" };

  const files = (manifest.files ?? []).filter((f) => f.path && (!pathPrefix || f.path.startsWith(pathPrefix)));
  const hits = [];
  const q = query.toLowerCase();
  for (const f of files) {
    const rel = `tilelang_compiler/${f.path}`;
    const full = normDataPath(rel);
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
      if (hits.length >= maxResults) return { ok: true, query, path_prefix: pathPrefix, hits, truncated: true };
    }
  }
  return { ok: true, query, path_prefix: pathPrefix, hits, truncated: false };
}

async function findNcuReports(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const manifest = await ncuManifest();
  if (!manifest) {
    return {
      ok: false,
      error: "NCU JSON export is not deployed yet",
      expected: "Run `python tilebench_run/ncu_export_json.py` in Tilebench and copy tilebench_run/ncu_json to dashboard data/ncu_json.",
    };
  }
  const op = asStr(args.op);
  const backend = asStr(args.backend);
  const dtype = asStr(args.dtype);
  const reports = (manifest.reports ?? []).filter((r) =>
    (!op || r.op === op) &&
    (!backend || r.backend === backend) &&
    (!dtype || r.dtype === dtype)
  );
  return { ok: true, reports: reports.map(publicNcuReport) };
}

async function listNcuMetrics(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const jsonPath = asStr(args.report_json);
  if (!jsonPath) return badInput("report_json is required; call find_ncu_reports first");
  const actionIndex = args.action_index == null ? undefined : asInt(args.action_index, 0, 0, 10_000);
  const pattern = asStr(args.pattern);
  const limit = asInt(args.limit, 80, 1, 500);
  const report = await readNcuReport(jsonPath);
  if (!report) return { ok: false, error: "NCU report JSON not found", report_json: jsonPath };

  const metrics = [];
  for (const action of selectActions(report, actionIndex)) {
    for (const metric of action.metrics ?? []) {
      if (!metricMatches(metric, pattern)) continue;
      metrics.push({
        action_index: action.action_index,
        action_name: action.name,
        name: metric.name,
        value: metric.value,
        unit: metric.unit,
        description: metric.description,
      });
      if (metrics.length >= limit) return { ok: true, report_json: jsonPath, metrics, truncated: true };
    }
  }
  return { ok: true, report_json: jsonPath, metrics, truncated: false };
}

async function getNcuMetric(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const jsonPath = asStr(args.report_json);
  const name = asStr(args.metric);
  if (!jsonPath || !name) return badInput("report_json and metric are required");
  const actionIndex = args.action_index == null ? undefined : asInt(args.action_index, 0, 0, 10_000);
  const report = await readNcuReport(jsonPath);
  if (!report) return { ok: false, error: "NCU report JSON not found", report_json: jsonPath };
  const matches = [];
  for (const action of selectActions(report, actionIndex)) {
    const metric = exactMetric(action, name);
    if (metric) {
      matches.push({
        action_index: action.action_index,
        action_name: action.name,
        metric,
      });
    }
  }
  return {
    ok: matches.length > 0,
    report_json: jsonPath,
    metric: name,
    matches,
    error: matches.length ? undefined : "metric not found in selected action(s)",
  };
}

async function getNcuRules(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const jsonPath = asStr(args.report_json);
  if (!jsonPath) return badInput("report_json is required");
  const actionIndex = args.action_index == null ? undefined : asInt(args.action_index, 0, 0, 10_000);
  const offset = asInt(args.offset, 0, 0, Number.MAX_SAFE_INTEGER);
  const maxChars = asInt(args.max_chars, DEFAULT_FILE_CHARS, 1_000, MAX_FILE_CHARS);
  const report = await readNcuReport(jsonPath);
  if (!report) return { ok: false, error: "NCU report JSON not found", report_json: jsonPath };
  return {
    ok: true,
    report_json: jsonPath,
    rule_results: selectActions(report, actionIndex).map((a) => ({
      action_index: a.action_index,
      action_name: a.name,
      rule_results: a.rule_results ?? [],
    })),
  };
}

async function getNcuSource(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const jsonPath = asStr(args.report_json);
  if (!jsonPath) return badInput("report_json is required");
  const actionIndex = args.action_index == null ? undefined : asInt(args.action_index, 0, 0, 10_000);
  const offset = asInt(args.offset, 0, 0, Number.MAX_SAFE_INTEGER);
  const maxChars = asInt(args.max_chars, DEFAULT_FILE_CHARS, 1_000, MAX_FILE_CHARS);
  const report = await readNcuReport(jsonPath);
  if (!report) return { ok: false, error: "NCU report JSON not found", report_json: jsonPath };
  return {
    ok: true,
    report_json: jsonPath,
    actions: selectActions(report, actionIndex).map((a) => ({
      action_index: a.action_index,
      action_name: a.name,
      cli_details_page: a.cli_details_page,
      cli_source_page: a.cli_source_page,
      source_files: Object.entries(a.source_files ?? {}).map(([file, content]) => ({
        file,
        imported: content.length > 0,
        chars: content.length,
        offset,
        max_chars: maxChars,
        next_offset: offset + maxChars < content.length ? offset + maxChars : null,
        content: content.slice(offset, offset + maxChars),
        truncated: offset + maxChars < content.length,
      })),
      cli_rule_results: a.cli_rule_results ?? [],
      source_markers: a.source_markers ?? [],
      timed_warp_samples: a.timed_warp_samples ?? [],
      addresses: a.addresses ?? [],
    })),
  };
}

async function compareNcuMetrics(input: unknown): Promise<ToolResult> {
  const args = asObj(input);
  const op = asStr(args.op);
  const dtype = asStr(args.dtype);
  const metrics = Array.isArray(args.metrics) ? args.metrics.filter((m): m is string => typeof m === "string") : [];
  if (!op || !dtype || metrics.length === 0) return badInput("op, dtype, and metrics[] are required");
  const manifest = await ncuManifest();
  if (!manifest) return { ok: false, error: "NCU JSON export is not deployed yet" };
  const reports = (manifest.reports ?? []).filter((r) => r.op === op && r.dtype === dtype && r.json);
  const rows = [];
  for (const r of reports) {
    const report = await readNcuReport(r.json ?? "");
    if (!report) continue;
    for (const action of report.actions ?? []) {
      const values: Record<string, unknown> = {};
      for (const metric of metrics) values[metric] = exactMetric(action, metric)?.value ?? null;
      rows.push({ backend: r.backend, dtype: r.dtype, report_json: r.json, action_index: action.action_index, action_name: action.name, values });
    }
  }
  return { ok: true, op, dtype, metrics, rows };
}

export const AGENT_TOOLS = [
  {
    name: "list_operators",
    description: "List TileBench operators and pooled board stats. Use before broad comparisons.",
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
    name: "get_operator",
    description: "Get one operator's benchmark stats, exposed TileLang source files, and deployed NCU JSON reports. Does not expose legacy .ncu-rep metadata.",
    input_schema: { type: "object", properties: { op: { type: "string" } }, required: ["op"] },
  },
  {
    name: "list_files",
    description: "List files in the deployed agent corpus: TileLang kernel source, TileLang compiler source snapshot, exported ncu_json, and exported ncu_source.",
    input_schema: {
      type: "object",
      properties: {
        op: { type: "string" },
        kind: { type: "string", enum: ["all", "source", "tilelang_compiler", "ncu_json", "ncu_source"] },
      },
    },
  },
  {
    name: "read_file",
    description: "Read a file returned by list_files. Restricted to the deployed data corpus. Use offset/next_offset to page through large files.",
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
    description: "Read impl_tilelang.py for an operator. Use offset/next_offset to page through large files.",
    input_schema: {
      type: "object",
      properties: {
        op: { type: "string" },
        file: { type: "string" },
        offset: { type: "integer" },
        max_chars: { type: "integer" },
      },
      required: ["op", "file"],
    },
  },
  {
    name: "grep_files",
    description: "Search deployed TileLang kernel source, TileLang compiler source, and exported NCU text/JSON files in-process. This is the Vercel-safe grep.",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string" },
        op: { type: "string" },
        backend: { type: "string", enum: ["triton", "cutile", "tilelang", "torch"] },
        kind: { type: "string", enum: ["source", "tilelang_compiler", "ncu_json", "ncu_source", "all"] },
        max_results: { type: "integer" },
        context_lines: { type: "integer" },
      },
      required: ["query"],
    },
  },
  {
    name: "list_tilelang_compiler_files",
    description: "List files in the pinned TileLang compiler source snapshot. Use before reading compiler internals.",
    input_schema: {
      type: "object",
      properties: {
        path_prefix: { type: "string" },
        pattern: { type: "string" },
        limit: { type: "integer" },
      },
    },
  },
  {
    name: "read_tilelang_compiler_file",
    description: "Read one file from the pinned TileLang compiler source snapshot. Use offset/next_offset to page through large files.",
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
    name: "grep_tilelang_compiler",
    description: "Search the pinned TileLang compiler source snapshot. Use for lowering, scheduling, codegen, autotune, and runtime internals questions.",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string" },
        path_prefix: { type: "string" },
        max_results: { type: "integer" },
        context_lines: { type: "integer" },
      },
      required: ["query"],
    },
  },
  {
    name: "find_ncu_reports",
    description: "Find structured NCU JSON reports by op/backend/dtype. Reports appear after offline export.",
    input_schema: {
      type: "object",
      properties: {
        op: { type: "string" },
        backend: { type: "string", enum: ["triton", "cutile", "tilelang", "torch"] },
        dtype: { type: "string" },
      },
    },
  },
  {
    name: "list_ncu_metrics",
    description: "List exact metric names/values/units/descriptions in an NCU JSON report, optionally filtered by substring.",
    input_schema: {
      type: "object",
      properties: {
        report_json: { type: "string" },
        action_index: { type: "integer" },
        pattern: { type: "string" },
        limit: { type: "integer" },
      },
      required: ["report_json"],
    },
  },
  {
    name: "get_ncu_metric",
    description: "Fetch one exact NCU metric by full metric name from a report/action.",
    input_schema: {
      type: "object",
      properties: {
        report_json: { type: "string" },
        action_index: { type: "integer" },
        metric: { type: "string" },
      },
      required: ["report_json", "metric"],
    },
  },
  {
    name: "get_ncu_rule_results",
    description: "Get Nsight Compute rule-engine results for a report/action.",
    input_schema: {
      type: "object",
      properties: { report_json: { type: "string" }, action_index: { type: "integer" } },
      required: ["report_json"],
    },
  },
  {
    name: "get_ncu_source",
    description: "Get NCU source/SASS/PTX/source-marker/timed-warp-sample records for a report/action. Source files are pageable with offset/next_offset.",
    input_schema: {
      type: "object",
      properties: {
        report_json: { type: "string" },
        action_index: { type: "integer" },
        offset: { type: "integer" },
        max_chars: { type: "integer" },
      },
      required: ["report_json"],
    },
  },
  {
    name: "compare_ncu_metrics",
    description: "Compare exact NCU metric values across available backend reports for one operator/dtype.",
    input_schema: {
      type: "object",
      properties: {
        op: { type: "string" },
        dtype: { type: "string" },
        metrics: { type: "array", items: { type: "string" } },
      },
      required: ["op", "dtype", "metrics"],
    },
  },
] as const;

const HANDLERS: Record<string, (input: unknown) => Promise<ToolResult>> = {
  list_operators: listOperators,
  get_operator: getOperatorTool,
  list_files: listFiles,
  read_file: readFileTool,
  read_source: readSource,
  grep_files: grepFiles,
  list_tilelang_compiler_files: listTileLangCompilerFiles,
  read_tilelang_compiler_file: readTileLangCompilerFile,
  grep_tilelang_compiler: grepTileLangCompiler,
  find_ncu_reports: findNcuReports,
  list_ncu_metrics: listNcuMetrics,
  get_ncu_metric: getNcuMetric,
  get_ncu_rule_results: getNcuRules,
  get_ncu_source: getNcuSource,
  compare_ncu_metrics: compareNcuMetrics,
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
