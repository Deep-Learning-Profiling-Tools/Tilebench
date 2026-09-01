import { ok } from "@/lib/http";
import { getOperators, getProfiles, getSourceIndex } from "@/lib/data";
import { notesStore } from "@/lib/store";
import fs from "node:fs/promises";
import path from "node:path";

async function ncuJsonCount(): Promise<number> {
  try {
    const raw = await fs.readFile(path.join(process.cwd(), "data", "ncu_json", "manifest.json"), "utf8");
    const manifest = JSON.parse(raw) as { reports?: unknown[] };
    return Array.isArray(manifest.reports) ? manifest.reports.length : 0;
  } catch {
    return 0;
  }
}

async function tileLangCompilerCount(): Promise<number> {
  try {
    const raw = await fs.readFile(path.join(process.cwd(), "data", "tilelang_compiler", "manifest.json"), "utf8");
    const manifest = JSON.parse(raw) as { total_files?: unknown; files?: unknown[] };
    if (typeof manifest.total_files === "number") return manifest.total_files;
    return Array.isArray(manifest.files) ? manifest.files.length : 0;
  } catch {
    return 0;
  }
}

export async function GET() {
  const [ops, profiles, src, ncuReports, tilelangCompilerFiles] = await Promise.all([
    getOperators(),
    getProfiles(),
    getSourceIndex(),
    ncuJsonCount(),
    tileLangCompilerCount(),
  ]);
  const store = notesStore();
  return ok({
    status: "ok",
    operators: ops.length,
    cases: ops.reduce((a, o) => a + (o.default?.cases ?? 0) + (o.autotune?.cases ?? 0), 0),
    profiled: profiles.filter((p) => p.has_report).length,
    ncu_json_reports: ncuReports,
    tilelang_compiler_files: tilelangCompilerFiles,
    source_operators: Object.keys(src).length,
    notes_store: { kind: store.kind, durable: store.durable },
    agent: {
      configured: Boolean(process.env.OPENROUTER_API_KEY),
      accepts_client_key: true,
      provider: "openrouter",
      model: process.env.TILEBENCH_AGENT_MODEL ?? "openai/gpt-5.6-luna",
    },
  });
}
