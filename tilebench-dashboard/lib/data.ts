import "server-only";
import fs from "node:fs/promises";
import path from "node:path";
import type { Operator, ProfileEntry, Note, PlatformData } from "./types";

const DATA = path.join(process.cwd(), "data");

async function readJson<T>(rel: string): Promise<T> {
  return JSON.parse(await fs.readFile(path.join(DATA, rel), "utf8")) as T;
}

let opsCache: Operator[] | null = null;
export async function getOperators(): Promise<Operator[]> {
  if (!opsCache) {
    const text = await fs.readFile(path.join(DATA, "operators.csv"), "utf8");
    const [header, ...lines] = text.trim().split(/\r?\n/);
    const keys = header.split(",");
    const value = (row: string[], key: string) => row[keys.indexOf(key)] ?? "";
    const mode = (row: string[], name: "default" | "autotune") => {
      const cases = value(row, `${name}_cases`);
      if (!cases) return null;
      return {
        triton: Number(value(row, `${name}_triton`)),
        cutile: Number(value(row, `${name}_cutile`)),
        tilelang: Number(value(row, `${name}_tilelang`)),
        tl_over_triton: Number(value(row, `${name}_tl_over_triton`)),
        tl_over_cutile: Number(value(row, `${name}_tl_over_cutile`)),
        cases: Number(cases),
      };
    };
    opsCache = lines.map((line) => {
      const row = line.split(",");
      return {
        op: value(row, "op"),
        tier: value(row, "tier") as Operator["tier"],
        target: value(row, "target") === "true",
        default: mode(row, "default"),
        autotune: mode(row, "autotune"),
        autotune_excluded_reason: value(row, "autotune_excluded_reason") || null,
      };
    });
  }
  return opsCache;
}

/** Every measured platform: board aggregates plus profiling-report coverage. */
let platCache: PlatformData | null = null;
export async function getPlatforms(): Promise<PlatformData> {
  if (!platCache) platCache = await readJson<PlatformData>("platforms.json");
  return platCache;
}

export async function getOperator(op: string): Promise<Operator | null> {
  return (await getOperators()).find((o) => o.op === op) ?? null;
}

let profCache: ProfileEntry[] | null = null;
export async function getProfiles(): Promise<ProfileEntry[]> {
  if (!profCache) profCache = await readJson<ProfileEntry[]>("profiles/manifest.json");
  return profCache;
}

export async function getProfile(op: string): Promise<ProfileEntry | null> {
  return (await getProfiles()).find((p) => p.op === op) ?? null;
}

/** The NCU write-up markdown for one operator, if it was profiled. */
export async function getProfileReport(op: string): Promise<string | null> {
  if (!/^[a-z0-9_]+$/.test(op)) return null;
  try {
    return await fs.readFile(path.join(DATA, "profiles", `${op}.md`), "utf8");
  } catch {
    return null;
  }
}

export async function getSourceIndex(): Promise<Record<string, Record<string, number>>> {
  return readJson("source/index.json");
}

/** One implementation file. Both segments are validated against the index. */
export async function getSourceFile(op: string, file: string): Promise<string | null> {
  const idx = await getSourceIndex();
  if (!idx[op] || !(file in idx[op])) return null;
  return fs.readFile(path.join(DATA, "source", op, file), "utf8");
}

export async function getSeedNotes(): Promise<Record<string, Note[]>> {
  return readJson("notes.seed.json");
}
