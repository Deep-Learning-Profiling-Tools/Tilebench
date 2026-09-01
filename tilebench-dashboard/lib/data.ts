import "server-only";
import fs from "node:fs/promises";
import path from "node:path";
import type { Operator, ProfileEntry, Note } from "./types";

const DATA = path.join(process.cwd(), "data");

async function readJson<T>(rel: string): Promise<T> {
  return JSON.parse(await fs.readFile(path.join(DATA, rel), "utf8")) as T;
}

let opsCache: Operator[] | null = null;
export async function getOperators(): Promise<Operator[]> {
  if (!opsCache) opsCache = await readJson<Operator[]>("operators.json");
  return opsCache;
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
