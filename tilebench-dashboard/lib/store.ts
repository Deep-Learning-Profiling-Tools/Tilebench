import "server-only";
import type { Note } from "./types";
import { getSeedNotes } from "./data";

/**
 * Notes storage.
 *
 * Two backends behind one interface so the deployment can start with no
 * database and gain one without touching a route handler:
 *
 *   MemoryStore  - default. Per-instance and NOT durable: serverless
 *                  instances are recycled, so writes vanish. Fine for
 *                  reading the seeded corpus and for local development.
 *   KvStore      - used automatically when KV_REST_API_URL and
 *                  KV_REST_API_TOKEN are present (Vercel KV / Upstash Redis).
 *                  Durable and shared across instances.
 */
export interface NotesStore {
  readonly durable: boolean;
  readonly kind: string;
  all(): Promise<Record<string, Note[]>>;
  get(op: string): Promise<Note[]>;
  add(op: string, note: Note): Promise<Note[]>;
  remove(op: string, ts: number): Promise<Note[]>;
}

class MemoryStore implements NotesStore {
  readonly durable = false;
  readonly kind = "memory";
  private data: Record<string, Note[]> | null = null;

  private async load() {
    if (!this.data) this.data = structuredClone(await getSeedNotes());
    return this.data;
  }
  async all() {
    return structuredClone(await this.load());
  }
  async get(op: string) {
    return (await this.load())[op] ?? [];
  }
  async add(op: string, note: Note) {
    const d = await this.load();
    (d[op] ??= []).push(note);
    d[op].sort((a, b) => a.ts - b.ts);
    return [...d[op]];
  }
  async remove(op: string, ts: number) {
    const d = await this.load();
    d[op] = (d[op] ?? []).filter((n) => n.ts !== ts);
    return [...d[op]];
  }
}

class KvStore implements NotesStore {
  readonly durable = true;
  readonly kind = "kv";
  private key = "tilebench:notes";
  constructor(private url: string, private token: string) {}

  private async cmd<T>(...args: (string | number)[]): Promise<T> {
    const r = await fetch(this.url, {
      method: "POST",
      headers: { Authorization: `Bearer ${this.token}`, "Content-Type": "application/json" },
      body: JSON.stringify(args),
      cache: "no-store",
    });
    if (!r.ok) throw new Error(`kv ${r.status}: ${await r.text()}`);
    return (await r.json()).result as T;
  }

  private async read(): Promise<Record<string, Note[]>> {
    const raw = await this.cmd<string | null>("GET", this.key);
    if (raw) return JSON.parse(raw);
    const seed = await getSeedNotes();
    await this.cmd("SET", this.key, JSON.stringify(seed));
    return seed;
  }
  private async write(d: Record<string, Note[]>) {
    await this.cmd("SET", this.key, JSON.stringify(d));
  }
  async all() {
    return this.read();
  }
  async get(op: string) {
    return (await this.read())[op] ?? [];
  }
  async add(op: string, note: Note) {
    const d = await this.read();
    (d[op] ??= []).push(note);
    d[op].sort((a, b) => a.ts - b.ts);
    await this.write(d);
    return d[op];
  }
  async remove(op: string, ts: number) {
    const d = await this.read();
    d[op] = (d[op] ?? []).filter((n) => n.ts !== ts);
    await this.write(d);
    return d[op];
  }
}

let store: NotesStore | null = null;
export function notesStore(): NotesStore {
  if (!store) {
    const url = process.env.KV_REST_API_URL;
    const token = process.env.KV_REST_API_TOKEN;
    store = url && token ? new KvStore(url, token) : new MemoryStore();
  }
  return store;
}
