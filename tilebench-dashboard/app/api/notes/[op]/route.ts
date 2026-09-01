import { ok, fail, SAFE } from "@/lib/http";
import { getOperator } from "@/lib/data";
import { notesStore } from "@/lib/store";

const MAX = 4000;

export async function GET(_req: Request, ctx: { params: Promise<{ op: string }> }) {
  const { op } = await ctx.params;
  if (!SAFE.test(op)) return fail(400, "bad operator name");
  return ok({ op, notes: await notesStore().get(op) });
}

export async function POST(req: Request, ctx: { params: Promise<{ op: string }> }) {
  const { op } = await ctx.params;
  if (!SAFE.test(op)) return fail(400, "bad operator name");
  if (!(await getOperator(op))) return fail(404, "unknown operator", { op });

  let payload: { body?: unknown; author?: unknown };
  try {
    payload = await req.json();
  } catch {
    return fail(400, "body must be JSON");
  }
  const body = typeof payload.body === "string" ? payload.body.trim() : "";
  if (!body) return fail(400, "note body is required");
  if (body.length > MAX) return fail(413, `note exceeds ${MAX} characters`);

  const store = notesStore();
  const notes = await store.add(op, {
    ts: Date.now(),
    body,
    ...(typeof payload.author === "string" && payload.author ? { author: payload.author } : {}),
  });
  return ok({ op, notes, durable: store.durable }, { status: 201 });
}

export async function DELETE(req: Request, ctx: { params: Promise<{ op: string }> }) {
  const { op } = await ctx.params;
  if (!SAFE.test(op)) return fail(400, "bad operator name");
  const ts = Number(new URL(req.url).searchParams.get("ts"));
  if (!Number.isFinite(ts)) return fail(400, "ts query parameter is required");
  return ok({ op, notes: await notesStore().remove(op, ts) });
}
