import { ok } from "@/lib/http";
import { notesStore } from "@/lib/store";

export async function GET() {
  const store = notesStore();
  const notes = await store.all();
  return ok({
    notes,
    total: Object.values(notes).reduce((a, v) => a + v.length, 0),
    operators: Object.keys(notes).length,
    store: { kind: store.kind, durable: store.durable },
  });
}
