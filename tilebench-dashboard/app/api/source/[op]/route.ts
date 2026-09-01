import { ok, fail, SAFE } from "@/lib/http";
import { getSourceIndex } from "@/lib/data";

export async function GET(_req: Request, ctx: { params: Promise<{ op: string }> }) {
  const { op } = await ctx.params;
  if (!SAFE.test(op)) return fail(400, "bad operator name");
  const idx = await getSourceIndex();
  if (!idx[op]) return fail(404, "no source for this operator", { op });
  return ok({
    op,
    files: Object.entries(idx[op])
      .map(([file, bytes]) => ({ file, bytes }))
      .sort((a, b) => a.file.localeCompare(b.file)),
  });
}
