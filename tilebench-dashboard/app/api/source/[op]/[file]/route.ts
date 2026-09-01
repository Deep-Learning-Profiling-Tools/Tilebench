import { fail, SAFE, SAFE_FILE } from "@/lib/http";
import { getSourceFile } from "@/lib/data";

/** Served as text/plain so it drops straight into a viewer or an agent prompt. */
export async function GET(_req: Request, ctx: { params: Promise<{ op: string; file: string }> }) {
  const { op, file } = await ctx.params;
  if (!SAFE.test(op) || !SAFE_FILE.test(file)) return fail(400, "bad path segment");
  const body = await getSourceFile(op, file);
  if (body === null) return fail(404, "no such source file", { op, file });
  return new Response(body, {
    headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "public, max-age=3600" },
  });
}
