import { ok, fail, SAFE } from "@/lib/http";
import { getProfile, getProfileReport } from "@/lib/data";

/** GET /api/profile/:op — NCU write-up plus the artifact index for that kernel. */
export async function GET(_req: Request, ctx: { params: Promise<{ op: string }> }) {
  const { op } = await ctx.params;
  if (!SAFE.test(op)) return fail(400, "bad operator name");

  const [profile, report] = await Promise.all([getProfile(op), getProfileReport(op)]);
  if (!profile) return fail(404, "no profile run for this operator", { op });
  return ok({ op, profile, report });
}
