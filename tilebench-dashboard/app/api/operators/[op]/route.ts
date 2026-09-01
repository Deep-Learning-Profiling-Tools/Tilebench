import { ok, fail, SAFE } from "@/lib/http";
import { getOperator, getProfile, getSourceIndex, getProfileReport } from "@/lib/data";
import { notesStore } from "@/lib/store";

/** Everything the drawer needs for one kernel, in a single round trip. */
export async function GET(_req: Request, ctx: { params: Promise<{ op: string }> }) {
  const { op } = await ctx.params;
  if (!SAFE.test(op)) return fail(400, "bad operator name");

  const operator = await getOperator(op);
  if (!operator) return fail(404, "unknown operator", { op });

  const [profile, src, notes, report] = await Promise.all([
    getProfile(op),
    getSourceIndex(),
    notesStore().get(op),
    getProfileReport(op),
  ]);

  return ok({
    operator,
    notes,
    profile: profile
      ? { ...profile, report_present: Boolean(report) }
      : null,
    source: src[op] ? Object.keys(src[op]).sort() : [],
  });
}
