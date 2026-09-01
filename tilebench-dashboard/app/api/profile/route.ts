import { ok } from "@/lib/http";
import { getProfiles } from "@/lib/data";

export async function GET() {
  const profiles = await getProfiles();
  return ok({
    profiles,
    // .ncu-rep binaries stay in the TileBench repo; only the write-ups ship.
    note: "ncu_reports lists artifacts by name and size; fetch them from repo_path in the TileBench checkout.",
    total_ncu_reports: profiles.reduce((a, p) => a + p.ncu_reports.length, 0),
  });
}
