import { ok } from "@/lib/http";
import { getPlatforms } from "@/lib/data";
import { poolPlatform } from "@/lib/metrics";

/** GET /api/platforms — every measured platform with per-operator stats and report coverage. */
export async function GET() {
  const data = await getPlatforms();
  return ok({
    ...data,
    platforms: data.platforms.map((p) => ({
      ...p,
      pooled: { default: poolPlatform(p, "default"), autotune: poolPlatform(p, "autotune") },
    })),
  });
}
