import type { Band } from "@/lib/metrics";

export const REPO = "https://github.com/Deep-Learning-Profiling-Tools/Tilebench";
/** Where a kernel's implementations live in the TileBench repository. */
export const sourceUrl = (op: string, file?: string) =>
  `${REPO}/${file ? "blob" : "tree"}/main/tilebench/benchmarks/operators/${op}${file ? `/${file}` : ""}`;

export function num(n: number | null | undefined, digits = 3): string {
  return n == null || Number.isNaN(n) ? "—" : n.toFixed(digits);
}

export function band(ratio: number | null | undefined): Band | "none" {
  if (ratio == null || Number.isNaN(ratio)) return "none";
  if (ratio >= 1.05) return "win";
  if (ratio >= 0.95) return "parity";
  if (ratio >= 0.7) return "behind";
  return "loss";
}

export function when(ts: number): string {
  return new Date(ts).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** A speedup over PyTorch, with its unit: 2.05×, 0.306×. */
export function times(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n)) return "—";
  return `${n.toFixed(n >= 10 ? 1 : n >= 1 ? 2 : 3)}×`;
}

/** The same figure in words: "2.05× faster than PyTorch", "3.27× slower than PyTorch". */
export function versus(n: number | null | undefined): string {
  if (n == null || Number.isNaN(n) || !(n > 0)) return "";
  if (n >= 0.995 && n < 1.005) return "same speed as PyTorch";
  return n > 1 ? `${n.toFixed(2)}× faster than PyTorch` : `${(1 / n).toFixed(2)}× slower than PyTorch`;
}

const NAMES: Record<string, string> = {
  triton: "Triton",
  cutile: "cuTile",
  tilelang: "TileLang",
  torch: "PyTorch",
};

export function label(name: string): string {
  return NAMES[name] ?? name;
}
