import type { Band } from "@/lib/metrics";

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
