export type Backend = "triton" | "cutile" | "tilelang";
export type Mode = "default" | "autotune";
export type Tier = "both" | "partial" | "excluded";

/** Per-mode aggregate. Backend figures are geomean speedup vs torch. */
export interface ModeStats {
  triton: number;
  cutile: number;
  tilelang: number;
  tl_over_triton: number;
  tl_over_cutile: number;
  cases: number;
}

export interface Operator {
  op: string;
  tier: Tier;
  /** flagged as a profiling target */
  target: boolean;
  default: ModeStats | null;
  autotune: ModeStats | null;
  autotune_excluded_reason?: string | null;
}

export interface Note {
  ts: number;
  body: string;
  author?: string;
}

export interface NcuReportFile {
  file: string;
  bytes: number;
}

export interface ProfileEntry {
  op: string;
  run: string;
  has_report: boolean;
  report_bytes: number;
  ncu_reports: NcuReportFile[];
  /** path within the TileBench repo; binaries are not shipped with the app */
  repo_path: string;
}

export interface PooledStats {
  triton: number;
  cutile: number;
  tilelang: number;
  tl_over_triton: number;
  tl_over_cutile: number;
  cases: number;
  operators: number;
}

/** One mode on one platform. A backend is absent when that hardware has no results for it. */
export type PlatformStats = Partial<Record<Backend, number>> & { cases: number };

export interface PlatformOperator {
  op: string;
  default: PlatformStats | null;
  autotune: PlatformStats | null;
}

export interface Platform {
  id: string;
  name: string;
  note: string;
  /** folder in the profiling-report dataset */
  hf_path: string;
  profiler: string;
  backends: Backend[];
  operators: PlatformOperator[];
  /** op -> report names (`<backend>_<dtype>`) present in the dataset */
  reports: Record<string, string[]>;
}

export interface PlatformData {
  results_source: string;
  reports_dataset: string;
  reports_revision: string | null;
  platforms: Platform[];
}
