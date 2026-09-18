import argparse
import csv
import json
import subprocess
from pathlib import Path

# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without requiring PYTHONPATH=.
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilebench.core.engine import run_benchmark_suite  # noqa: E402

def _archive_logs() -> None:
    """Snapshot results/logs/ onto the raw-log archive branch (local commit only;
    publish it with `scripts/archive_logs.sh --push`). Never fails the benchmark."""
    proc = subprocess.run(["bash", str(Path(__file__).with_name("archive_logs.sh"))],
                          capture_output=True, text=True)
    def last(text):
        return (text.strip().splitlines() or ["no output"])[-1]
    if proc.returncode == 0:
        print(f"Raw-log archive → {last(proc.stdout)}")
    else:
        print(f"Warning: raw-log archive skipped ({last(proc.stderr)}); "
              f"benchmark results are unaffected")


# Display labels for the tile-language backends (torch is the implicit baseline).
_BACKEND_LABEL = {"triton": "Triton", "cutile": "cuTile", "tilelang": "TileLang", "nki": "NKI"}
_SPEEDUP_CODE = {"triton": "T", "cutile": "C", "tilelang": "TL", "nki": "N"}


def _parse_params_label(label: str) -> dict[str, str]:
    parsed = {}
    for part in label.split(","):
        if "=" not in part:
            return {}
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _param_value_matches(actual, expected: str) -> bool:
    if str(actual) == expected:
        return True
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return False


def _label_matches_result(label: str, result: dict) -> bool:
    parsed = _parse_params_label(label)
    if not parsed:
        return False
    params = result["params"]
    for key, expected in parsed.items():
        if key == "n" and key not in params:
            actual = result["problem_size"]
        elif key in params:
            actual = params[key]
        else:
            return False
        if not _param_value_matches(actual, expected):
            return False
    return True


def _csv_aware_param_formatter(csv_path: str, timing_results: list[dict], fallback):
    """Reuse frozen CSV param labels when a filtered run has no varying keys."""
    path = Path(csv_path)
    if not path.exists():
        return fallback

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        old_rows = list(reader)

    preferred = {}
    for r in timing_results:
        matches = [
            row["params"]
            for row in old_rows
            if row.get("dtype") == r["dtype"] and _label_matches_result(row["params"], r)
        ]
        if len(matches) == 1:
            preferred[id(r)] = matches[0]

    def _fmt_params(r):
        return preferred.get(id(r), fallback(r))

    return _fmt_params


def _autotune_key(result: dict) -> tuple[str, int, str]:
    return (
        json.dumps(result["params"], sort_keys=True),
        result["problem_size"],
        result["dtype"],
    )


def _merge_into_autotune_log(log_path: str, autotune_results: list[dict],
                             active: list[str]) -> bool:
    """Merge active backend configs into an existing combined autotune log."""
    cfg_keys = [f"{b}_autotune_cfg" for b in active]
    if not cfg_keys or not any(k in r for r in autotune_results for k in cfg_keys):
        return False

    path = Path(log_path)
    if not path.exists():
        return False

    with path.open() as f:
        old_rows = json.load(f)
    index = {_autotune_key(r): r for r in old_rows}

    missing = [r for r in autotune_results if _autotune_key(r) not in index]
    if missing:
        examples = [
            (r["params"], r["problem_size"], r["dtype"])
            for r in missing[:5]
        ]
        raise SystemExit(
            f"autotune log merge: {len(missing)} case(s) have no matching row "
            f"in {log_path}, e.g. {examples} — refusing to merge"
        )

    updated = 0
    for r in autotune_results:
        old = index[_autotune_key(r)]
        for key in cfg_keys:
            if key in r:
                old[key] = r[key]
                updated += 1

    with path.open("w") as f:
        json.dump(old_rows, f, indent=4)
        f.write("\n")

    print(f"Merged {updated} autotune config(s) into {log_path}")
    return True


def _split(results: list[dict], active: list[str]) -> tuple[list[dict], list[dict]]:
    """Keep only torch + the active backends' keys, so backends that were not
    run never appear (as nan) in the timing / autotune logs."""
    timing_keys = {"params", "problem_size", "dtype", "torch_ms", "torch_stats"}
    autotune_keys = {"params", "problem_size", "dtype"}
    for b in active:
        timing_keys |= {f"{b}_ms", f"{b}_stats", f"{b}_ok", f"{b}_err", f"speedup_{b}"}
        autotune_keys.add(f"{b}_autotune_cfg")
    timing = [{k: v for k, v in r.items() if k in timing_keys} for r in results]
    autotune = [{k: v for k, v in r.items() if k in autotune_keys} for r in results]
    return timing, autotune


def _merge_into_csv(csv_path: str, timing_results: list[dict], active: list[str],
                    fmt_params) -> None:
    """In-place merge of a torch+{tilelang,nki} run into the frozen summary CSV.

    The frozen torch/triton/cutile columns are never touched. Rows are matched
    by (params, dtype).

    - tilelang: the freshly measured tilelang_ms is re-based onto the frozen
      timing environment via the per-case scale torch_frozen/torch_new, so the
      written tilelang_ms is directly comparable with the frozen triton/cutile
      columns. speedup_tilelang = torch_new/tilelang_new is scale-invariant
      and written as measured.
    - nki (run on the Neuron host): torch_nki_ms (torch timed on the Neuron
      device) and nki_ms are appended as-is — cross-hardware, so no scaling;
      NKI compares against its own torch reference.
    """
    path = Path(csv_path)
    if not path.exists():
        raise SystemExit(
            f"csv merge: {csv_path} does not exist — run the full "
            f"torch/triton/cutile benchmark first to create the frozen CSV"
        )
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = list(next(reader))
        old_rows = [dict(zip(header, row)) for row in reader]
    index = {(r["params"], r["dtype"]): r for r in old_rows}

    # Validate the full run maps onto frozen rows before mutating anything.
    keys = [(fmt_params(r), r["dtype"]) for r in timing_results]
    unmatched = [k for k in keys if k not in index]
    if unmatched:
        raise SystemExit(
            f"csv merge: {len(unmatched)} case(s) have no matching row in "
            f"{csv_path} (case grids diverged?), e.g. {unmatched[:5]} — "
            f"refusing to merge"
        )

    new_cols = []
    if "tilelang" in active:
        new_cols += ["tilelang_ms", "speedup_tilelang"]
    if "nki" in active:
        new_cols += ["torch_nki_ms", "nki_ms", "speedup_nki"]
    for c in new_cols:
        if c not in header:
            header.append(c)

    scales = []
    for r, key in zip(timing_results, keys):
        old = index[key]
        torch_frozen = float(old["torch_ms"])
        torch_new = r["torch_ms"]
        if "tilelang" in active:
            tl = r["tilelang_ms"]
            if tl > 0 and torch_new > 0 and torch_frozen > 0:
                scale = torch_frozen / torch_new
                scales.append(scale)
                old["tilelang_ms"] = f"{tl * scale:.4f}"
                old["speedup_tilelang"] = f"{r['speedup_tilelang']:.2f}"
            else:
                old["tilelang_ms"] = "nan"
                old["speedup_tilelang"] = "0.00"
        if "nki" in active:
            old["torch_nki_ms"] = f"{torch_new:.4f}" if torch_new > 0 else "nan"
            old["nki_ms"] = f"{r['nki_ms']:.4f}" if r["nki_ms"] > 0 else "nan"
            old["speedup_nki"] = f"{r['speedup_nki']:.2f}"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, restval="")
        writer.writeheader()
        writer.writerows(old_rows)

    print(f"Merged {len(keys)} case(s) into {csv_path} "
          f"(columns: {', '.join(new_cols)}; "
          f"{len(old_rows) - len(keys)} frozen row(s) untouched)")
    if scales:
        s = sorted(scales)
        print(f"  torch drift scale (frozen/new): min {s[0]:.3f} / "
              f"median {s[len(s) // 2]:.3f} / max {s[-1]:.3f}")


def main():

    parser = argparse.ArgumentParser(description="Run TileBench benchmarks")
    parser.add_argument("--operator", type=str, default="vector_add",
                        help="Operator to benchmark")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for timing results "
                             "(default: results/logs/time_measurement_logs/<operator>_results.json)")
    parser.add_argument("--autotune-log", type=str, default=None,
                        help="Output path for autotune config log "
                             "(default: results/logs/autotune_logs/<operator>_autotune.json)")
    parser.add_argument("--warmup", type=int, default=None,
                        help="Warmup iterations (default: 20)")
    parser.add_argument("--repeat", type=int, default=None,
                        help="Measured iterations inside Proton scope (default: 100)")
    parser.add_argument("--use-cuda-graph", action="store_true",
                        help="Capture CUDA graph; replay inside Proton scope")
    parser.add_argument("--proton-scope-name", type=str, default=None,
                        help="Proton scope name (default: launch)")
    parser.add_argument("--proton-backend", type=str, default=None,
                        help="Proton backend, e.g. cupti")
    parser.add_argument("--proton-context", type=str, default=None,
                        help="Proton context: shadow or python")
    parser.add_argument("--flush-l2", action="store_true",
                        help="Best-effort L2 flush before each iteration")
    parser.add_argument("--autotune", action="store_true",
                        help="Enable autotune (overrides config.yaml autotune setting)")
    parser.add_argument("--case-indices", type=str, default=None,
                        help="Comma-separated case indices to run, e.g. 0,1,3")
    parser.add_argument("--tile-language", type=str, default=None,
                        help="Comma-separated tile-language backends to run: "
                             "triton, cutile, tilelang, nki (or 'all'). torch "
                             "always runs as the speedup baseline. Default: all.")
    parser.add_argument("--no-archive", action="store_true",
                        help="Do not snapshot results/logs/ onto the raw-log archive branch")
    parser.add_argument("--keep-proton-files", action="store_true",
                        help="Keep intermediate Proton .hatchet files for inspection")
    parser.add_argument("--proton-output-dir", type=str, default=None,
                        help="Directory to store kept Proton files (default: system temp dir)")
    args = parser.parse_args()

    # Resolve which tile-language backends to run. torch is always on (speedup
    # baseline); omitting the flag runs them all (backward-compatible).
    _TILE_LANGUAGES = ("triton", "cutile", "tilelang", "nki")
    if args.tile_language is None:
        enabled_backends = set(_TILE_LANGUAGES)
    else:
        tokens = [t.strip().lower() for t in args.tile_language.split(",") if t.strip()]
        if "all" in tokens:
            enabled_backends = set(_TILE_LANGUAGES)
        else:
            enabled_backends = set()
            for t in tokens:
                if t == "torch":
                    continue  # always on; ignore if explicitly listed
                if t not in _TILE_LANGUAGES:
                    parser.error(
                        f"unknown --tile-language backend '{t}'; "
                        f"choose from {', '.join(_TILE_LANGUAGES)} (or 'all')"
                    )
                enabled_backends.add(t)

    # Active tile-language backends in canonical column order (drives all output).
    active = [b for b in _TILE_LANGUAGES if b in enabled_backends]

    overrides: dict = {}
    if args.warmup is not None:
        overrides["warmup"] = args.warmup
    if args.repeat is not None:
        overrides["repeat"] = args.repeat
    if args.use_cuda_graph:
        overrides["use_cuda_graph"] = True
    if args.proton_scope_name is not None:
        overrides["proton_scope_name"] = args.proton_scope_name
    if args.proton_backend is not None:
        overrides["proton_backend"] = args.proton_backend
    if args.proton_context is not None:
        overrides["proton_context"] = args.proton_context
    if args.flush_l2:
        overrides["flush_l2"] = True
    if args.autotune:
        overrides["autotune"] = True
    if args.case_indices is not None:
        overrides["case_indices"] = [
            int(x.strip()) for x in args.case_indices.split(",") if x.strip()
        ]
    if args.keep_proton_files:
        overrides["keep_proton_files"] = True
    if args.proton_output_dir is not None:
        overrides["proton_output_dir"] = args.proton_output_dir

    # Resolve output paths (operator-bound defaults)
    output_path = args.output or (
        f"results/logs/time_measurement_logs/{args.operator}_results.json"
    )
    autotune_path = args.autotune_log or (
        f"results/logs/autotune_logs/{args.operator}_autotune.json"
    )

    print(f"Starting benchmark for operator: {args.operator}")
    print(f"Tile-language backends: torch (baseline) + "
          f"{', '.join(sorted(enabled_backends)) or '(none)'}")
    results = run_benchmark_suite(
        args.operator, benchmark_overrides=overrides, enabled_backends=enabled_backends
    )

    if not results:
        print(
            f"\nNo cases produced results for '{args.operator}' — every case was "
            f"skipped (see the 'Skipped:' messages above). Refusing to overwrite "
            f"existing logs/CSV with empty data."
        )
        return

    timing_results, autotune_results = _split(results, active)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(autotune_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(timing_results, f, indent=4)
    print(f"Timing results  → {output_path}")

    merged_autotune_log = (
        active
        and set(active) <= {"tilelang", "nki"}
        and _merge_into_autotune_log(autotune_path, autotune_results, active)
    )
    if not merged_autotune_log:
        with open(autotune_path, "w") as f:
            json.dump(autotune_results, f, indent=4)
        print(f"Autotune log    → {autotune_path}")
    if not args.no_archive:
        _archive_logs()

    # Determine which param keys actually vary across ALL cases in this run.
    # Keys that are constant (same value in every case) are hidden to reduce noise.
    # When nothing varies (e.g. only dtype differs), fall back to problem_size.
    all_params = [r["params"] for r in timing_results]
    all_keys   = sorted({k for p in all_params for k in p})
    varying_keys = [k for k in all_keys if len({p.get(k) for p in all_params}) > 1]

    def _fallback_fmt_params(r):
        if varying_keys:
            return ", ".join(f"{k}={r['params'][k]}" for k in varying_keys if k in r["params"])
        return f"n={r['problem_size']}"

    mode_suffix = "autotune" if args.autotune else "default"
    csv_path = f"results/csv/{args.operator}_{mode_suffix}.csv"
    fmt_params = _csv_aware_param_formatter(csv_path, timing_results, _fallback_fmt_params)

    col_w = max((len(fmt_params(r)) for r in timing_results), default=20) + 2

    print("\nSummary:")
    header = f"{'Params':<{col_w}} | {'Dtype':>8} | {'Torch(ms)':>10}"
    header += "".join(f" | {_BACKEND_LABEL[b] + '(ms)':>12}" for b in active)
    header += "".join(f" | {'Speedup(' + _SPEEDUP_CODE[b] + ')':>11}" for b in active)
    print(header)
    print("-" * len(header))
    for r in timing_results:
        line = f"{fmt_params(r):<{col_w}} | {r['dtype']:8s} | {r['torch_ms']:10.4f}"
        line += "".join(f" | {r[f'{b}_ms']:12.4f}" for b in active)
        line += "".join(f" | {r[f'speedup_{b}']:11.2f}" for b in active)
        print(line)

    # Save summary as CSV. Filename suffix mirrors the run mode so default
    # and autotune sweeps don't overwrite each other:
    #   results/csv/<op>_default.csv   (no --autotune)
    #   results/csv/<op>_autotune.csv  (--autotune)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    # tilelang/nki runs never overwrite the frozen torch/triton/cutile CSV:
    # when only those backends ran and the summary CSV already exists, the
    # results are MERGED into it in place (see _merge_into_csv for the
    # per-case tilelang re-basing and the nki torch_nki_ms column). The
    # plain writer below only ever runs for triton/cutile sweeps or when
    # no summary CSV exists yet.
    if active and set(active) <= {"tilelang", "nki"} and Path(csv_path).exists():
        _merge_into_csv(csv_path, timing_results, active, fmt_params)
        return
    if active and set(active) <= {"tilelang", "nki"}:
        print(f"Note: {csv_path} does not exist yet — writing a fresh "
              f"torch+{'/'.join(active)} CSV (nothing to merge into)")
    # Direct cuTile/Triton latency ratio (>1 means cuTile slower), emitted
    # whenever both backends ran so the committed 8-column CSVs are
    # reproducible by this script alone.
    ratio = "triton" in active and "cutile" in active
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["params", "dtype", "torch_ms"]
            + [f"{b}_ms" for b in active]
            + [f"speedup_{b}" for b in active]
            + (["triton_vs_cutile"] if ratio else [])
        )
        for r in timing_results:
            writer.writerow(
                [fmt_params(r), r["dtype"], f"{r['torch_ms']:.4f}"]
                + [f"{r[f'{b}_ms']:.4f}" for b in active]
                + [f"{r[f'speedup_{b}']:.2f}" for b in active]
                + ([f"{r['cutile_ms'] / r['triton_ms']:.4f}"] if ratio else [])
            )
    print(f"Summary CSV     → {csv_path}")


if __name__ == "__main__":
    main()
