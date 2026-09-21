import argparse
import json
import os
import shutil
import socket
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without setting PYTHONPATH
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilebench.core.engine import run_benchmark_suite  # noqa: E402
from tilebench.paths import (OPERATOR_ROOT, hardware_label,  # noqa: E402
                             results_logs_dir, results_runs_dir)


def discover_operators(operators_root: Path) -> list[str]:
    operators = []
    if not operators_root.exists():
        return operators
    for child in sorted(operators_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "config.yaml").exists():
            operators.append(child.name)
    return operators


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_run_dir(base_dir: Path, run_name: str = "") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_name}" if run_name else ""
    run_dir = base_dir / f"{timestamp}{suffix}"
    index = 1
    while run_dir.exists():
        run_dir = base_dir / f"{timestamp}{suffix}_{index}"
        index += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def copy_profile_artifacts(patterns: list[str], run_dir: Path, repo_root: Path) -> list[str]:
    copied = []
    profiles_dir = run_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    if not patterns:
        (profiles_dir / "README.txt").write_text(
            "No profile artifacts were provided for this run.\n"
            "Pass --profile-artifact with file/dir/glob paths to collect them here.\n",
            encoding="utf-8",
        )
        return copied

    matched_paths = []
    for pattern in patterns:
        matched_paths.extend(repo_root.glob(pattern))

    if not matched_paths:
        (profiles_dir / "README.txt").write_text(
            "No profile artifacts matched provided patterns.\n",
            encoding="utf-8",
        )
        return copied

    for src in matched_paths:
        src = src.resolve()
        dst = profiles_dir / src.name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        elif src.is_file():
            if dst.exists():
                stem, suffix = dst.stem, dst.suffix
                i = 1
                while (profiles_dir / f"{stem}_{i}{suffix}").exists():
                    i += 1
                dst = profiles_dir / f"{stem}_{i}{suffix}"
            shutil.copy2(src, dst)
        copied.append(str(dst.relative_to(run_dir)))

    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all TileBench operators and save results per run.")
    parser.add_argument(
        "--gpu",
        type=hardware_label,
        required=True,
        metavar="LABEL",
        help="Hardware label of the machine being measured, e.g. B200 or GH200. "
             "Selects the result namespace results/<gpu>/ and is recorded in the run manifest.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Root directory for per-run outputs (default: results/<gpu>/runs).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run name suffix for easier identification.",
    )
    parser.add_argument(
        "--operators",
        type=str,
        nargs="*",
        default=None,
        help="Optional operator list. If omitted, auto-discovers all operators.",
    )
    parser.add_argument(
        "--profile-artifact",
        type=str,
        nargs="*",
        default=[],
        help="Optional file/dir/glob paths to copy into this run's profiles directory.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    operators_root = OPERATOR_ROOT
    operators = args.operators if args.operators else discover_operators(operators_root)
    if not operators:
        print("No operators found to run.")
        return 1

    results_root = Path(args.results_root) if args.results_root else results_runs_dir(args.gpu)
    run_dir = create_run_dir(results_root, args.run_name)
    operators_dir = run_dir / "operators"
    operators_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = logs_dir / "run.log"

    manifest = {
        "gpu": args.gpu,
        "created_at_utc": _now_utc_str(),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "operators_requested": operators,
        "operators_succeeded": [],
        "operators_failed": [],
        "artifacts": {},
    }

    print(f"GPU/result namespace: {args.gpu}")
    print(f"Run directory: {run_dir}")
    with run_log_path.open("w", encoding="utf-8") as run_log:
        run_log.write(f"[{_now_utc_str()}] Starting run for {len(operators)} operators\n")
        for op in operators:
            print(f"=== Running {op} ===")
            run_log.write(f"\n[{_now_utc_str()}] START operator={op}\n")
            try:
                results = run_benchmark_suite(op, logs_dir=results_logs_dir(args.gpu))
                out_path = operators_dir / f"{op}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                manifest["operators_succeeded"].append(op)
                run_log.write(f"[{_now_utc_str()}] DONE operator={op} output={out_path}\n")
            except Exception:
                err = traceback.format_exc()
                manifest["operators_failed"].append({"operator": op, "error": err})
                err_path = operators_dir / f"{op}.error.txt"
                err_path.write_text(err, encoding="utf-8")
                run_log.write(f"[{_now_utc_str()}] FAIL operator={op}\n{err}\n")
                print(f"  FAILED: {op}")

        copied_profiles = copy_profile_artifacts(args.profile_artifact, run_dir, repo_root)
        manifest["artifacts"]["profiles"] = copied_profiles
        manifest["finished_at_utc"] = _now_utc_str()

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nRun complete.")
    print(f"Summary: {summary_path}")
    print(f"Succeeded: {len(manifest['operators_succeeded'])}")
    print(f"Failed: {len(manifest['operators_failed'])}")

    return 0 if not manifest["operators_failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
