#!/usr/bin/env python3
"""Build data/platforms.json: per-hardware board aggregates plus profiling-report coverage.

Benchmark numbers come from a TileBench results tree (results/<GPU>/csv/<op>_<mode>.csv).
Report coverage comes from the public file listing of the Hugging Face dataset; no report
is downloaded.

    python scripts/sync_platforms.py --results-root ../Tilebench/results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import urllib.request
from pathlib import Path

DASHBOARD = Path(__file__).resolve().parents[1]
OUT = DASHBOARD / "data" / "platforms.json"
HF_DATASET = "bcui2/NCU_report"
HF_API = f"https://huggingface.co/api/datasets/{HF_DATASET}"
MODES = ("default", "autotune")
BACKENDS = ("triton", "cutile", "tilelang")

# results/<id>/csv on the TileBench side, <hf_path>/ on the dataset side.
PLATFORMS = [
    {"id": "B200", "name": "NVIDIA B200", "note": "Blackwell · sm_100",
     "hf_path": "NVIDIA_B200", "profiler": "Nsight Compute"},
    {"id": "GH200", "name": "NVIDIA GH200", "note": "Grace Hopper · sm_90",
     "hf_path": "NVIDIA_GH200", "profiler": "Nsight Compute"},
    {"id": "MI300X", "name": "AMD Instinct MI300X", "note": "ROCm",
     "hf_path": "AMD_MI300X", "profiler": "rocprof-compute"},
]


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def aggregate(path: Path) -> dict[str, float | int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no benchmark rows")
    stats: dict[str, float | int] = {}
    for backend in BACKENDS:
        if f"{backend}_ms" not in rows[0]:
            continue
        ratios = []
        for line, row in enumerate(rows, start=2):
            torch_ms, ms = float(row["torch_ms"]), float(row[f"{backend}_ms"])
            if not (math.isfinite(torch_ms) and math.isfinite(ms) and torch_ms > 0 and ms > 0):
                raise ValueError(f"{path}:{line}: latency values must be finite and positive")
            ratios.append(torch_ms / ms)
        stats[backend] = round(geometric_mean(ratios), 6)
    stats["cases"] = len(rows)
    return stats


def list_reports(hf_path: str) -> dict[str, list[str]]:
    """op -> sorted report names (`<backend>_<dtype>`), from the dataset's file listing."""
    reports: dict[str, set[str]] = {}
    url: str | None = f"{HF_API}/tree/main/{hf_path}?recursive=true&limit=1000"
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            entries = json.load(response)
            link = response.headers.get("Link") or ""
        for entry in entries:
            parts = entry["path"].split("/")
            if entry["type"] != "file" or len(parts) < 3:
                continue
            # NVIDIA: <op>/<backend>_<dtype>.ncu-rep   AMD: <op>/<backend>_<dtype>/...
            name = parts[2].removesuffix(".ncu-rep")
            if len(parts) == 3 and not parts[2].endswith(".ncu-rep"):
                continue
            reports.setdefault(parts[1], set()).add(name)
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
    return {op: sorted(names) for op, names in sorted(reports.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", type=Path, default=DASHBOARD.parent / "Tilebench" / "results")
    parser.add_argument("--source", default="", help="provenance label, e.g. a commit hash")
    args = parser.parse_args()

    with urllib.request.urlopen(HF_API, timeout=60) as response:
        hf_sha = json.load(response).get("sha")

    platforms = []
    for spec in PLATFORMS:
        csv_dir = args.results_root / spec["id"] / "csv"
        ops = sorted(p.name.removesuffix("_default.csv") for p in csv_dir.glob("*_default.csv"))
        if not ops:
            raise SystemExit(f"no results under {csv_dir}")
        operators = [
            {"op": op, **{mode: aggregate(csv_dir / f"{op}_{mode}.csv") for mode in MODES}}
            for op in ops
        ]
        backends = [b for b in BACKENDS if any(b in o[m] for o in operators for m in MODES)]
        platforms.append({**spec, "backends": backends, "operators": operators,
                          "reports": list_reports(spec["hf_path"])})
        print(f"{spec['id']}: {len(operators)} operators, backends {backends}, "
              f"{sum(len(v) for v in platforms[-1]['reports'].values())} reports")

    OUT.write_text(json.dumps({
        "results_source": args.source,
        "reports_dataset": HF_DATASET,
        "reports_revision": hf_sha,
        "platforms": platforms,
    }, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
