#!/usr/bin/env python3
"""Refresh dashboard aggregates from TileBench per-case benchmark CSVs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


DASHBOARD = Path(__file__).resolve().parents[1]
TILEBENCH = DASHBOARD.parent
if not (TILEBENCH / "results" / "csv").is_dir():
    TILEBENCH = DASHBOARD.parent / "Tilebench"
DATA_JSON = DASHBOARD / "data" / "operators.json"
DATA_CSV = DASHBOARD / "data" / "operators.csv"
RESULTS = TILEBENCH / "results" / "csv"
MODES = ("default", "autotune")
METRICS = ("triton", "cutile", "tilelang", "tl_over_triton", "tl_over_cutile")


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def aggregate(path: Path) -> dict[str, float | int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no benchmark rows")

    columns = ("torch_ms", "triton_ms", "cutile_ms", "tilelang_ms")
    parsed: list[dict[str, float]] = []
    for line, row in enumerate(rows, start=2):
        try:
            values = {column: float(row[column]) for column in columns}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line}: incomplete latency data") from error
        if not all(math.isfinite(value) and value > 0 for value in values.values()):
            raise ValueError(f"{path}:{line}: latency values must be finite and positive")
        parsed.append(values)

    ratios = {
        "triton": [row["torch_ms"] / row["triton_ms"] for row in parsed],
        "cutile": [row["torch_ms"] / row["cutile_ms"] for row in parsed],
        "tilelang": [row["torch_ms"] / row["tilelang_ms"] for row in parsed],
        "tl_over_triton": [row["triton_ms"] / row["tilelang_ms"] for row in parsed],
        "tl_over_cutile": [row["cutile_ms"] / row["tilelang_ms"] for row in parsed],
    }
    return {**{key: round(geometric_mean(values), 3) for key, values in ratios.items()}, "cases": len(rows)}


def csv_row(operator: dict) -> dict[str, object]:
    row: dict[str, object] = {
        "op": operator["op"],
        "tier": operator["tier"],
        "target": str(operator["target"]).lower(),
        "autotune_excluded_reason": operator.get("autotune_excluded_reason") or "",
    }
    for mode in MODES:
        stats = operator.get(mode)
        for metric in (*METRICS, "cases"):
            row[f"{mode}_{metric}"] = "" if stats is None else stats[metric]
    return row


def main() -> None:
    operators = {operator["op"]: operator for operator in json.loads(DATA_JSON.read_text())}
    source_ops = sorted(path.name.removesuffix("_default.csv") for path in RESULTS.glob("*_default.csv"))
    for op in source_ops:
        operator = operators.get(op, {"op": op, "tier": "both", "target": False})
        for mode in MODES:
            operator[mode] = aggregate(RESULTS / f"{op}_{mode}.csv")
        operator["tier"] = "both"
        operator["autotune_excluded_reason"] = None
        operators[op] = operator

    ordered = [operators[op] for op in sorted(operators)]
    DATA_JSON.write_text(json.dumps(ordered, indent=2) + "\n")

    fieldnames = ["op", "tier", "target"]
    for mode in MODES:
        fieldnames.extend(f"{mode}_{metric}" for metric in (*METRICS, "cases"))
    fieldnames.append("autotune_excluded_reason")
    with DATA_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_row(operator) for operator in ordered)

    print(f"wrote {len(ordered)} operators to {DATA_CSV} and {DATA_JSON}")


if __name__ == "__main__":
    main()
