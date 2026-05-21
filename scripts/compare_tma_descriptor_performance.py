#!/usr/bin/env python3
"""Compare descriptor/TMA and non-descriptor Triton timing CSVs.

This script compares per-case benchmark CSVs under two Tilebench result roots:

* descriptor/TMA results, default: ``results``
* non-descriptor baseline results, default: ``results_exp_timing``

It reads ``csv/<operator>_{default,autotune}.csv`` files from both roots,
joins rows by ``operator, mode, params, dtype``, and writes case-level and
aggregate comparison CSVs.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MODES = ("default", "autotune")
NUMERIC_COLUMNS = (
    "torch_ms",
    "triton_ms",
    "cutile_ms",
    "speedup_triton",
    "speedup_cutile",
)


@dataclass(frozen=True)
class RowKey:
    operator: str
    mode: str
    params: str
    dtype: str


def normalize_dtype(dtype: str) -> str:
    text = (dtype or "").strip().lower()
    aliases = {
        "float16": "fp16",
        "torch.float16": "fp16",
        "half": "fp16",
        "float32": "fp32",
        "torch.float32": "fp32",
        "float": "fp32",
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
        "int8": "int8",
        "torch.int8": "int8",
        "fp8": "fp8",
        "fp16": "fp16",
        "fp32": "fp32",
        "bf16": "bf16",
    }
    return aliases.get(text, text)


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def fmt_float(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.8g}"


def operator_from_mode_csv(path: Path, mode: str) -> str:
    suffix = f"_{mode}.csv"
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"unexpected file name for mode {mode}: {path}")
    return name[: -len(suffix)]


def iter_mode_csvs(root: Path) -> Iterable[tuple[str, str, Path]]:
    csv_dir = root / "csv"
    if not csv_dir.is_dir():
        return
    for mode in MODES:
        for path in sorted(csv_dir.glob(f"*_{mode}.csv")):
            yield operator_from_mode_csv(path, mode), mode, path


def read_case_rows(root: Path) -> dict[RowKey, dict[str, str]]:
    rows: dict[RowKey, dict[str, str]] = {}
    for operator, mode, path in iter_mode_csvs(root):
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            required = {"params", "dtype", "triton_ms"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                continue
            for row in reader:
                params = (row.get("params") or "").strip()
                dtype = normalize_dtype(row.get("dtype") or "")
                if not params or not dtype:
                    continue
                key = RowKey(operator=operator, mode=mode, params=params, dtype=dtype)
                rows[key] = row
    return rows


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def pct_delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / old * 100.0


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def geomean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None and v > 0 and not math.isnan(v)]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float):
                    formatted[field] = fmt_float(value)
                elif value is None:
                    formatted[field] = ""
                else:
                    formatted[field] = value
            writer.writerow(formatted)


def build_case_comparison(
    descriptor_rows: dict[RowKey, dict[str, str]],
    baseline_rows: dict[RowKey, dict[str, str]],
) -> list[dict[str, object]]:
    all_keys = sorted(
        set(descriptor_rows) | set(baseline_rows),
        key=lambda k: (k.operator, k.mode, k.dtype, k.params),
    )
    out: list[dict[str, object]] = []
    for key in all_keys:
        desc = descriptor_rows.get(key)
        base = baseline_rows.get(key)
        desc_triton = parse_float(desc.get("triton_ms") if desc else None)
        base_triton = parse_float(base.get("triton_ms") if base else None)
        desc_torch = parse_float(desc.get("torch_ms") if desc else None)
        base_torch = parse_float(base.get("torch_ms") if base else None)
        desc_cutile = parse_float(desc.get("cutile_ms") if desc else None)
        base_cutile = parse_float(base.get("cutile_ms") if base else None)
        speedup = ratio(base_triton, desc_triton)
        out.append(
            {
                "operator": key.operator,
                "mode": key.mode,
                "dtype": key.dtype,
                "params": key.params,
                "descriptor_present": desc is not None,
                "baseline_present": base is not None,
                "descriptor_triton_ms": desc_triton,
                "baseline_triton_ms": base_triton,
                "descriptor_vs_baseline_speedup": speedup,
                "descriptor_triton_delta_pct": pct_delta(desc_triton, base_triton),
                "descriptor_torch_ms": desc_torch,
                "baseline_torch_ms": base_torch,
                "descriptor_cutile_ms": desc_cutile,
                "baseline_cutile_ms": base_cutile,
                "descriptor_torch_speedup": ratio(desc_torch, desc_triton),
                "baseline_torch_speedup": ratio(base_torch, base_triton),
                "descriptor_cutile_over_triton": ratio(desc_triton, desc_cutile),
                "baseline_cutile_over_triton": ratio(base_triton, base_cutile),
            }
        )
    return out


def aggregate_case_rows(case_rows: list[dict[str, object]], group_fields: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in case_rows:
        if not (row.get("descriptor_present") and row.get("baseline_present")):
            continue
        groups[tuple(row[field] for field in group_fields)].append(row)

    out: list[dict[str, object]] = []
    for group_key, rows in sorted(groups.items()):
        desc_ms = [row["descriptor_triton_ms"] for row in rows]
        base_ms = [row["baseline_triton_ms"] for row in rows]
        speedups = [row["descriptor_vs_baseline_speedup"] for row in rows]
        item = {field: value for field, value in zip(group_fields, group_key)}
        item.update(
            {
                "n_cases": len(rows),
                "descriptor_triton_ms_mean": mean(desc_ms),
                "baseline_triton_ms_mean": mean(base_ms),
                "descriptor_vs_baseline_speedup_mean": mean(speedups),
                "descriptor_vs_baseline_speedup_geomean": geomean(speedups),
                "descriptor_triton_delta_pct_mean": mean(
                    row["descriptor_triton_delta_pct"] for row in rows
                ),
            }
        )
        out.append(item)
    return out


def write_missing_report(
    out_dir: Path,
    descriptor_rows: dict[RowKey, dict[str, str]],
    baseline_rows: dict[RowKey, dict[str, str]],
) -> None:
    missing_rows = []
    for key in sorted(set(descriptor_rows) ^ set(baseline_rows), key=lambda k: (k.operator, k.mode, k.dtype, k.params)):
        missing_rows.append(
            {
                "operator": key.operator,
                "mode": key.mode,
                "dtype": key.dtype,
                "params": key.params,
                "missing_from": "descriptor" if key not in descriptor_rows else "baseline",
            }
        )
    write_csv(
        out_dir / "missing_cases.csv",
        ["operator", "mode", "dtype", "params", "missing_from"],
        missing_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-root", type=Path, default=Path("results"))
    parser.add_argument("--baseline-root", type=Path, default=Path("results_exp_timing"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/tma_descriptor_comparison"))
    args = parser.parse_args()

    descriptor_rows = read_case_rows(args.descriptor_root)
    baseline_rows = read_case_rows(args.baseline_root)
    case_rows = build_case_comparison(descriptor_rows, baseline_rows)

    case_fields = [
        "operator",
        "mode",
        "dtype",
        "params",
        "descriptor_present",
        "baseline_present",
        "descriptor_triton_ms",
        "baseline_triton_ms",
        "descriptor_vs_baseline_speedup",
        "descriptor_triton_delta_pct",
        "descriptor_torch_ms",
        "baseline_torch_ms",
        "descriptor_cutile_ms",
        "baseline_cutile_ms",
        "descriptor_torch_speedup",
        "baseline_torch_speedup",
        "descriptor_cutile_over_triton",
        "baseline_cutile_over_triton",
    ]
    write_csv(args.out_dir / "case_comparison.csv", case_fields, case_rows)

    aggregate_fields = [
        "operator",
        "mode",
        "dtype",
        "n_cases",
        "descriptor_triton_ms_mean",
        "baseline_triton_ms_mean",
        "descriptor_vs_baseline_speedup_mean",
        "descriptor_vs_baseline_speedup_geomean",
        "descriptor_triton_delta_pct_mean",
    ]
    write_csv(
        args.out_dir / "operator_dtype_mode_summary.csv",
        aggregate_fields,
        aggregate_case_rows(case_rows, ("operator", "mode", "dtype")),
    )

    operator_fields = [
        "operator",
        "mode",
        "n_cases",
        "descriptor_triton_ms_mean",
        "baseline_triton_ms_mean",
        "descriptor_vs_baseline_speedup_mean",
        "descriptor_vs_baseline_speedup_geomean",
        "descriptor_triton_delta_pct_mean",
    ]
    write_csv(
        args.out_dir / "operator_mode_summary.csv",
        operator_fields,
        aggregate_case_rows(case_rows, ("operator", "mode")),
    )

    dtype_fields = [
        "mode",
        "dtype",
        "n_cases",
        "descriptor_triton_ms_mean",
        "baseline_triton_ms_mean",
        "descriptor_vs_baseline_speedup_mean",
        "descriptor_vs_baseline_speedup_geomean",
        "descriptor_triton_delta_pct_mean",
    ]
    write_csv(
        args.out_dir / "dtype_mode_summary.csv",
        dtype_fields,
        aggregate_case_rows(case_rows, ("mode", "dtype")),
    )

    write_missing_report(args.out_dir, descriptor_rows, baseline_rows)

    matched = sum(1 for row in case_rows if row["descriptor_present"] and row["baseline_present"])
    print(f"descriptor rows: {len(descriptor_rows)}")
    print(f"baseline rows:   {len(baseline_rows)}")
    print(f"matched rows:    {matched}")
    print(f"output dir:      {args.out_dir}")


if __name__ == "__main__":
    main()
