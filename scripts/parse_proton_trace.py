import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _metric_time_ns(metrics: dict[str, Any]) -> float:
    if not isinstance(metrics, dict):
        return 0.0
    for key in ("time (ns)", "time(ns)", "time_ns"):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    val_us = metrics.get("time (us)")
    if isinstance(val_us, (int, float)):
        return float(val_us) * 1e3
    val_ms = metrics.get("time (ms)")
    if isinstance(val_ms, (int, float)):
        return float(val_ms) * 1e6
    return 0.0


def _iter_nodes(node: dict[str, Any]):
    yield node
    children = node.get("children", [])
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                yield from _iter_nodes(child)


def _kernel_time_ns(node: dict[str, Any]) -> float:
    total = 0.0
    metrics = node.get("metrics", {})
    if isinstance(metrics, dict):
        dev = str(metrics.get("device_type", "")).upper()
        if dev in {"CUDA", "HIP"}:
            total += _metric_time_ns(metrics)
    children = node.get("children", [])
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                total += _kernel_time_ns(child)
    return total


def extract_launch_time_ns(hatchet_data: Any, scope_name: str = "launch") -> float:
    roots: list[dict[str, Any]] = []
    if isinstance(hatchet_data, list):
        for item in hatchet_data:
            if isinstance(item, dict) and isinstance(item.get("frame"), dict):
                roots.append(item)
    elif isinstance(hatchet_data, dict) and isinstance(hatchet_data.get("frame"), dict):
        roots.append(hatchet_data)

    scope_lower = scope_name.lower()
    scope_total_ns = 0.0
    fallback_total_ns = 0.0
    for root in roots:
        fallback_total_ns += _kernel_time_ns(root)
        for node in _iter_nodes(root):
            frame = node.get("frame", {})
            if isinstance(frame, dict) and str(frame.get("name", "")).lower() == scope_lower:
                scope_total_ns += _kernel_time_ns(node)
    if scope_total_ns > 0:
        return scope_total_ns
    return fallback_total_ns


def summarize_samples_ms(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {}
    ts = torch.tensor(samples_ms, dtype=torch.float64)
    return {
        "count": float(len(samples_ms)),
        "mean": float(ts.mean().item()),
        "std": float(ts.std(unbiased=False).item()),
        "cold_start": float(samples_ms[0]),
        "p50": float(torch.quantile(ts, 0.5).item()),
        "p90": float(torch.quantile(ts, 0.9).item()),
        "p99": float(torch.quantile(ts, 0.99).item()),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract launch timing samples from Proton hatchet files.")
    parser.add_argument("--input", required=True, help="Path to one .hatchet file or a directory of .hatchet files.")
    parser.add_argument("--scope", default="launch", help="Scope name to filter. Default: launch")
    parser.add_argument("--print-samples", action="store_true", help="Print per-file sample list in ms.")
    args = parser.parse_args()

    target = Path(args.input)
    files: list[Path]
    if target.is_dir():
        files = sorted(target.glob("*.hatchet"))
    else:
        files = [target]

    if not files:
        print(f"No .hatchet files found for input: {args.input}")
        return

    samples_ms: list[float] = []
    for file_path in files:
        with file_path.open("r") as f:
            hatchet = json.load(f)
        ns = extract_launch_time_ns(hatchet, args.scope)
        samples_ms.append(ns / 1e6)

    stats = summarize_samples_ms(samples_ms)
    if not stats:
        print(f"No timing samples extracted from scope '{args.scope}'.")
        return

    print(f"Scope '{args.scope}' timing summary (ms):")
    print(json.dumps(stats, indent=2))
    if args.print_samples:
        print("samples_ms:", samples_ms)


if __name__ == "__main__":
    main()
