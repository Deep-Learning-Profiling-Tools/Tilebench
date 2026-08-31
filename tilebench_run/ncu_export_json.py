"""Export NCU .ncu-rep files to queryable JSON.

The .ncu-rep file remains the source of truth. This script builds a structured
index for dashboards and agents using NVIDIA's Python report interface, plus
optional CLI page dumps for audit/debugging.

Examples:
  python tilebench_run/ncu_export_json.py
  python tilebench_run/ncu_export_json.py --reports tilebench_run/ncu/softmax/tilelang_fp16.ncu-rep
  python tilebench_run/ncu_export_json.py --no-cli-pages
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ncu_common import ncu_bin, ncu_json_dir, repo_root


ROOT = repo_root()
NCU_DIR = ROOT / "tilebench_run" / "ncu"
BACKENDS = ("triton", "cutile", "tilelang", "torch")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "items"):
        try:
            return {str(k): _jsonable(v) for k, v in value.items()}
        except Exception:
            pass
    return str(value)


def _call(obj: Any, name: str, default: Any = None) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        return default
    try:
        return _jsonable(fn())
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def _call_name_base(action: Any, attr: str) -> Any:
    if not hasattr(action, attr):
        return None
    try:
        return _jsonable(action.name(getattr(action, attr)))
    except Exception as exc:
        return {"__error__": f"{type(exc).__name__}: {exc}"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_ncu_report_paths() -> None:
    explicit = os.environ.get("NCU_REPORT_PYTHON")
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(sorted(glob.glob("/opt/nvidia/nsight-compute/*/extras/python"), reverse=True))
    candidates.extend(sorted(glob.glob("/usr/local/cuda*/nsight-compute*/extras/python"), reverse=True))
    candidates.extend(sorted(glob.glob("/usr/local/cuda*/NsightCompute-*/extras/python"), reverse=True))
    for cand in candidates:
        if cand and cand not in sys.path:
            sys.path.append(cand)


def import_ncu_report():
    _add_ncu_report_paths()
    try:
        import ncu_report  # type: ignore
        return ncu_report
    except Exception as exc:
        raise SystemExit(
            "error: could not import NVIDIA's ncu_report module. Install the "
            "`ncu-report` Python package, set NCU_REPORT_PYTHON to Nsight "
            "Compute's extras/python directory, or run this on a machine with "
            "Nsight Compute installed. Original error: "
            f"{type(exc).__name__}: {exc}"
        )


def parse_report_name(path: Path) -> dict[str, Any]:
    op = path.parent.name
    if op == "reports" and path.parent.parent.name:
        op = path.parent.parent.name
    stem = path.name.removesuffix(".ncu-rep").removesuffix(".ncu-repz")
    backend = None
    dtype = None
    parts = stem.split("_")
    for idx, part in enumerate(parts):
        if part in BACKENDS:
            backend = part
            dtype = "_".join(parts[idx + 1:]) or None
            break
    return {"op": op, "backend": backend, "dtype": dtype, "stem": stem}


def source_info_record(info: Any) -> Any:
    if info is None:
        return None
    return {
        "file_name": _call(info, "file_name"),
        "line": _call(info, "line"),
        "column": _call(info, "column"),
        "function_name": _call(info, "function_name"),
    }


def metric_record(metric: Any) -> dict[str, Any]:
    return {
        "name": _call(metric, "name"),
        "value": _call(metric, "value"),
        "unit": _call(metric, "unit"),
        "description": _call(metric, "description"),
        "metric_type": _call(metric, "metric_type"),
        "metric_subtype": _call(metric, "metric_subtype"),
        "rollup_operation": _call(metric, "rollup_operation"),
    }


def nvtx_record(action: Any) -> Any:
    state = getattr(action, "nvtx_state", lambda: None)()
    if state is None:
        return None
    domains = _call(state, "domains", [])
    out = []
    for domain_id in domains if isinstance(domains, list) else []:
        domain = state.domain_by_id(domain_id)
        out.append({
            "domain_id": domain_id,
            "name": _call(domain, "name"),
            "push_pop_ranges": _call(domain, "push_pop_ranges", []),
            "start_end_ranges": _call(domain, "start_end_ranges", []),
        })
    return out


def address_records(action: Any, source_markers: Any, timed_warp_samples: Any) -> list[dict[str, Any]]:
    addrs = set()
    if isinstance(source_markers, list):
        for marker in source_markers:
            if isinstance(marker, dict) and isinstance(marker.get("source_address"), int):
                addrs.add(marker["source_address"])
    if isinstance(timed_warp_samples, list):
        for sample in timed_warp_samples:
            if isinstance(sample, dict) and isinstance(sample.get("pc"), int):
                addrs.add(sample["pc"])

    records = []
    for addr in sorted(addrs):
        try:
            info = action.source_info(addr)
        except Exception:
            info = None
        records.append({
            "address": addr,
            "sass": _jsonable(getattr(action, "sass_by_pc", lambda _a: "")(addr)),
            "ptx": _jsonable(getattr(action, "ptx_by_pc", lambda _a: "")(addr)),
            "source_info": source_info_record(info),
        })
    return records


def action_record(action: Any, range_index: int, action_index: int) -> dict[str, Any]:
    raw_metric_names = _call(action, "metric_names", [])
    metric_names = raw_metric_names if isinstance(raw_metric_names, list) else []
    metrics = []
    for name in metric_names:
        metric = action.metric_by_name(name)
        if metric is not None:
            metrics.append(metric_record(metric))

    source_markers = _call(action, "source_markers", [])
    timed_warp_samples = _call(action, "timed_warp_samples", [])
    return {
        "range_index": range_index,
        "action_index": action_index,
        "name": _call(action, "name"),
        "name_demangled": _call_name_base(action, "NameBase_DEMANGLED"),
        "name_mangled": _call_name_base(action, "NameBase_MANGLED"),
        "workload_type": _call(action, "workload_type"),
        "nvtx": nvtx_record(action),
        "metrics": metrics,
        "rule_results": _call(action, "rule_results_as_dicts", []),
        "source_files": _call(action, "source_files", {}),
        "source_markers": source_markers,
        "timed_warp_samples": timed_warp_samples,
        "addresses": address_records(action, source_markers, timed_warp_samples),
    }


def export_report_json(rep_path: Path, out_dir: Path, ncu_report: Any) -> dict[str, Any]:
    meta = parse_report_name(rep_path)
    report = ncu_report.load_report(str(rep_path))
    actions = []
    for range_index in range(report.num_ranges()):
        rng = report.range_by_idx(range_index)
        for action_index in range(rng.num_actions()):
            actions.append(action_record(rng.action_by_idx(action_index), range_index, action_index))

    record = {
        "schema_version": 1,
        "source_report": str(rep_path.relative_to(ROOT) if rep_path.is_relative_to(ROOT) else rep_path),
        "source_report_sha256": _sha256(rep_path),
        "source_report_bytes": rep_path.stat().st_size,
        **meta,
        "num_ranges": report.num_ranges(),
        "num_actions": len(actions),
        "actions": actions,
    }

    op_dir = out_dir / str(meta["op"])
    op_dir.mkdir(parents=True, exist_ok=True)
    out_path = op_dir / f"{meta['stem']}.json"
    out_path.write_text(json.dumps(record, indent=2, default=_jsonable))
    return {
        "op": meta["op"],
        "backend": meta["backend"],
        "dtype": meta["dtype"],
        "report": record["source_report"],
        "json": str(out_path.relative_to(out_dir)),
        "sha256": record["source_report_sha256"],
        "bytes": record["source_report_bytes"],
        "num_actions": record["num_actions"],
    }


def export_cli_pages(rep_path: Path, out_dir: Path) -> dict[str, Any]:
    meta = parse_report_name(rep_path)
    pages_dir = out_dir / str(meta["op"]) / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for page in ("raw", "details", "source"):
        out_path = pages_dir / f"{meta['stem']}.{page}.csv"
        cmd = [ncu_bin(), "--import", str(rep_path), "--csv", "--page", page]
        run = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        out_path.write_text(run.stdout)
        outputs[page] = {
            "path": str(out_path.relative_to(out_dir)),
            "returncode": run.returncode,
            "stderr_tail": run.stderr[-2000:],
        }
    return outputs


def parse_cli_rules(details_csv: Path) -> list[dict[str, Any]]:
    try:
        with details_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    rules = []
    for row in rows:
        if not row.get("Rule Name"):
            continue
        rules.append({
            "section": row.get("Section Name"),
            "rule_name": row.get("Rule Name"),
            "rule_type": row.get("Rule Type"),
            "description": row.get("Rule Description"),
            "estimated_speedup_type": row.get("Estimated Speedup Type"),
            "estimated_speedup": row.get("Estimated Speedup"),
        })
    return rules


def enrich_json_with_cli_pages(out_dir: Path, entry: dict[str, Any]) -> None:
    report_json = out_dir / entry["json"]
    try:
        record = json.loads(report_json.read_text())
    except Exception:
        return
    record["cli_pages"] = entry.get("cli_pages", {})
    actions = record.get("actions") or []
    details_rel = entry.get("cli_pages", {}).get("details", {}).get("path")
    source_rel = entry.get("cli_pages", {}).get("source", {}).get("path")
    rules = parse_cli_rules(out_dir / details_rel) if details_rel else []
    if actions:
        actions[0]["cli_details_page"] = details_rel
        actions[0]["cli_source_page"] = source_rel
        actions[0]["cli_rule_results"] = rules
        if not actions[0].get("rule_results"):
            actions[0]["rule_results"] = rules
    report_json.write_text(json.dumps(record, indent=2, default=_jsonable))


def iter_reports(args: argparse.Namespace) -> list[Path]:
    if args.reports:
        paths = []
        for raw in args.reports:
            matches = sorted(Path(p) for p in glob.glob(raw))
            paths.extend(matches or [Path(raw)])
        return [p.resolve() for p in paths]
    return sorted(args.ncu_dir.resolve().glob("*/*.ncu-rep"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncu-dir", type=Path, default=NCU_DIR)
    ap.add_argument("--out-dir", type=Path, default=ncu_json_dir())
    ap.add_argument("--reports", nargs="*", help="Specific .ncu-rep paths or globs.")
    ap.add_argument("--no-cli-pages", action="store_true",
                    help="Skip raw/details/source CSV dumps from `ncu --import`.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    reports = iter_reports(args)
    if args.limit:
        reports = reports[:args.limit]
    if not reports:
        raise SystemExit(f"no .ncu-rep files found under {args.ncu_dir}")

    ncu_report = import_ncu_report()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, rep_path in enumerate(reports, start=1):
        print(f"[{i}/{len(reports)}] {rep_path}")
        entry = export_report_json(rep_path, args.out_dir, ncu_report)
        if not args.no_cli_pages:
            entry["cli_pages"] = export_cli_pages(rep_path, args.out_dir)
            enrich_json_with_cli_pages(args.out_dir, entry)
        manifest.append(entry)

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "reports": manifest}, indent=2))
    print(f"wrote {manifest_path} ({len(manifest)} reports)")


if __name__ == "__main__":
    main()
