"""Export exact scalar metrics and complete action inventories from saved NCU."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def extract(report, report_path, requested=None):
    result = {"report": str(report_path), "actions": [], "metrics": [], "errors": []}
    for ri in range(report.num_ranges()):
        current = report.range_by_idx(ri)
        for ai in range(current.num_actions()):
            action = current.action_by_idx(ai)
            names = list(action.metric_names())
            identity = {"report": str(report_path), "range_index": ri,
                        "action_index": ai, "kernel": action.name()}
            result["actions"].append({**identity, "metric_names": names})
            selected = names if requested is None else list(dict.fromkeys(requested))
            for name in selected:
                record = {**identity, "metric": name}
                if name not in names:
                    result["errors"].append({**record, "error": "metric_not_collected"})
                    continue
                try:
                    metric = action[name]
                    value = metric.value()
                    if not isinstance(value, (int, float, str, bool)):
                        raise TypeError(f"Unsupported scalar type: {type(value).__name__}")
                    if isinstance(value, float) and not math.isfinite(value):
                        raise ValueError("Nonfinite metric value")
                    result["metrics"].append({**record, "value": value, "unit": metric.unit()})
                except (RuntimeError, TypeError, ValueError) as error:
                    result["errors"].append({**record, "error": str(error)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--metric", action="append", help="Exact name; repeat to select")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve():
        parser.error("Output must not overwrite the input report")
    try:
        import ncu_report
    except ImportError:
        parser.error("NCU Python report API unavailable; configure the workspace-approved module path")
    report = ncu_report.load_report(str(args.report.resolve()))
    result = extract(report, args.report, args.metric)
    digest = hashlib.sha256()
    with args.report.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    result["report_sha256"] = digest.hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "actions": len(result["actions"]),
                      "metrics": len(result["metrics"]), "errors": len(result["errors"])}))


if __name__ == "__main__":
    main()
