"""Export per-instruction profiler listings for the dashboard agent.

scripts/export_profiles.py keeps each kernel's 40 hottest instructions. This
writes the whole SASS listing of every kernel in address order with what the
agent needs to locate a mechanism: execution count, profiler samples, the stall
reasons behind those samples and the source line of each instruction, plus
per-kernel roll-ups (stall totals, samples by source line, dynamic opcode
totals, and how many global loads a thread issues before it first waits).

    python scripts/export_listings.py --reports <dir> [--reports <dir> ...] [--only relu]

<dir> holds NVIDIA_<platform>/<op>/<backend>_<dtype>.ncu-rep. Needs Nsight
Compute's Python module (<nsight-compute>/extras/python on PYTHONPATH); no GPU.

It also reads AMD_<platform>/<op>/<backend>_<dtype>/analysis/pc_sampling_instructions.csv
(rocprof-compute). Those captures list only the instructions that received a
sample and carry no execution counts.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STALLS = ["long_scoreboard", "short_scoreboard", "wait", "barrier", "membar", "mio_throttle", "lg_throttle",
          "math_pipe_throttle", "tex_throttle", "not_selected", "no_instructions", "branch_resolving",
          "dispatch_stall", "drain", "imc_miss", "sleeping", "other"]


def instances(metric):
    ids = metric.correlation_ids()
    return {ids.as_uint64(i): metric.as_uint64(i) for i in range(metric.num_instances())}


def kernel_listing(action):
    names = set(action.metric_names())
    if "inst_executed" not in names:
        return {"name": action.name(), "samples": 0, "instructions": [],
                "note": "reduced capture: this report has no per-instruction data"}
    executed = instances(action["inst_executed"])
    samples = instances(action["smsp__pcsamp_sample_count"]) if "smsp__pcsamp_sample_count" in names else {}
    stalls = {}
    for reason in STALLS:
        name = f"smsp__pcsamp_warps_issue_stalled_{reason}"
        if name in names:
            for pc, value in instances(action[name]).items():
                if value:
                    stalls.setdefault(pc, {})[reason] = int(value)
    pcs = sorted(set(executed) | set(samples))
    base = pcs[0] if pcs else 0
    rows, totals, by_line, opcodes, chain = [], {}, {}, {}, []
    for pc in pcs:
        sass = action.sass_by_pc(pc).strip()
        info = action.source_info(pc)
        source = f"{Path(info.file_name()).name}:{info.line()}" if info and info.line() else None
        count, hits, reasons = int(executed.get(pc, 0)), int(samples.get(pc, 0)), stalls.get(pc, {})
        tokens = [t for t in sass.split() if not t.startswith("@")]
        opcode = tokens[0].split(".")[0] if tokens else "?"
        opcodes[opcode] = opcodes.get(opcode, 0) + count
        for reason, value in reasons.items():
            totals[reason] = totals.get(reason, 0) + value
        if source:
            line = by_line.setdefault(source, {"samples": 0, "executed": 0, "stalls": {}})
            line["samples"] += hits
            line["executed"] += count
            for reason, value in reasons.items():
                line["stalls"][reason] = line["stalls"].get(reason, 0) + value
        offset = f"{pc - base:#06x}"
        rows.append({"offset": offset, "executed": count, "samples": hits, "stalls": reasons, "source": source, "sass": sass})
        if tokens and tokens[0].startswith("LDG"):
            chain.append(("load", offset, tokens[0], count, 0))
        elif reasons.get("long_scoreboard"):
            chain.append(("wait", offset, tokens[0] if tokens else "?", count, reasons["long_scoreboard"]))
    total = sum(r["samples"] for r in rows)
    busiest = max((c[3] for c in chain), default=0)
    hot = [c for c in chain if c[3] * 4 >= busiest and (c[0] == "load" or c[4] >= max(20, 0.002 * total))]
    runs, current = [], 0
    for kind, *_ in hot:
        if kind == "load":
            current += 1
        elif current:
            runs.append(current)
            current = 0
    lines = sorted(by_line.items(), key=lambda kv: -kv[1]["samples"])[:20]
    return {
        "name": action.name(), "samples": total, "static_instructions": len(rows),
        "stall_samples": dict(sorted(totals.items(), key=lambda kv: -kv[1])),
        "samples_by_source_line": [{"source": k, **v} for k, v in lines],
        "opcode_executions": dict(sorted(opcodes.items(), key=lambda kv: -kv[1])[:40]),
        "load_order": {"max_loads_before_a_wait": max(runs) if runs else None, "load_then_wait_groups": len(runs),
                       "sites": [{"kind": k, "offset": o, "op": op, "executed": n, **({"long_scoreboard": w} if k == "wait" else {})}
                                 for k, o, op, n, w in hot[:60]]},
        "instructions": rows,
    }


def amd_listing(path):
    csv.field_size_limit(10 ** 9)
    kernels = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["operator_kernel"] == "True":
                kernels.setdefault(row["kernel_name"], []).append(row)
    out = []
    for name, found in kernels.items():
        rows, totals, by_line = [], {}, {}
        for row in sorted(found, key=lambda r: int(r["offset"], 16)):
            reasons = {k: int(v) for k, v in json.loads(row["stall_reasons"] or "{}").items()}
            source = ":".join((Path(row["source_line"].rsplit(":", 1)[0]).name, row["source_line"].rsplit(":", 1)[1])) \
                if ":" in row["source_line"] else None
            hits = int(row["samples"])
            for reason, value in reasons.items():
                totals[reason] = totals.get(reason, 0) + value
            if source:
                line = by_line.setdefault(source, {"samples": 0, "stalls": {}})
                line["samples"] += hits
                for reason, value in reasons.items():
                    line["stalls"][reason] = line["stalls"].get(reason, 0) + value
            rows.append({"offset": row["offset"], "executed": None, "samples": hits, "issued": int(row["issued"]),
                         "stalls": reasons, "source": source, "sass": row["instruction"]})
        lines = sorted(by_line.items(), key=lambda kv: -kv[1]["samples"])[:20]
        out.append({
            "name": name, "samples": sum(r["samples"] for r in rows), "static_instructions": len(rows),
            "stall_samples": dict(sorted(totals.items(), key=lambda kv: -kv[1])),
            "samples_by_source_line": [{"source": k, **v} for k, v in lines],
            "instructions": rows,
            "note": "rocprof-compute PC sampling: only instructions that received a sample are listed, in address order, "
                    "with no execution counts. The complete assembly is in the intermediate code (kind amdgcn).",
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "profile_listings")
    parser.add_argument("--only", action="append", default=[], help="operator to export; repeat")
    parser.add_argument("--force", action="store_true", help="re-export reports whose listing already exists")
    args = parser.parse_args()
    found = {}
    for base in args.reports:
        for path in sorted(base.glob("NVIDIA_*/*/*.ncu-rep")):
            platform, op = path.parent.parent.name.replace("NVIDIA_", ""), path.parent.name
            if not args.only or op in args.only:
                found.setdefault((platform, op, path.stem), path)
        for path in sorted(base.glob("AMD_*/*/*/analysis/pc_sampling_instructions.csv")):
            report = path.parent.parent
            platform, op = report.parent.parent.name.replace("AMD_", ""), report.parent.name
            if not args.only or op in args.only:
                found.setdefault((platform, op, report.name), path)
    if any(path.suffix == ".ncu-rep" for path in found.values()):
        try:
            import ncu_report
        except ImportError as error:
            parser.error(f"Nsight Compute's Python module is not importable: {error}")
    manifest_path = args.out / "manifest.json"
    previous = json.loads(manifest_path.read_text())["reports"] if manifest_path.exists() else []
    entries = {(e["platform"], e["op"], e["backend"], e["dtype"]): e for e in previous}
    def save():
        args.out.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps({"reports": [entries[k] for k in sorted(entries)]}, indent=1) + "\n")

    def summary(platform, op, backend, dtype, stem, kernels):
        return {"platform": platform, "op": op, "backend": backend, "dtype": dtype, "file": f"{platform}/{op}/{stem}.json.gz",
                "kernels": [{"name": k["name"], "samples": k["samples"], "instructions": len(k["instructions"])} for k in kernels]}

    for (platform, op, stem), path in sorted(found.items()):
        backend, _, dtype = stem.partition("_")
        existing = args.out / platform / op / f"{stem}.json.gz"
        if existing.exists() and not args.force:
            if (platform, op, backend, dtype) not in entries:
                kernels = json.loads(gzip.decompress(existing.read_bytes()))["kernels"]
                entries[(platform, op, backend, dtype)] = summary(platform, op, backend, dtype, stem, kernels)
            continue
        try:
            if path.suffix == ".csv":
                kernels = amd_listing(path)
            else:
                report = ncu_report.load_report(str(path.resolve()))
                kernels = [kernel_listing(report.range_by_idx(r).action_by_idx(a))
                           for r in range(report.num_ranges()) for a in range(report.range_by_idx(r).num_actions())]
        except Exception as error:
            print(f"FAILED {platform}/{op}/{stem}: {error}")
            continue
        target = args.out / platform / op
        target.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"platform": platform, "op": op, "backend": backend, "dtype": dtype, "kernels": kernels},
                             separators=(",", ":")).encode()
        (target / f"{stem}.json.gz").write_bytes(gzip.compress(payload, mtime=0))
        entries[(platform, op, backend, dtype)] = summary(platform, op, backend, dtype, stem, kernels)
        save()
        print(f"ok {platform}/{op}/{stem}: {len(kernels)} kernels, {sum(len(k['instructions']) for k in kernels)} instructions")
    save()
    print(f"{len(entries)} reports in {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
