import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/opt/nvidia/nsight-compute/2026.1.1/extras/python")
sys.path.insert(0, str(Path("tools").resolve()))
import ncu_report
from sass_listing import parse_listing

for backend in ("tilelang", "triton", "cutile"):
    path = f"evidence/{backend}_fp32.ncu-rep"
    report = ncu_report.load_report(str(Path(path).resolve()))
    action = report.range_by_idx(0).action_by_idx(0)
    print(backend, [n for n in dir(action) if any(s in n for s in ("sass", "ptx", "source", "rule"))])
    rules = action.rule_results_as_dicts()
    Path(f"output/rules_{backend}.json").write_text(json.dumps(rules, indent=2, default=str) + "\n")
    pcs = set()
    for name in action.metric_names():
        m = action[name]
        if m.has_correlation_ids():
            c = m.correlation_ids()
            for i in range(m.num_instances()):
                try:
                    pc = c.as_uint64(i)
                    if action.sass_by_pc(pc):
                        pcs.add(pc)
                except (RuntimeError, TypeError):
                    pass
    for pc in range(min(pcs), max(pcs) + 16, 16):
        if action.sass_by_pc(pc):
            pcs.add(pc)
    sass = {pc: action.sass_by_pc(pc) for pc in pcs}
    text = f"Kernel Name: {action.name()}\n" + "\n".join(f"{pc:#x} {line}" for pc, line in sorted(sass.items())) + "\n"
    Path(f"output/sass_{backend}.txt").write_text(text)
    parsed = parse_listing(text, strict=True)
    Path(f"output/sass_counts_{backend}.json").write_text(json.dumps(parsed, indent=2) + "\n")
    print(backend, "SASS", len(sass), "opcodes", parsed["kernels"][0]["opcodes"])
    ptx = {pc: action.ptx_by_pc(pc) for pc in pcs if action.ptx_by_pc(pc)}
    Path(f"output/ptx_{backend}.txt").write_text("\n".join(f"{pc:#x} {line}" for pc, line in sorted(ptx.items())) + "\n")
    print(backend, "PTX", len(ptx), "PM names", [n for n in action.metric_names() if n.startswith("pmsampling:")])
    hotspots = []
    for name in action.metric_names():
        if not name.startswith("smsp__pcsamp_warps_issue_stalled_") or name.endswith((".sum", ".avg", ".max", ".min")):
            continue
        metric = action[name]
        if not metric.has_correlation_ids():
            continue
        corr = metric.correlation_ids()
        for i in range(metric.num_instances()):
            value = metric.as_uint64(i)
            if not value:
                continue
            pc = corr.as_uint64(i)
            info = action.source_info(pc)
            hotspots.append({"metric": name, "samples": value, "pc": hex(pc), "sass": sass.get(pc),
                             "file": info.file_name() if info else None, "line": info.line() if info else None})
    Path(f"output/stall_hotspots_{backend}.json").write_text(json.dumps(sorted(hotspots, key=lambda r:r["samples"], reverse=True), indent=2) + "\n")
    print(backend, "hotspots", len(hotspots))
    Path(f"output/source_files_{backend}.json").write_text(json.dumps(dict(action.source_files()), indent=2) + "\n")
