"""Status table, pilot table and the local Markdown report of a diagnostics run.

The benchmark report uses the native benchmark modes only (``benchmark_records``): native
PyTorch eager vs native NKI. xla_* rows are legacy diagnostics; they appear only in the
separate legacy report (``write_report(store, legacy_xla=True)``). native_torch_compiled rows
(torch.compile) are a diagnostic too. Neither enters a benchmark coverage count, speedup or
table. Speedups are only formed inside one software
stack and one timing method (``schema.comparable``).
"""
from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

from tilebench.neuron_diag.schema import (BENCHMARK_MODES, EXECUTION_MODES, LEGACY_DIAGNOSTIC_MODES,
                                          STATUSES, Record, speedup)

_MODE_ORDER = list(EXECUTION_MODES)


class DuplicateValidRows(RuntimeError):
    pass


def latest_records(rows: list[dict]) -> dict[tuple, dict]:
    """The one valid row per (operator, case_id, mode). Rows marked invalid or superseded
    (``valid_for_analysis`` false, see store.row_status) are excluded explicitly; if more
    than one valid row remains for a key the report refuses to pick one."""
    best: dict[tuple, dict] = {}
    dups = []
    for r in rows:
        if r.get("valid_for_analysis") is False:
            continue
        k = (r["operator"], r["case_id"], r["mode"])
        if k in best:
            dups.append(k)
        best[k] = r
    if dups:
        raise DuplicateValidRows(f"{len(dups)} keys have more than one valid row (first: {dups[0]}); "
                                 "mark the stale rows with RunStore.mark_invalid")
    return best


def benchmark_records(rows: list[dict]) -> dict[tuple, dict]:
    """The rows a TileBench++ Trn2 benchmark result may use: valid native_* rows only."""
    return latest_records([r for r in rows if r["mode"] in BENCHMARK_MODES])


def legacy_xla_records(rows: list[dict]) -> dict[tuple, dict]:
    """Archived legacy XLA diagnostic rows (xla_*), for the legacy report only."""
    return latest_records([r for r in rows if r["mode"] in LEGACY_DIAGNOSTIC_MODES])


def _fmt(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _median(xs):
    return statistics.median(xs) if xs else None


def coverage(latest: dict) -> dict:
    cov: dict = collections.defaultdict(collections.Counter)
    for (op, cid, mode), r in latest.items():
        cov[mode][r["status"]] += 1
    return cov


def host_read_only(r: dict) -> bool:
    """A 'cpu_fallback' whose only evidence is device->host scalar reads (rows written
    before runtime.py separated them)."""
    import re

    from tilebench.neuron_diag.runtime import HOST_READ_COUNTERS

    if r.get("status") != "cpu_fallback":
        return False
    names = re.findall(r"'(aten::[^']+)'", r.get("fallback_evidence", "").split(";")[0])
    return bool(names) and all(n in HOST_READ_COUNTERS for n in names)


def status_table(latest: dict) -> list[str]:
    ops = sorted({k[0] for k in latest})
    modes = [m for m in _MODE_ORDER if any(k[2] == m for k in latest)]
    lines = ["| operator | case | dtype | " + " | ".join(modes) + " |",
             "|---|---|---|" + "---|" * len(modes)]
    for op in ops:
        cids = sorted({k[1] for k in latest if k[0] == op})
        for cid in cids:
            dtype = next((latest[(op, cid, m)].get("dtype", "") for m in modes if (op, cid, m) in latest), "")
            cells = []
            for m in modes:
                r = latest.get((op, cid, m))
                if r is None:
                    cells.append("not planned")
                    continue
                c = r["status"]
                if r.get("fallback_status") in ("undetermined", "unable_to_determine") and c == "pass":
                    c += " (fallback?)"
                if host_read_only(r):
                    c += " (host scalar read only)"
                cells.append(c)
            lines.append(f"| {op} | {cid} | {dtype} | " + " | ".join(cells) + " |")
    return lines


def pilot_table(latest: dict, operators, stacks=(("native", "native_torch_eager", "native_nki"),)) -> list[str]:
    lines = ["| operator | case | dtype | shape | stack | torch wall ms | NKI wall ms | wall speedup | "
             "torch device ms | NKI device ms | device speedup | NKI exec/iter | NKI pcores/exec | fallback (torch/NKI) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for op in operators:
        cids = sorted({k[1] for k in latest if k[0] == op})
        for cid in cids:
            for stack, tmode, nmode in stacks:
                t, n = latest.get((op, cid, tmode)), latest.get((op, cid, nmode))
                if t is None and n is None:
                    continue
                rt = Record.from_json(t) if t else None
                rn = Record.from_json(n) if n else None
                ws = speedup(rt, rn, "wall_ms") if rt and rn else None
                ds = speedup(rt, rn, "device_ms") if rt and rn else None
                dev = ((n or {}).get("compile_info") or {}).get("device") or {}
                ivs = dev.get("per_iteration_intervals") or []
                pc = sorted({p for iv in ivs for p in iv.get("pcores_per_execution", [])})
                any_r = t or n
                lines.append(
                    f"| {op} | {cid} | {any_r.get('dtype')} | {json.dumps(any_r.get('shape'))} | {stack} | "
                    f"{_fmt((t or {}).get('wall_ms'))} | {_fmt((n or {}).get('wall_ms'))} | {_fmt(ws, 2)} | "
                    f"{_fmt((t or {}).get('device_ms'))} | {_fmt((n or {}).get('device_ms'))} | {_fmt(ds, 2)} | "
                    f"{_fmt(dev.get('executions_per_iteration'))} | {pc or '—'} | "
                    f"{(t or {}).get('fallback_status', '—')}/{(n or {}).get('fallback_status', '—')} |")
    return lines


def failures(latest: dict) -> list[str]:
    lines = ["| operator | case | mode | status | reason / error |", "|---|---|---|---|---|"]
    for (op, cid, mode), r in sorted(latest.items()):
        if r["status"] in ("pass",):
            continue
        why = (r.get("reason") or "") + (" " + r["error"] if r.get("error") else "")
        why = why.replace("|", "/").replace("\n", " ")[:300]
        lines.append(f"| {op} | {cid} | {mode} | {r['status']} | {why} |")
    return lines


def write_report(store, *, legacy_xla: bool = False) -> Path:
    """report.md: the native benchmark view. With ``legacy_xla`` the archived XLA rows go to
    legacy_xla/report.md instead, under a banner; the two views never share a table."""
    from tilebench.neuron_diag.cases import PILOT_OPERATORS

    rows = store.records()
    if legacy_xla:
        latest = legacy_xla_records(rows)
        md = ["# LEGACY XLA diagnostics — archived, not a TileBench++ benchmark result", "",
              f"Run directory: `{store.dir}`", "",
              "PyTorch/XLA rows (xla_torch, xla_nki), kept only to explain the NKI host overhead "
              "seen on the old stack. They are excluded from every benchmark aggregate, speedup, "
              "coverage count, table, figure and classification.", ""]
        stacks = (("xla", "xla_torch", "xla_nki"),)
    else:
        if getattr(store, "archived", False):
            raise ValueError(f"{store.dir} is an archived legacy XLA run: only the legacy report exists")
        latest = benchmark_records(rows)
        md = ["# TileBench++ Trn2 native benchmark — local report", "",
              f"Run directory: `{store.dir}`", "",
              "Native stack only: PyTorch eager vs NKI. Legacy XLA rows (see "
              "legacy_xla/) and torch.compile diagnostic rows are excluded.", "",
              "Local and private: this file and everything next to it stay out of Git and out of "
              "any PR.", ""]
        stacks = (("native", "native_torch_eager", "native_nki"),)
        notes = store.dir / "notes.md"
        if notes.is_file():
            md += [notes.read_text(), ""]
    cov = coverage(latest)
    env_dir = store.dir / "env"
    wanted_env = "xla" if legacy_xla else "native"
    for p in sorted(env_dir.glob("*.json")) if env_dir.is_dir() else []:
        snap = json.loads(p.read_text())
        if snap.get("stack", p.stem) != wanted_env:
            continue
        md += [f"## Environment `{p.stem}`", ""] + env_summary(snap) + [""]
    mans = sorted((store.dir / "sources").glob("manifest*.json"),
                  key=lambda p: int(p.stem.split(".v")[1]) if ".v" in p.stem else 1) \
        if (store.dir / "sources").is_dir() else []
    man = mans[-1] if mans else store.dir / "sources" / "manifest.json"
    if man.is_file():
        manifest = json.loads(man.read_text())
        md += ["## Source / compatibility inventory (static scan: presence in source, not device cost)", "",
               f"main commit `{manifest['main_commit'][:10]}`; framework hash "
               f"`{manifest['framework_sha256'][:12]}`", ""] + inventory_table(manifest) + [""]
    md += ["## Coverage (latest record per operator/case/mode)", "",
           "| mode | " + " | ".join(s for s in STATUSES) + " |",
           "|---|" + "---|" * len(STATUSES)]
    for m in _MODE_ORDER:
        if m in cov:
            md.append(f"| {m} | " + " | ".join(str(cov[m].get(s, 0)) for s in STATUSES) + " |")
    md += ["", "## Pilot operators", ""] + pilot_table(latest, PILOT_OPERATORS, stacks)
    md += ["", "Speedups are torch/NKI inside one stack and one timing method only. "
           "`device ms` is the per-iteration busy sum of attributable device executions "
           "(native: torch.profiler NeuronConfig(RUNTIME) trace; legacy XLA: runtime inspect); "
           "it is null when no attributable device trace exists.", ""]
    if legacy_xla and man.is_file():
        md += ["## Historical reference (PR-branch CSV, not a controlled comparison)", "",
               "Committed NKI columns of each PR head for the same cases. Different run "
               "conditions (branch config warmup/repeat, possibly earlier code); shown only to "
               "check whether this run is in the same range.", ""]
        md += historical_reference(json.loads(man.read_text()), latest, PILOT_OPERATORS) + [""]
    md += ["## Status of every planned case", ""] + status_table(latest)
    md += ["", "## Non-passing cases", ""] + failures(latest)
    path = store.path("legacy_xla", "report.md") if legacy_xla else store.path("report.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(md) + "\n")
    tmp.replace(path)  # the report is derived data; regenerating it replaces only itself
    return path


def inventory_table(manifest: dict) -> list[str]:
    """A1 source/compatibility inventory, from the static scan (source presence only)."""
    lines = ["| operator | PR | commit | layout | kernels | launch sites in run() (static) | LNC | "
             "program_id split | device loops | wrapper ops in run() (static) | legacy XLA code nki/torch | "
             "notes |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in manifest["operators"]:
        if e["status"] != "found":
            lines.append(f"| {e['operator']} | — | — | — | — | — | — | — | — | — | — | no NKI impl: {e.get('reason', '')} |")
            continue
        n = e["impl_nki"]
        lnc = n["lnc"]
        lnc_s = ("helper _lnc_degree()" if lnc["lnc_helper"] else
                 f"hard-coded {lnc['hardcoded_num_cores']}" if lnc["hardcoded_num_cores"] else "no LNC helper or constant in source")
        notes = []
        if e["layout"] == "old":
            notes.append("old layout, imports patched")
        if n["module_level_unguarded_jit"]:
            notes.append("@nki.jit not guarded by `nki is None`")
        if e["cross_operator_imports"]:
            notes.append(f"imports {e['cross_operator_imports']}")
        if e["impl_torch"]["xla_dependencies"]:
            notes.append("impl_torch keeps a legacy XLA compatibility branch (native runs its neuron/generic path)")
        if e["impl_torch"]["uses_torch_compile"]:
            notes.append("impl_torch uses torch.compile")
        if n["declared_unsupported"]:
            notes.append("declares unsupported: " + "; ".join(x.strip()[:60] for x in n["declared_unsupported"][:2]))
        if e.get("generator", {}).get("branch_function_differs"):
            notes.append("branch generator differs (main used)")
        pr = e.get("pr") or {}
        lines.append(
            f"| {e['operator']} | #{pr.get('number', '?')} | {e['commit'][:8]} | {e['layout']} | "
            f"{', '.join(n['kernels']) or '(imported)'} | {', '.join(n['launch_sites_in_run_static']) or '—'} | "
            f"{lnc_s} | {'yes' if (lnc['program_id'] or lnc['num_programs']) else 'no'} | "
            f"{', '.join(n['device_loops']) or '—'} | "
            f"{', '.join(f'{k}×{v}' for k, v in n['wrapper_ops_in_run_static'].items()) or '—'} | "
            f"{', '.join(n['xla_dependencies']) or '—'} / {', '.join(e['impl_torch']['xla_dependencies']) or '—'} | "
            f"{'; '.join(notes).replace('|', '/')} |")
    return lines


def env_summary(snap: dict) -> list[str]:
    n = snap.get("neuron") or {}
    pk = snap.get("packages") or {}
    return [f"- stack label: `{snap.get('stack')}`; python {snap.get('python')} ({snap.get('platform')})",
            f"- packages: " + ", ".join(f"{k} {v}" for k, v in sorted(pk.items())),
            f"- instance: {n.get('instance_type')}; logical NeuronCore config {n.get('logical_neuroncore_config')}; "
            f"driver module {n.get('driver_module_version')}",
            f"- Neuron packages: " + ", ".join(n.get("neuron_packages") or []),
            f"- other processes on the device at snapshot time: {snap.get('neuron_processes') or 'none'}",
            f"- container: {snap.get('container')}"]


def historical_reference(manifest: dict, latest: dict, operators) -> list[str]:
    """NKI columns committed on each PR branch for the pilot cases (measured on the legacy
    XLA path). Legacy report only: other run conditions (warmup/repeat of the branch config,
    earlier code), never a controlled A/B with this run."""
    import csv
    import io
    import subprocess

    from tilebench.paths import REPO_ROOT

    entries = {e["operator"]: e for e in manifest["operators"]}
    lines = ["| operator | case | dtype | params | branch commit | branch warmup/repeat | torch_nki_ms | nki_ms | speedup_nki |",
             "|---|---|---|---|---|---|---|---|---|"]
    for op in operators:
        e = entries.get(op)
        if not e or e["status"] != "found":
            continue
        text = None
        for path in (f"results/B200/csv/{op}_default.csv", f"results/csv/{op}_default.csv"):
            r = subprocess.run(["git", "show", f"{e['commit']}:{path}"], cwd=REPO_ROOT,
                               capture_output=True, text=True)
            if r.returncode == 0:
                text = r.stdout
                break
        if text is None:
            continue
        rows = list(csv.DictReader(io.StringIO(text)))
        bench = (e.get("config") or {}).get("benchmark") or {}
        for cid in sorted({k[1] for k in latest if k[0] == op}):
            rec = next((latest[k] for k in latest if k[0] == op and k[1] == cid), None)
            if rec is None:
                continue
            shape = rec.get("shape") or {}
            want = ",".join(f"{k}={v}" for k, v in shape.items())
            hit = [row for row in rows if row.get("dtype") == rec.get("dtype")
                   and all(f"{k}={v}" in row.get("params", "") for k, v in shape.items())]
            for row in hit[:1]:
                lines.append(f"| {op} | {cid} | {rec.get('dtype')} | {row.get('params')} | {e['commit'][:8]} | "
                             f"{bench.get('warmup')}/{bench.get('repeat')} | {row.get('torch_nki_ms', '—')} | "
                             f"{row.get('nki_ms', '—')} | {row.get('speedup_nki', '—')} |")
    return lines
