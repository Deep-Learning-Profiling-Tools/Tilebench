"""ROCm Compute Profiler (rocprof-compute) support: environment, kernel
selection and capture validation for the AMD profiling harness.

Library only. The command line lives in scripts/profiling/rocprof_compute_driver.py;
the profiled process is scripts/profiling/rocprof_compute_harness.py.

One profiled pair is outputs/rocprof_compute/<gpu>/<op>/<backend>_<dtype>/
(tilebench.paths.rocprof_compute_pair_dir):

    workload/      the complete directory `rocprof-compute profile` wrote: the
                   canonical artifact, never edited (analysis runs on a copy)
    pc_sampling/   the PC-sampling pass, a workload of its own (rocprof-compute
                   3.7 collects either counters or PC samples in one profile)
    analysis/      output derived from them: `rocprof-compute analyze` text
                   reports and CSV exports, and the per-instruction PC-sampling
                   table (ISA, source line, stall reasons)
    logs/          the prime, profile and PC-sampling runs' console output
    capture.json   the capture validation record

Facts about rocprof-compute 3.7.0 on MI300X this module relies on (measured):
  - `-k REGEX` selects dispatches by kernel name, matched from the start of the
    name. A regex with a `$` silently disables the filter (every dispatch is
    profiled), so the selection is start-anchored only and the capture is
    validated against the exact names afterwards.
  - Each counter pass is a separate run of the workload and leaves one
    results_<pass>.csv in the workload, one row per (dispatch, counter).
  - The PC-sampling pass samples every dispatch of the process (the kernel
    filter does not apply); its ps_file_results.json correlates each sample
    with a dispatch, so the operator's samples are told apart afterwards.
"""
import csv
import json
import os
import re
from pathlib import Path

from tilebench.profiling import ncu_kernel_select as ks

#: Files `rocprof-compute profile` writes into every counter workload
#: (perfmon/ is a directory). A workload missing one is incomplete.
WORKLOAD_REQUIRED = ("profiling_config.yaml", "sysinfo.csv", "log.txt", "perfmon")
#: The PC-sampling pass writes its samples here, in its own workload directory.
PC_SAMPLING_RESULTS = "ps_file_results.json"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def rocm_root(tool) -> Path:
    """The ROCm installation that ships the rocprof-compute launcher `tool`
    (<root>/bin/rocprof-compute), symlinks resolved."""
    return Path(os.path.realpath(tool)).parent.parent


def profiler_env(base: dict, tool) -> dict:
    """Environment for rocprof-compute and the workload it starts.

    - <root>/lib is appended to LD_LIBRARY_PATH, as rocprofv3 itself does. A
      workload that brings its own HIP/HSA runtime (the PyTorch ROCm wheel
      ships libamdhip64/libhsa-runtime64 in torch/lib) otherwise fails counter
      collection with `aqlprofile API table load failed` and aborts: only the
      versioned libhsa-amd-aqlprofile64.so.1 is in the linker cache. Appended,
      so torch and Triton keep the runtime they load from torch/lib.
    - ROCM_VER is set from <root>/.info/version when the caller has not set
      it: rocprof-compute reads the version from $ROCM_PATH/.info/ (default
      /opt/rocm) and falls back to ROCM_VER; a multi-version layout
      (/opt/rocm/core-<ver>/) has no /opt/rocm/.info/."""
    env = dict(base)
    root = rocm_root(tool)
    lib = str(root / "lib")
    parts = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
    if lib not in parts:
        parts.append(lib)
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    version = root / ".info" / "version"
    if not env.get("ROCM_VER") and version.is_file():
        env["ROCM_VER"] = version.read_text().strip()
    return env


def kernel_filter_regex(names) -> str:
    """`rocprof-compute profile -k` regex selecting only these kernels.

    Start-anchored, never end-anchored (a `$` disables the filter, see the
    module docstring). The operator kernels are Triton functions, so their
    names are identifiers; anything else is refused rather than escaped."""
    unique = sorted(set(names or []))
    if not unique:
        raise ValueError("no kernel names to select")
    bad = [n for n in unique if not _IDENTIFIER.fullmatch(n)]
    if bad:
        raise ValueError(f"kernel names are not plain identifiers: {bad}")
    return "^(?:" + "|".join(unique) + ")"


def pass_files(workload_dir) -> list[Path]:
    """The per-pass counter results of a workload, one file per run."""
    return sorted(Path(workload_dir).glob("results_*.csv"))


def dispatch_sequence(results_csv) -> list[str]:
    """Kernel names of the dispatches in one pass file, in launch order.

    The file has one row per (dispatch, counter); a dispatch is counted once
    per Dispatch_ID, and repeated launches of one kernel stay separate."""
    first_seen: dict[str, tuple[int, int, str]] = {}
    with open(results_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            did = row["Dispatch_ID"]
            if did not in first_seen:
                first_seen[did] = (int(row["Start_Timestamp"]), int(did), row["Kernel_Name"])
    return [name for _, _, name in sorted(first_seen.values())]


def sequence_problems(expected: list[str], captured: list[str]) -> list[str]:
    """Why `captured` is not exactly the expected compute-kernel sequence.

    Empty when it is: non-empty, no aux kernel (ATen helper, memcpy/memset,
    runtime blit, generator, cache eviction), only expected names, the same
    launch count (repeats counted, never deduplicated) and the same order."""
    problems = []
    if not captured:
        problems.append("nothing captured")
        return problems
    aux = sorted({n for n in captured if ks.is_aux_kernel(n)})
    if aux:
        problems.append(f"non-operator kernels captured: {[n[:80] for n in aux]}")
    unexpected = sorted(set(captured) - set(expected))
    if unexpected:
        problems.append(f"unexpected kernels: {[n[:80] for n in unexpected]}")
    if len(captured) != len(expected):
        problems.append(f"launch-count mismatch: captured {len(captured)} != expected {len(expected)}")
    elif captured != expected:
        i = next(i for i, (a, b) in enumerate(zip(captured, expected)) if a != b)
        problems.append(f"launch order differs at launch {i}: captured {captured[i]!r}, "
                        f"expected {expected[i]!r}")
    return problems


def validate_workload(workload_dir, expected: list[str]) -> dict:
    """Capture validation of one counter workload against the kernel manifest.

    Every pass is a separate run of the workload, and the passes are joined by
    kernel and occurrence afterwards, so EVERY pass must hold exactly the
    expected sequence of ONE impl.run()."""
    workload_dir = Path(workload_dir)
    problems = []
    missing = [f for f in WORKLOAD_REQUIRED if not (workload_dir / f).exists()]
    if missing:
        problems.append(f"workload incomplete, missing {missing}")
    files = pass_files(workload_dir)
    if not files:
        problems.append("no results_*.csv counter pass in the workload")
    per_pass, captured = {}, None
    for f in files:
        seq = dispatch_sequence(f)
        per_pass[f.name] = len(seq)
        for p in sequence_problems(expected, seq):
            problems.append(f"{f.name}: {p}")
        if captured is None:
            captured = seq
    return {"ok": not problems, "problems": problems, "passes": len(files),
            "launches_per_pass": per_pass, "captured_names": captured or []}


def load_pc_sampling(workload_dir) -> dict | None:
    """The rocprofiler-sdk tool record of a PC-sampling workload, or None."""
    path = Path(workload_dir) / PC_SAMPLING_RESULTS
    if not path.is_file():
        return None
    return json.loads(path.read_text())["rocprofiler-sdk-tool"][0]


def validate_pc_sampling(workload_dir, expected: list[str]) -> dict:
    """Validation and summary of one PC-sampling workload.

    The pass samples the whole process, so the operator's dispatches (the
    non-aux kernels, in launch order) must be exactly the expected sequence;
    the summary counts the samples of those dispatches and how many map to an
    instruction (ISA) and to a source line."""
    tool = load_pc_sampling(workload_dir)
    if tool is None:
        return {"ok": False, "problems": [f"no {PC_SAMPLING_RESULTS}"]}
    names = {k["kernel_id"]: k.get("formatted_kernel_name") or k["kernel_name"]
             for k in tool["kernel_symbols"]}
    dispatches = sorted(tool["buffer_records"]["kernel_dispatch"],
                        key=lambda d: (d["start_timestamp"], d["dispatch_info"]["dispatch_id"]))
    op_dispatch = {}
    for d in dispatches:
        name = names.get(d["dispatch_info"]["kernel_id"], "")
        if not ks.is_aux_kernel(name):
            op_dispatch[d["dispatch_info"]["dispatch_id"]] = name
    captured = list(op_dispatch.values())
    problems = sequence_problems(expected, captured)

    method = "stochastic" if tool["buffer_records"].get("pc_sample_stochastic") else (
        "host_trap" if tool["buffer_records"].get("pc_sample_host_trap") else None)
    samples = tool["buffer_records"].get(f"pc_sample_{method}", []) if method else []
    instructions = tool["strings"].get("pc_sample_instructions", [])
    comments = tool["strings"].get("pc_sample_comments", [])
    op_samples = [s for s in samples if s["record"].get("dispatch_id") in op_dispatch]
    with_isa = sum(1 for s in op_samples if s.get("inst_index", -1) < len(instructions)
                   and instructions[s["inst_index"]])
    with_source = sum(1 for s in op_samples if s.get("inst_index", -1) < len(comments)
                      and comments[s["inst_index"]])
    per_kernel: dict[str, int] = {}
    for s in op_samples:
        k = op_dispatch[s["record"]["dispatch_id"]]
        per_kernel[k] = per_kernel.get(k, 0) + 1
    # Sampling is statistical: a short kernel can get no sample at all. That
    # is not a broken capture; it only means this pair has no source/ISA
    # correlation, which the caller records.
    return {"ok": not problems, "problems": problems, "method": method,
            "correlation_available": with_isa > 0 and with_source > 0,
            "captured_names": captured, "samples_total": len(samples),
            "samples_operator": len(op_samples), "samples_per_kernel": per_kernel,
            "samples_with_isa": with_isa, "samples_with_source_line": with_source}


#: Columns of the derived per-instruction PC-sampling table.
PC_INSTRUCTION_COLUMNS = ("kernel_name", "operator_kernel", "code_object_id", "offset",
                          "instruction", "source_line", "samples", "issued", "stalled",
                          "stall_reasons", "instruction_types")


def pc_sampling_instruction_rows(workload_dir) -> list[dict]:
    """The PC samples of a PC-sampling workload aggregated per instruction.

    Derived from ps_file_results.json alone (rocprof-compute's CSV export has
    no PC-sampling view; its text report has the same table). One row per
    (kernel, code object, offset) for every sample correlated with a
    dispatch: the decoded instruction, its source line, sample counts and
    the stall reasons. operator_kernel marks the operator's own kernels; the
    other rows are the input generators and the eviction fill, which the pass
    also samples."""
    tool = load_pc_sampling(workload_dir)
    if tool is None:
        return []
    names = {k["kernel_id"]: k.get("formatted_kernel_name") or k["kernel_name"]
             for k in tool["kernel_symbols"]}
    dispatch_kernel = {d["dispatch_info"]["dispatch_id"]: names.get(d["dispatch_info"]["kernel_id"], "")
                       for d in tool["buffer_records"]["kernel_dispatch"]}
    instructions = tool["strings"].get("pc_sample_instructions", [])
    comments = tool["strings"].get("pc_sample_comments", [])
    samples = (tool["buffer_records"].get("pc_sample_stochastic")
               or tool["buffer_records"].get("pc_sample_host_trap") or [])
    rows: dict[tuple, dict] = {}
    for s in samples:
        rec = s["record"]
        kernel = dispatch_kernel.get(rec.get("dispatch_id"))
        if kernel is None:
            continue                     # not correlated with a dispatch
        idx = s.get("inst_index", -1)
        key = (kernel, rec["pc"]["code_object_id"], rec["pc"]["code_object_offset"])
        row = rows.setdefault(key, {
            "kernel_name": kernel, "operator_kernel": not ks.is_aux_kernel(kernel),
            "code_object_id": key[1], "offset": hex(key[2]),
            "instruction": instructions[idx] if 0 <= idx < len(instructions) else "",
            "source_line": comments[idx] if 0 <= idx < len(comments) else "",
            "samples": 0, "issued": 0, "stalled": 0,
            "stall_reasons": {}, "instruction_types": {}})
        row["samples"] += 1
        if "wave_issued" in rec:
            row["issued" if rec["wave_issued"] else "stalled"] += 1
        reason = (rec.get("snapshot") or {}).get("stall_reason")
        if reason and not rec.get("wave_issued"):
            reason = reason.rsplit("REASON_", 1)[-1]
            row["stall_reasons"][reason] = row["stall_reasons"].get(reason, 0) + 1
        itype = rec.get("inst_type")
        if itype:
            itype = itype.rsplit("TYPE_", 1)[-1]
            row["instruction_types"][itype] = row["instruction_types"].get(itype, 0) + 1
    out = sorted(rows.values(), key=lambda r: (not r["operator_kernel"], r["kernel_name"],
                                               -r["samples"], r["offset"]))
    for r in out:
        r["stall_reasons"] = json.dumps(r["stall_reasons"], sort_keys=True)
        r["instruction_types"] = json.dumps(r["instruction_types"], sort_keys=True)
    return out


def write_pc_sampling_instructions(workload_dir, out_csv) -> int:
    """Write pc_sampling_instruction_rows() as CSV; returns the row count."""
    rows = pc_sampling_instruction_rows(workload_dir)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PC_INSTRUCTION_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
