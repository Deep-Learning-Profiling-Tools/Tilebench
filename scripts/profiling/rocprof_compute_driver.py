"""ROCm Compute Profiler sweep driver — profile every valid (op, dtype, backend)
of one GPU's catalogue with rocprof-compute, at the sweep-max input with the
exact autotune winner, and validate every capture against the kernel manifest.

The AMD counterpart of ncu_driver.py. Inputs, both of --gpu, never another GPU's:
  outputs/profiling/<gpu>/ncu_catalogue.json   sweep-max case + autotune winner
                                               (scripts/profiling/ncu_catalogue.py)
  outputs/profiling/<gpu>/kernel_counts.json   the compute-kernel launch sequence
                                               of ONE impl.run() (probe_kernel_count.py)

A valid pair is a catalogue (op, dtype) whose autotune winner for the backend
exists. A dtype the GPU's autotune run has no result for (an unsupported dtype
such as fp8_e4m3fn on MI300X) is excluded and listed, never replaced.

For each pair, into outputs/rocprof_compute/<gpu>/<op>/<backend>_<dtype>/
(tilebench.paths.rocprof_compute_pair_dir; layout in tilebench.profiling.rocprof_compute):
  1. prime     the harness once without a profiler (JIT warmup, Triton cache)
  2. profile   `rocprof-compute profile` with its default (full) metric
               collection and roofline, kernel-name selection only:
               workload/
  3. validate  every counter pass holds exactly the manifest's launch sequence
               (names, count, order; aux kernels absent)
  4. pc        a PC-sampling pass (--experimental --pc-sampling): pc_sampling/,
               validated the same way; source/ISA correlation recorded
  5. analyze   `rocprof-compute analyze` on copies of both workloads: the full
               text report and the CSV export, under analysis/, plus the
               per-instruction PC-sampling table (pc_sampling_instructions.csv)
  6. record    capture.json in the pair directory and an entry in
               outputs/rocprof_compute/<gpu>/sweep_log.json

Resumable: an entry recorded ok is skipped only after its artifact validates
again against the current manifest; anything else is profiled again (an
artifact on disk is never taken as success by itself). Failed pairs keep their
error. At the end the coverage of the valid pairs is checked and written to
outputs/rocprof_compute/<gpu>/coverage.json.

Usage:
  python scripts/profiling/rocprof_compute_driver.py --gpu MI300X \\
      --rocprof-compute /opt/rocm/bin/rocprof-compute \\
      --rocprof-compute-python <python with rocprof-compute's requirements>
  ... --ops vector_add            # one operator (comma-separated list)
  ... --ops vector_add --dtypes fp32
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without setting PYTHONPATH
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilebench.paths import (REPO_ROOT, hardware_label, kernel_counts_path,  # noqa: E402
                             rocprof_compute_output_dir, rocprof_compute_pair_dir)
from tilebench.profiling import ncu_kernel_select as ks  # noqa: E402
from tilebench.profiling import rocprof_compute as rc  # noqa: E402

# The process rocprof-compute profiles: a sibling script, located from this file.
HARNESS = Path(__file__).resolve().with_name("rocprof_compute_harness.py")
BACKEND = "triton"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


class Profiler:
    """How to start rocprof-compute on this node."""

    def __init__(self, tool: str, python: str | None, timeout_s: int):
        self.tool = tool
        self.python = python
        self.timeout_s = timeout_s
        self.env = rc.profiler_env(os.environ, tool)

    def cmd(self, *args) -> list[str]:
        return ([self.python] if self.python else []) + [self.tool, *args]

    def run(self, args: list[str], env: dict, cwd: Path, log: Path) -> int:
        """Run rocprof-compute; its whole output goes to `log`."""
        with open(log, "w") as fh:
            fh.write("$ " + " ".join(self.cmd(*args)) + "\n")
            fh.flush()
            try:
                r = subprocess.run(self.cmd(*args), env=env, cwd=str(cwd), stdout=fh,
                                   stderr=subprocess.STDOUT, timeout=self.timeout_s)
                return r.returncode
            except subprocess.TimeoutExpired:
                fh.write(f"\nTIMEOUT after {self.timeout_s}s\n")
                return -1

    def version(self) -> dict:
        r = subprocess.run(self.cmd("--version"), env=self.env, capture_output=True, text=True)
        out = r.stdout + r.stderr
        info = {"rocm_version": self.env.get("ROCM_VER"), "tool": self.tool}
        for line in out.splitlines():
            if "rocprofiler-compute version:" in line:
                info["rocprof_compute_version"] = line.split(":", 1)[1].strip()
            elif line.strip().startswith("Git revision:"):
                info["rocprof_compute_git"] = line.split(":", 1)[1].strip()
        return info


def valid_pairs(catalogue: list, ops: set, dtypes: set) -> tuple[list, list]:
    """(valid pairs to profile, excluded pairs with the reason)."""
    pairs, excluded = [], []
    for c in catalogue:
        if ops and c["op"] not in ops:
            continue
        for dt in c["dtypes"]:
            if dtypes and dt not in dtypes:
                continue
            winner = (c["autotune_winner_per_dtype"].get(dt) or {}).get(BACKEND)
            if winner is None:
                excluded.append({"op": c["op"], "dtype": dt, "backend": BACKEND,
                                 "reason": "no autotune winner on this GPU (the autotune run "
                                           "has no result for this dtype, e.g. UNSUPPORTED_DTYPE)"})
                continue
            pairs.append((c["op"], dt, c["default_params_per_dtype"][dt], winner))
    return pairs, excluded


def harness_env(base: dict, op: str, dtype: str, params: dict, cfg: dict, mode: str) -> dict:
    env = dict(base)
    env.update({"PROF_OP": op, "PROF_BACKEND": BACKEND, "PROF_DTYPE": dtype,
                "PROF_PARAMS_JSON": json.dumps(params), "PROF_CFG_JSON": json.dumps(cfg),
                "PROF_MODE": mode, "PYTHONPATH": str(REPO_ROOT)})
    return env


def analyze(prof: Profiler, workload: Path, analysis: Path, stem: str, tag: str,
            scratch: Path) -> dict:
    """`rocprof-compute analyze` of a copy of `workload` (analyze writes its
    joined pmc_perf.csv and dispatch/kernel tables into the directory it
    reads; the stored workload stays exactly what `profile` wrote).
    Writes analysis/<tag>_report.txt and analysis/<tag>_csv/."""
    copy = scratch / stem
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(workload, copy)
    out = {}
    report = analysis / f"{tag}_report.txt"
    out["report_rc"] = prof.run(["analyze", "-p", str(copy)], prof.env, analysis, report)
    csv_name = f"{tag}_csv"
    shutil.rmtree(analysis / csv_name, ignore_errors=True)
    out["csv_rc"] = prof.run(["analyze", "-p", str(copy), "--output-format", "csv",
                              "--output-name", csv_name], prof.env, analysis,
                             analysis / f"{tag}_csv.log")
    out["csv_files"] = sorted(p.name for p in (analysis / csv_name).glob("*.csv"))
    shutil.rmtree(copy, ignore_errors=True)
    return out


def profile_pair(prof: Profiler, gpu: str, op: str, dtype: str, params: dict, cfg: dict,
                 expected: list[str], pc: dict | None, scratch: Path) -> dict:
    pair = rocprof_compute_pair_dir(gpu, op, BACKEND, dtype)
    shutil.rmtree(pair, ignore_errors=True)        # a fresh artifact, never a mix of attempts
    (pair / "logs").mkdir(parents=True)
    analysis = pair / "analysis"
    analysis.mkdir()
    entry = {"op": op, "dtype": dtype, "backend": BACKEND, "params": params, "winner": cfg,
             "expected_count": len(expected), "expected_names": expected,
             "workload": str(pair / "workload"), "started": _now()}
    regex = rc.kernel_filter_regex(expected)
    entry["kernel_filter"] = regex
    t0 = time.time()

    # 1. prime: JIT warmup outside the profiler
    with open(pair / "logs" / "prime.log", "w") as fh:
        try:
            r = subprocess.run([sys.executable, str(HARNESS)], cwd=str(REPO_ROOT),
                               env=harness_env(prof.env, op, dtype, params, cfg, "prime"),
                               stdout=fh, stderr=subprocess.STDOUT, timeout=prof.timeout_s)
            prime_rc = r.returncode
        except subprocess.TimeoutExpired:
            prime_rc = -1
    if prime_rc != 0:
        return {**entry, "ok": False, "error": f"prime run failed (rc={prime_rc}), see logs/prime.log"}

    env = harness_env(prof.env, op, dtype, params, cfg, "profile")
    workload_cmd = ["--", sys.executable, str(HARNESS)]

    # 2. full counter profile (rocprof-compute defaults: every block, roofline)
    profile_args = ["profile", "--output-directory", str(pair / "workload"), "-k", regex,
                    *workload_cmd]
    entry["profile_command"] = " ".join(prof.cmd(*profile_args))
    rc_profile = prof.run(profile_args, env, REPO_ROOT, pair / "logs" / "profile.log")
    entry["profile_rc"] = rc_profile

    # 3. capture validation: never just the exit code
    v = rc.validate_workload(pair / "workload", expected)
    entry.update({"captured_count": len(v["captured_names"]), "captured_names": v["captured_names"],
                  "counter_passes": v["passes"], "launches_per_pass": v["launches_per_pass"]})
    problems = ([f"rocprof-compute profile rc={rc_profile}"] if rc_profile != 0 else []) + v["problems"]

    # 4. PC-sampling pass (only on top of a valid counter capture)
    entry["pc_sampling"] = bool(pc)
    entry["source_isa_correlation"] = False
    if pc and not problems:
        pc_args = ["--experimental", "profile", "--pc-sampling",
                   "--pc-sampling-method", pc["method"],
                   "--pc-sampling-interval", str(pc["interval"]),
                   "--output-directory", str(pair / "pc_sampling"), "-k", regex, *workload_cmd]
        entry["pc_sampling_command"] = " ".join(prof.cmd(*pc_args))
        rc_pc = prof.run(pc_args, env, REPO_ROOT, pair / "logs" / "pc_sampling.log")
        pv = rc.validate_pc_sampling(pair / "pc_sampling", expected)
        entry["pc_sampling_summary"] = {k: pv.get(k) for k in (
            "method", "samples_total", "samples_operator", "samples_per_kernel",
            "samples_with_isa", "samples_with_source_line")}
        entry["source_isa_correlation"] = bool(pv.get("correlation_available"))
        if rc_pc != 0:
            problems.append(f"PC-sampling profile rc={rc_pc}")
        problems += [f"pc_sampling: {p}" for p in pv["problems"]]

    # 5. analysis of the stored workloads (re-analysis without re-profiling)
    if not problems:
        stem = f"{op}__{BACKEND}_{dtype}"
        a = analyze(prof, pair / "workload", analysis, stem, "workload", scratch)
        entry["analysis"] = {"workload": a}
        if a["report_rc"] != 0 or a["csv_rc"] != 0 or not a["csv_files"]:
            problems.append(f"analyze of workload failed: {a}")
        if pc:
            a = analyze(prof, pair / "pc_sampling", analysis, stem + "__pc", "pc_sampling", scratch)
            # rocprof-compute's CSV export has no PC-sampling view: the
            # per-instruction table (ISA, source line, stall reasons) is
            # derived from the raw samples instead.
            a["instruction_rows"] = rc.write_pc_sampling_instructions(
                pair / "pc_sampling", analysis / "pc_sampling_instructions.csv")
            entry["analysis"]["pc_sampling"] = a
            if a["report_rc"] != 0:
                problems.append(f"analyze of pc_sampling failed: {a}")

    entry.update({"ok": not problems, "error": "; ".join(problems) if problems else None,
                  "elapsed_s": round(time.time() - t0, 1), "finished": _now()})
    _write_json(pair / "capture.json", entry)
    return entry


def artifact_still_valid(entry: dict, expected: list[str], pc: dict | None) -> bool:
    """Whether a recorded success still holds: its workloads exist and validate
    against the current manifest, and its analysis is present."""
    pair = Path(entry["workload"]).parent
    if not (pair / "capture.json").is_file() or entry.get("expected_names") != expected:
        return False
    if not rc.validate_workload(pair / "workload", expected)["ok"]:
        return False
    if not (pair / "analysis" / "workload_report.txt").is_file():
        return False
    if pc and not rc.validate_pc_sampling(pair / "pc_sampling", expected)["ok"]:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="ROCm Compute Profiler sweep over one GPU's catalogue.")
    ap.add_argument("--gpu", type=hardware_label, required=True, metavar="LABEL",
                    help="Hardware label (e.g. MI300X): reads outputs/profiling/<gpu>/, "
                         "writes outputs/rocprof_compute/<gpu>/")
    ap.add_argument("--ops", default="", help="Comma-separated operators (default: all)")
    ap.add_argument("--dtypes", default="", help="Comma-separated dtypes (default: all)")
    ap.add_argument("--rocprof-compute", default=shutil.which("rocprof-compute"),
                    help="rocprof-compute launcher (default: from PATH)")
    ap.add_argument("--rocprof-compute-python", default=None,
                    help="Python interpreter that has rocprof-compute's requirements, used to "
                         "start the launcher (default: the launcher's own shebang)")
    ap.add_argument("--no-pc-sampling", action="store_true",
                    help="Skip the PC-sampling pass")
    ap.add_argument("--pc-sampling-method", default="stochastic", choices=["stochastic", "host_trap"])
    ap.add_argument("--pc-sampling-interval", type=int, default=65536,
                    help="Cycles (stochastic; rocprof-compute's minimum is 65536) or us (host_trap)")
    ap.add_argument("--timeout", type=int, default=7200, help="Seconds per profiler run")
    ap.add_argument("--force", action="store_true", help="Profile pairs already recorded ok again")
    args = ap.parse_args()
    if not args.rocprof_compute:
        sys.exit("error: rocprof-compute not found; pass --rocprof-compute")

    try:
        catalogue = ks.load_catalogue(args.gpu)
        ks.load_kernel_counts(args.gpu)                # raises the explanatory error if absent
        manifest = {(r["op"], r["dtype"], r["backend"]): r
                    for r in json.loads(kernel_counts_path(args.gpu).read_text())}
    except ks.MissingProfilingMetadataError as e:
        sys.exit(f"error: {e}")

    ops = {o.strip() for o in args.ops.split(",") if o.strip()}
    dtypes = {d.strip() for d in args.dtypes.split(",") if d.strip()}
    pairs, excluded = valid_pairs(catalogue, ops, dtypes)
    pc = None if args.no_pc_sampling else {"method": args.pc_sampling_method,
                                           "interval": args.pc_sampling_interval}

    prof = Profiler(args.rocprof_compute, args.rocprof_compute_python, args.timeout)
    version = prof.version()
    out_dir = rocprof_compute_output_dir(args.gpu)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sweep_log.json"
    log = {}
    if log_path.exists():
        log = {(e["op"], e["dtype"], e["backend"]): e for e in json.loads(log_path.read_text())}
    print(f"rocprof-compute {version.get('rocprof_compute_version')} "
          f"({version.get('rocprof_compute_git')}), ROCm {version.get('rocm_version')}; "
          f"{len(pairs)} valid pairs, {len(excluded)} excluded", flush=True)

    scratch = Path(tempfile.mkdtemp(prefix="rocprof_compute_analyze_"))
    try:
        for i, (op, dt, params, cfg) in enumerate(pairs, 1):
            key = (op, dt, BACKEND)
            row = manifest.get(key) or {}
            expected = row.get("names") or []
            prev = log.get(key)
            if (prev and prev.get("ok") and not args.force
                    and artifact_still_valid(prev, expected, pc)):
                print(f"[{i}/{len(pairs)}] {op}/{dt}: ok (validated, skipped)", flush=True)
                continue
            if row.get("count") is None or row.get("count") != len(expected) or not expected:
                entry = {"op": op, "dtype": dt, "backend": BACKEND, "ok": False,
                         "error": f"no usable kernel manifest row: {row or 'missing'}"}
            elif row.get("first_call_identical") is not True:
                entry = {"op": op, "dtype": dt, "backend": BACKEND, "ok": False,
                         "error": "first call launches a different compute sequence: the "
                                  "single-call harness cannot capture the manifest sequence"}
            else:
                print(f"[{i}/{len(pairs)}] {op}/{dt}: profiling {len(expected)} launches", flush=True)
                entry = profile_pair(prof, args.gpu, op, dt, params, cfg, expected, pc, scratch)
            entry.update({"profiler": version, "gpu": args.gpu})
            if prev and not prev.get("ok") and prev.get("error"):
                entry["previous_errors"] = (prev.get("previous_errors") or []) + [prev["error"]]
            log[key] = entry
            _write_json(log_path, sorted(log.values(), key=lambda e: (e["op"], e["dtype"], e["backend"])))
            print(f"    -> {'ok' if entry['ok'] else 'FAILED: ' + str(entry.get('error'))[:300]}"
                  f"  (captured {entry.get('captured_count')}/{len(expected)}, "
                  f"source/ISA={entry.get('source_isa_correlation')}, "
                  f"{entry.get('elapsed_s', 0)}s)", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    # Coverage of the full valid set (independent of --ops/--dtypes filters).
    all_pairs, all_excluded = valid_pairs(catalogue, set(), set())
    expected_keys = [(op, dt, BACKEND) for op, dt, _, _ in all_pairs]
    ok_keys = [k for k in expected_keys if log.get(k, {}).get("ok")]
    failed = [{"op": k[0], "dtype": k[1], "error": log[k].get("error")}
              for k in expected_keys if k in log and not log[k].get("ok")]
    missing = [{"op": k[0], "dtype": k[1]} for k in expected_keys if k not in log]
    coverage = {"gpu": args.gpu, "backend": BACKEND, "profiler": version,
                "expected_valid_pairs": len(expected_keys), "successful_pairs": len(ok_keys),
                "complete": len(ok_keys) == len(expected_keys),
                "failed": failed, "missing": missing, "excluded": all_excluded,
                "written": _now()}
    _write_json(out_dir / "coverage.json", coverage)
    print(f"\n=== coverage: {len(ok_keys)}/{len(expected_keys)} valid pairs ok, "
          f"{len(failed)} failed, {len(missing)} not yet profiled, "
          f"{len(all_excluded)} excluded === complete={coverage['complete']}", flush=True)


if __name__ == "__main__":
    main()
