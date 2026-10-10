"""Run every smoke test in smoke_tests/ (one device process at a time) and write
smoke_results.json next to this file.

Each test is a standalone script that imports smoke_tests/_common.py (which sets the
evaluator environment: NEURON_LOGICAL_NC_CONFIG=2, NEURON_CC_FLAGS="--target trn2 --lnc 2",
NEURON_PLATFORM_TARGET_OVERRIDE=trn2, NEURON_RT_NUM_CORES=1) and prints one line
"SMOKE_RESULT {json}".

All compiler/runtime caches go to a NEW directory per run (under --cache-base, default
the system temp dir), so every kernel is compiled by this run.

Usage:
  /home/ubuntu/workspace/native_venv/bin/python run_smoke.py [--cache-base DIR] [--only a_,b_]
"""

import argparse
import datetime
import glob
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "smoke_tests")
PER_TEST_TIMEOUT = 900  # seconds


def versions():
    v = {"python": platform.python_version(), "executable": sys.executable}
    for p in ["nki", "neuronx-cc", "torch", "torch-neuronx", "numpy", "islpy"]:
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    try:
        v["neuron-ls"] = subprocess.run(["neuron-ls"], capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        v["neuron-ls"] = f"<unavailable: {e!r}>"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-base", default=tempfile.gettempdir())
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=os.path.join(HERE, "smoke_results.json"))
    args = ap.parse_args()

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cache_root = tempfile.mkdtemp(prefix=f"nki_ref_smoke_{stamp}_", dir=args.cache_base)
    env = dict(os.environ)
    env["NKI_SMOKE_CACHE_ROOT"] = cache_root
    env["PYTHONUNBUFFERED"] = "1"

    scripts = sorted(p for p in glob.glob(os.path.join(TESTS, "[a-z]_*.py")))
    if args.only:
        prefixes = tuple(args.only.split(","))
        scripts = [p for p in scripts if os.path.basename(p).startswith(prefixes)]

    results = []
    t_all = time.time()
    for path in scripts:
        name = os.path.splitext(os.path.basename(path))[0]
        t = time.time()
        print(f"== {name}", flush=True)
        try:
            proc = subprocess.run(
                ["timeout", str(PER_TEST_TIMEOUT), sys.executable, path],
                cwd=TESTS, env=env, capture_output=True, text=True,
                timeout=PER_TEST_TIMEOUT + 30,
            )
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            rc, out, err = -1, e.stdout or "", f"timeout after {PER_TEST_TIMEOUT} s"
        rec = None
        for line in out.splitlines():
            if line.startswith("SMOKE_RESULT "):
                rec = json.loads(line[len("SMOKE_RESULT "):])
        if rec is None:
            rec = {"test": name, "status": "fail", "cases": [],
                   "harness_error": f"no SMOKE_RESULT line (exit code {rc})",
                   "stderr_tail": err.splitlines()[-25:]}
        rec["script"] = f"smoke_tests/{name}.py"
        rec["exit_code"] = rc
        rec["wall_seconds"] = round(time.time() - t, 2)
        results.append(rec)
        for c in rec.get("cases", []):
            print(f"   {c['status']:4s} {c['case']}", flush=True)

    n_cases = sum(len(r.get("cases", [])) for r in results)
    n_pass = sum(1 for r in results for c in r.get("cases", []) if c["status"] == "pass")
    n_neg = sum(1 for r in results for c in r.get("cases", []) if c.get("negative"))
    doc = {
        "generated_by": "evidence/run_smoke.py",
        "generated_at_utc": stamp,
        "host": platform.node(),
        "versions": versions(),
        "environment": {
            "NEURON_LOGICAL_NC_CONFIG": "2",
            "NEURON_CC_FLAGS": "--target trn2 --lnc 2",
            "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
            "NEURON_RT_NUM_CORES": "1",
            "caches": "TORCH_NEURONX_NEFF_CACHE_DIR, TORCH_NEURONX_HLO_CACHE_DIR, TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR, "
                      "NKI_TRACE_CACHE_URL, NKI_COMPILE_CACHE_URL, TORCH_NEURONX_DEBUG_DIR -> fresh per-run directory",
            "cache_root": cache_root,
        },
        "conventions": {
            "negative": "negative=true cases pass when the call raises; the observed error text is recorded in 'error'",
            "error_text": "exception type + whitespace-collapsed message, truncated to 900 characters",
        },
        "summary": {
            "tests": len(results),
            "tests_passed": sum(1 for r in results if r["status"] == "pass"),
            "cases": n_cases,
            "cases_passed": n_pass,
            "negative_cases": n_neg,
            "failed_cases": [f"{r['test']}::{c['case']}" for r in results for c in r.get("cases", [])
                             if c["status"] != "pass"],
            "wall_seconds": round(time.time() - t_all, 1),
        },
        "tests": results,
    }
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    print(json.dumps(doc["summary"], indent=1))


if __name__ == "__main__":
    main()
