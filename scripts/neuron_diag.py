#!/usr/bin/env python
"""Native Neuron (Trn2) diagnostics entry point (does not touch results/ or
run_bench.py).

    python scripts/neuron_diag.py env      --run-id R
    python scripts/neuron_diag.py sources  --run-id R [--prs prs.json]
    python scripts/neuron_diag.py run      --run-id R --ops matrix_copy,rmsnorm --cases pilot
                                           --native-python <native venv>/bin/python
                                           [--warmup 20 --repeat 100] [--smoke]
    python scripts/neuron_diag.py report   --run-id R

The benchmark modes are native_torch_eager and native_nki (speedup = eager / NKI). The xla_*
modes are legacy diagnostics (`run --legacy-xla-diagnostics`; `report --legacy-xla` writes their
separate view) and native_torch_compiled is a torch.compile diagnostic (`run
--compile-diagnostics`); neither enters the benchmark report.

Every run lives in .local/neuron_native_diagnostics/<run-id>/ (Git ignored). `env` creates
the run directory; later commands resume it. `run` is resumable: a finished
(operator, case, mode) with the same source and environment hash is skipped.
Workers run strictly one at a time, each in a fresh process with a timeout.
"""
from __future__ import annotations

import argparse
import json
import shutil
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from tilebench.neuron_diag import cases as case_sel  # noqa: E402
from tilebench.neuron_diag.envinfo import env_hash, snapshot  # noqa: E402
from tilebench.neuron_diag.schema import (BENCHMARK_MODES, COMPILE_DIAGNOSTIC_MODES,  # noqa: E402
                                          EXECUTION_MODES, LEGACY_DIAGNOSTIC_MODES, Record)
from tilebench.neuron_diag.store import RunStore  # noqa: E402

# Neuron settings shared by the XLA-stack workers: the production benchmark's
# LNC=2 launch convention and one logical core per process.
XLA_ENV = {"NEURON_LOGICAL_NC_CONFIG": "2", "NEURON_CC_FLAGS": "--target trn2 --lnc 2",
           "NEURON_RT_NUM_CORES": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2"}


# Both stacks get the same core allocation and LNC setting, so neither can look
# faster by holding more NeuronCores.
STACK_ENV = {"xla": XLA_ENV, "native": dict(XLA_ENV)}

# (stack, python) -> reason, when the runtime probe said the stack is not available
_BLOCKED_STACK: dict = {}


def _worker_env(stack: str, py: str, cache: Path) -> dict:
    """Launch env for one worker. The native stack spawns `neuronx-cc` from PATH and keeps its
    NEFF/HLO/NKI-trace/inductor caches in global default locations; both are pinned here so a
    native worker uses its own interpreter's compiler and the same private per-case cache that
    NEURON_COMPILE_CACHE_URL gives the XLA stack."""
    env = dict(STACK_ENV[stack], NEURON_COMPILE_CACHE_URL=str(cache))
    if stack == "native":
        env["PATH"] = os.pathsep.join([str(Path(py).parent), os.environ.get("PATH", "")])
        env.update(TORCH_NEURONX_NEFF_CACHE_DIR=str(cache / "neff"),
                   TORCH_NEURONX_HLO_CACHE_DIR=str(cache / "hlo"),
                   TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=str(cache / "local"),
                   NKI_TRACE_CACHE_URL=str(cache / "nki_trace"),
                   TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
                   TORCH_NEURONX_DEBUG_DIR=str(cache / "debug"))
    return env


def _store(run_id: str, create: bool = False) -> RunStore:
    return RunStore(run_id, resume=not create)


def _load(store: RunStore, rel: str):
    with open(store.path(rel)) as f:
        return json.load(f)


def _run_child(cmd: list[str], *, overlay: Path, env_extra: dict, timeout: int,
               log_path: Path) -> tuple[str, int | None]:
    """Run a worker in its own process group; returns ("ok"|"timeout"|"error", rc)."""
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = str(overlay)
    env["TILEBENCH_REPO_ROOT"] = str(overlay)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=str(overlay), env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return "timeout", None
    return ("ok" if rc == 0 else "error"), rc


# ---------------------------------------------------------------- env / sources
def cmd_env(a) -> None:
    store = _store(a.run_id, create=not a.resume)
    snap = snapshot(a.stack_label)
    rel = f"env/{a.stack_label}.json"
    store.write_new(rel, snap)
    print(f"environment snapshot -> {store.path(rel)} (env_hash {env_hash(snap)[:12]})")


def cmd_sources(a) -> None:
    from tilebench.neuron_diag import sources

    store = _store(a.run_id)
    prs = None
    if a.prs:
        with open(a.prs) as f:
            prs = {p["head"]: {"number": p["number"], "state": p["state"], "head_sha": p["sha"]}
                   for p in json.load(f)}
    manifest = sources.build_manifest(prs)
    overlays_root = store.root / "overlays"
    for e in manifest["operators"]:
        if e["status"] == "found":
            e["overlay"] = str(sources.materialize_overlay(e, overlays_root))
            if prs and e.get("pr") and e["pr"]["head_sha"] != e["commit"]:
                e["pr_head_mismatch"] = True
    existing = sorted(store.path("sources").glob("manifest*.json")) if store.path("sources").is_dir() else []
    name = "manifest.json" if not existing else f"manifest.v{len(existing) + 1}.json"
    store.write_new(f"sources/{name}", manifest)
    n = sum(e["status"] == "found" for e in manifest["operators"])
    print(f"{n}/{len(manifest['operators'])} operators with an NKI source -> "
          f"{store.path('sources/' + name)}")


def cmd_compat_patch(a) -> None:
    """Next manifest version = latest manifest with the declared runtime compatibility patch
    applied to the named operators (all other entries copied unchanged)."""
    from tilebench.neuron_diag import sources

    store = _store(a.run_id)
    manifest = _load(store, _latest_manifest(store))
    for i, e in enumerate(manifest["operators"]):
        if e["operator"] in a.ops.split(","):
            if e.get("runtime_compat_patch"):
                raise SystemExit(f"{e['operator']} is already patched")
            manifest["operators"][i] = sources.apply_runtime_compat_patch(e, store.root / "overlays")
            p = manifest["operators"][i]
            print(f"{p['operator']}: {p['base_source_hash'][:16]} -> {p['source_hash'][:16]} "
                  f"(patch {p['runtime_compat_patch']['sha256'][:16]})")
    n = len(list(store.path("sources").glob("manifest*.json")))
    store.write_new(f"sources/manifest.v{n + 1}.json", manifest)
    print(f"-> sources/manifest.v{n + 1}.json")


def cmd_native_device(a) -> None:
    """Device timing for already-recorded native rows (tilebench/neuron_diag/native_device.py).
    Each passing row gets an updated copy with device fields, which supersedes it explicitly."""
    store = _store(a.run_id)
    manifest = _load(store, _latest_manifest(store))
    entries = {e["operator"]: e for e in manifest["operators"]}
    snap = _load(store, "env/native.json")
    ehash = env_hash(snap)
    py = a.native_python
    worker = REPO / "tilebench" / "neuron_diag" / "native_device.py"
    modes = a.modes.split(",")
    rows = [r for r in store.records() if r.get("stack") == "native" and r["operator"] in a.ops.split(",")
            and r["mode"] in modes and r["status"] == "pass" and not r.get("device_ms")]
    if a.cases not in ("all",):
        keep = {int(x) for x in a.cases.split(",")}
        rows = [r for r in rows if r["case_id"] in keep]
    for r in sorted(rows, key=lambda r: (r["operator"], r["case_id"], r["mode"])):
        op, cid, m = r["operator"], r["case_id"], r["mode"]
        e = entries[op]
        if r["source_hash"] != e["source_hash"]:
            print(f"[skip] {op} c{cid} {m}: row source hash differs from the manifest", flush=True)
            continue
        bres = json.loads(store.path("bundles", op, f"c{cid:03d}", "bundle_result.json").read_text())
        vcfg = (r.get("config") or {}).get("verify") or {}
        ts = int(time.time())
        out = store.path("device", op, f"c{cid:03d}", f"{m}.native_device.{ts}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        spec = {"operator": op, "mode": m, "params": r["shape"], "dtype": r["dtype"],
                "block_size": (r.get("config") or {}).get("block_size", 1024),
                "bundle": bres["main"]["path"], "bundle_sha256": bres["main"]["file_sha256"],
                "atol": float(vcfg["atol"]) if "atol" in vcfg else None,
                "rtol": float(vcfg["rtol"]) if "rtol" in vcfg else None,
                "run_mutates_inputs": (r.get("compile_info") or {}).get("run_mutates_inputs", False),
                "warmup": r["warmup"], "repeat": r["repeat"],
                "profile_dir": str(out.with_suffix("")) + ".profile", "out": str(out)}
        spec_path = out.with_suffix(".spec.json")
        spec_path.write_text(json.dumps(spec))
        cache = store.path("compile_cache", op, f"c{cid:03d}", "native")
        t0 = time.time()
        st, rc = _run_child([py, str(worker), "--spec", str(spec_path)], overlay=Path(e["overlay"]),
                            env_extra=_worker_env("native", py, cache), timeout=a.device_timeout,
                            log_path=store.path("logs", op, f"c{cid:03d}_{m}_native_device.log"))
        res = json.loads(out.read_text()) if out.is_file() else {"status": "unavailable",
                                                                  "reason": f"device worker {st} rc={rc}"}
        new = {k: v for k, v in r.items() if k not in ("row_id", "valid_for_analysis", "recorded_at",
                                                        "env_hash", "invalid_reason", "superseded_by")}
        new["compile_info"] = dict(new.get("compile_info") or {})
        dev = {k: v for k, v in res.items() if k != "iterations"}
        dev["device_timing_status"] = "available" if res.get("status") == "ok" else "unavailable"
        dev["artifact"] = str(out)
        new["compile_info"]["device"] = dev
        if res.get("status") == "ok":
            new.update(device_ms=res["device_ms"], device_samples_ms=res["device_samples_ms"],
                       timing_method="native_device_trace", artifact_identity=res["artifact_identity"],
                       device_unavailable_reason="")
        else:
            new.update(device_ms=None, device_unavailable_reason=f"native device timing unavailable: {res.get('reason')}"[:2000])
        store.append(Record.from_json(new), env_hash=ehash, supersede_reason="device_timing_added")
        print(f"[{dev['device_timing_status']:>11}] {op} c{cid} {m} device_ms={new.get('device_ms')} "
              f"execs/iter={res.get('executions_per_iteration')} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- run
def _latest_manifest(store: RunStore) -> str:
    """sources/manifest.json, or the newest manifest.vN.json written by a later `sources`."""
    files = list(store.path("sources").glob("manifest*.json"))

    def version(p):
        stem = p.stem
        return int(stem.split(".v")[1]) if ".v" in stem else 1
    return "sources/" + max(files, key=version).name


def _hardware(snap: dict) -> dict:
    n = snap.get("neuron") or {}
    return {"instance_type": n.get("instance_type"),
            "logical_neuroncore_config": n.get("logical_neuroncore_config"),
            "launch_env": {k: XLA_ENV[k] for k in sorted(XLA_ENV)},
            "native_launch_env_same_as_xla": STACK_ENV["native"] == STACK_ENV["xla"],
            "NEURON_RT_VISIBLE_CORES": os.environ.get("NEURON_RT_VISIBLE_CORES")}


def _select(op: str, cfg: dict, how: str) -> list[int]:
    if how == "smoke":
        return case_sel.smoke_cases(op, cfg)
    if how == "pilot":
        return sorted({i for d in case_sel.pilot_cases(op, cfg).values() for i in d.values()})
    return [int(x) for x in how.split(",") if x.strip()]


def cmd_run(a) -> None:
    modes = a.modes.split(",")
    for m in modes:
        if m not in EXECUTION_MODES:
            raise SystemExit(f"unknown mode {m}")
    legacy = [m for m in modes if m in LEGACY_DIAGNOSTIC_MODES]
    if legacy and not a.legacy_xla_diagnostics:
        raise SystemExit(f"{legacy} are legacy XLA diagnostic modes, not benchmark modes; "
                         "pass --legacy-xla-diagnostics (and use a separate run id) to run them")
    comp = [m for m in modes if m in COMPILE_DIAGNOSTIC_MODES]
    if comp and not a.compile_diagnostics:
        raise SystemExit(f"{comp} is a torch.compile diagnostic mode, not a benchmark mode; "
                         "pass --compile-diagnostics to run it")
    store = _store(a.run_id)
    manifest = _load(store, _latest_manifest(store))
    snap = _load(store, f"env/{a.stack_label}.json")
    ehash = env_hash(snap)
    entries = {e["operator"]: e for e in manifest["operators"]}
    ops = list(entries) if a.ops == "all" else a.ops.split(",")
    if store.archived:
        raise SystemExit(f"{store.dir} is an archived legacy XLA run; start a new run id")
    py_for = {"xla": a.xla_python or sys.executable, "native": a.native_python or sys.executable}
    hw = _hardware(snap)
    plan_rows = []
    for op in ops:
        e = entries.get(op)
        if e is None:
            raise SystemExit(f"{op} not in the source manifest")
        if e["status"] != "found":
            for m in modes:
                if EXECUTION_MODES[m][1] == "nki":
                    store.append(Record(operator=op, case_id=-1, mode=m, status="no_nki_impl",
                                        reason=e.get("reason", "")), env_hash=ehash)
            continue
        overlay = Path(e["overlay"])
        cfg = yaml.safe_load((overlay / "tilebench" / "benchmarks" / "operators" / op /
                              "config.yaml").read_text())
        all_cases = dict(case_sel.indexed_cases(op, cfg))
        # The correctness contract is main's config; a branch-only verify override
        # (a looser tolerance) is recorded, never applied.
        main_cfg_path = REPO / "tilebench" / "benchmarks" / "operators" / op / "config.yaml"
        main_verify = (yaml.safe_load(main_cfg_path.read_text()) or {}).get("verify") or {} \
            if main_cfg_path.is_file() else None
        branch_verify = cfg.get("verify") or {}
        vcfg = main_verify if main_verify is not None else branch_verify
        verify_note = {"verify_source": "main config.yaml" if main_verify is not None else "branch config.yaml",
                       "branch_verify_override": branch_verify if branch_verify != (main_verify or {}) else None}
        for cid in _select(op, cfg, a.cases):
            case = all_cases[cid]
            params = {k: v for k, v in case.items() if k not in ("dtype", "block_size")}
            dtype = case.get("dtype", "fp32")
            plan_rows.append({"operator": op, "case_id": cid, "params": params, "dtype": dtype})
            _run_case(store, a, e, overlay, cid, case, params, dtype, vcfg, modes, py_for, hw, ehash,
                      verify_note)
    store.write_new(f"plans/{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.json",
                    {"argv": sys.argv, "cases": plan_rows})


def _bundles(store, a, e, overlay, cid, params, dtype, py) -> dict:
    op = e["operator"]
    bdir = store.path("bundles", op, f"c{cid:03d}")
    res_path = bdir / "bundle_result.json"
    if res_path.is_file():
        return json.loads(res_path.read_text())
    bdir.mkdir(parents=True, exist_ok=True)
    spec = {"operator": op, "params": params, "dtype": dtype, "seed": 1000 + cid,
            "dir": str(bdir), "out": str(res_path)}
    (bdir / "bundle_spec.json").write_text(json.dumps(spec))
    status, _ = _run_child([py, "-m", "tilebench.neuron_diag.bundles", "--spec",
                            str(bdir / "bundle_spec.json")], overlay=overlay, env_extra={},
                           timeout=a.bundle_timeout, log_path=store.path("logs", op, f"c{cid:03d}_bundle.log"))
    if not res_path.is_file():
        return {"error": f"bundle worker {status}"}
    return json.loads(res_path.read_text())


def _run_case(store, a, e, overlay, cid, case, params, dtype, vcfg, modes, py_for, hw, ehash,
              verify_note=None):
    op, shash = e["operator"], e["source_hash"]
    todo = [m for m in modes
            if not (a.resume_skip and store.resume_state(op, cid, m, source_hash=shash,
                                                          env_hash=ehash) == "done")]
    if not todo:
        print(f"[skip] {op} c{cid} (all modes done)", flush=True)
        return
    base = dict(operator=op, case_id=cid, shape=params, dtype=dtype, source_hash=shash,
                hardware=hw, warmup=a.warmup, repeat=a.repeat, smoke=a.smoke)
    bres = _bundles(store, a, e, overlay, cid, params, dtype, a.bundle_python or sys.executable)
    if "error" in bres:
        for m in todo:
            store.append(Record(mode=m, status="unsupported_dtype_shape",
                                reason=f"input generation / CPU reference failed: {bres['error']}"[:2000],
                                **base), env_hash=ehash)
        print(f"[ref-fail] {op} c{cid} {dtype}: {bres['error'][:120]}", flush=True)
        return
    recs: dict[str, dict] = {}
    for m in todo:
        stack = EXECUTION_MODES[m][0]
        blocked = _BLOCKED_STACK.get((stack, py_for[stack]))
        if blocked:
            rec = dict(base, mode=m, status="blocked_env",
                       reason=f"{blocked} (runtime probe result cached from the first case of this invocation)",
                       device_unavailable_reason="not executed (blocked_env)")
            rec["hardware"] = hw
            recs[m] = rec
            print(f"[{'blocked_env':>22}] {op} c{cid} {dtype} {m} (cached probe)", flush=True)
            continue
        out = store.path("workers", op, f"c{cid:03d}", f"{m}.json")
        if out.exists():
            out = out.with_name(f"{m}.{int(time.time())}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        spec = {"operator": op, "case_id": cid, "mode": m, "params": params, "dtype": dtype,
                "block_size": case.get("block_size", 1024), "source_hash": shash,
                "config": {"verify": vcfg, "block_size": case.get("block_size", 1024),
                           **(verify_note or {})},
                "bundle": bres["main"]["path"], "bundle_sha256": bres["main"]["file_sha256"],
                "alt_bundle": bres["alt"]["path"], "alt_bundle_sha256": bres["alt"]["file_sha256"],
                "atol": float(vcfg["atol"]) if "atol" in vcfg else None,
                "rtol": float(vcfg["rtol"]) if "rtol" in vcfg else None,
                "warmup": a.warmup, "repeat": a.repeat, "smoke": a.smoke, "out": str(out),
                "device_unavailable_reason": (
                    "XLA device timing taken by the runtime-inspect phase" if stack == "xla"
                    else "no native device-trace timing verified for this build")}
        spec_path = out.with_suffix(".spec.json")
        spec_path.write_text(json.dumps(spec))
        t0 = time.time()
        # Private compile cache per (operator, case, stack): the first call of the first
        # mode on a stack is a cold compile, not a hit in a cache shared with other runs.
        cache = store.path("compile_cache", op, f"c{cid:03d}", stack)
        cache.mkdir(parents=True, exist_ok=True)
        wenv = _worker_env(stack, py_for[stack], cache)
        spec["config"]["compile_cache"] = {"dir": str(cache), "cold_before_first_mode": not any(cache.iterdir())}
        spec["config"]["neuronx_cc_on_path"] = shutil.which("neuronx-cc", path=wenv.get("PATH", os.environ.get("PATH")))
        spec_path.write_text(json.dumps(spec))
        st, rc = _run_child([py_for[stack], "-m", "tilebench.neuron_diag.worker", "--spec", str(spec_path)],
                            overlay=overlay, env_extra=wenv,
                            timeout=a.case_timeout, log_path=store.path("logs", op, f"c{cid:03d}_{m}.log"))
        if st == "timeout":
            rec = {"mode": m, "status": "timeout", "reason": f"worker exceeded {a.case_timeout}s"}
        elif out.is_file():
            rec = json.loads(out.read_text())
        else:
            rec = {"mode": m, "status": "runtime_failure", "reason": f"worker exited rc={rc} without a record"}
        rec.update({k: v for k, v in base.items() if k not in rec or k in ("hardware", "smoke")})
        rec["artifact_path"] = str(out)
        if a.warm_recheck and rec.get("status") in ("pass", "cpu_fallback"):
            rec.setdefault("compile_info", {})["persistent_cache"] = _warm_recheck(
                a, py_for[stack], overlay, stack, spec, out, cache, store, op, cid, m)
        recs[m] = rec
        if rec.get("status") == "blocked_env":
            _BLOCKED_STACK[(stack, py_for[stack])] = rec.get("reason", "")
        print(f"[{rec['status']:>22}] {op} c{cid} {dtype} {m} wall="
              f"{rec.get('wall_ms') and round(rec['wall_ms'], 4)} ({time.time() - t0:.0f}s)", flush=True)

    if a.profiling_crosscheck:
        _profiling_crosscheck(store, a, overlay, cid, recs, py_for["xla"])
    if a.device_timing and any(EXECUTION_MODES[m][0] == "xla" and recs[m].get("status") in ("pass", "cpu_fallback")
                               for m in recs):
        _xla_device_phase(store, a, e, overlay, cid, params, dtype, case, vcfg, bres, recs, py_for["xla"])
    for m, rec in recs.items():
        if rec.get("status") not in ("pass", "cpu_fallback"):
            rec["device_unavailable_reason"] = f"not executed to completion (status {rec.get('status')})"
        elif not a.device_timing and EXECUTION_MODES[m][0] == "xla":
            rec["device_unavailable_reason"] = "device timing phase disabled (--no-device-timing)"
        store.append(Record.from_json(rec), env_hash=ehash)


def _warm_recheck(a, py, overlay, stack, spec, out, cache, store, op, cid, m) -> dict:
    """Run the same mode again in a fresh process with the same (now populated) compile
    cache: its first call shows whether the persistent cache is hit."""
    spec2 = dict(spec, out=str(out.with_name(f"{m}.warm.{int(time.time())}.json")))
    p2 = Path(spec2["out"]).with_suffix(".spec.json")
    p2.write_text(json.dumps(spec2))
    st, rc = _run_child([py, "-m", "tilebench.neuron_diag.worker", "--spec", str(p2)],
                        overlay=overlay, env_extra=_worker_env(stack, py, cache),
                        timeout=a.case_timeout, log_path=store.path("logs", op, f"c{cid:03d}_{m}_warm.log"))
    r2 = json.loads(Path(spec2["out"]).read_text()) if Path(spec2["out"]).is_file() else {}
    return {"status": r2.get("status", st), "warm_first_call_ms": r2.get("first_call_ms"),
            "warm_wall_ms": r2.get("wall_ms"), "cache_dir": str(cache), "record": spec2["out"]}


def _profiling_crosscheck(store, a, overlay, cid, recs, py):
    """Wall time of the same XLA-stack run() with the runtime-inspect profiler on
    (the setting the device-timing phase uses), to measure observer perturbation."""
    for m in ("xla_torch", "xla_nki"):
        rec = recs.get(m)
        if not rec or rec.get("status") != "pass":
            continue
        op = rec["operator"]
        spec_path = Path(rec["artifact_path"]).with_suffix(".spec.json")
        spec = json.loads(spec_path.read_text())
        out = spec_path.with_name(f"{m}.profiled.{int(time.time())}.json")
        spec["out"] = str(out)
        pspec = out.with_suffix(".spec.json")
        pspec.write_text(json.dumps(spec))
        inspect_dir = store.path("crosscheck", op, f"c{cid:03d}", m)
        inspect_dir.mkdir(parents=True, exist_ok=True)
        env = dict(XLA_ENV, NEURON_RT_INSPECT_ENABLE="1", NEURON_RT_INSPECT_DEVICE_PROFILE="1",
                   NEURON_RT_INSPECT_SYSTEM_PROFILE="1", NEURON_RT_INSPECT_OUTPUT_DIR=str(inspect_dir))
        st, _ = _run_child([py, "-m", "tilebench.neuron_diag.worker", "--spec", str(pspec)],
                           overlay=overlay, env_extra=env, timeout=a.case_timeout,
                           log_path=store.path("logs", op, f"c{cid:03d}_{m}_profiled.log"))
        pr = json.loads(out.read_text()) if out.is_file() else {"status": st}
        rec["compile_info"]["profiling_crosscheck"] = {
            "status": pr.get("status"), "wall_ms_profiler_on": pr.get("wall_ms"),
            "wall_ms_profiler_off": rec.get("wall_ms"),
            "wall_split_profiler_on": (pr.get("compile_info") or {}).get("wall_split_ms"),
            "record": str(out)}


def _xla_device_phase(store, a, e, overlay, cid, params, dtype, case, vcfg, bres, recs, py):
    op = e["operator"]
    out = store.path("device", op, f"c{cid:03d}", f"xla_device.{int(time.time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = {"operator": op, "params": params, "dtype": dtype,
            "block_size": case.get("block_size", 1024),
            "bundle": bres["main"]["path"], "bundle_sha256": bres["main"]["file_sha256"],
            "atol": float(vcfg["atol"]) if "atol" in vcfg else None,
            "rtol": float(vcfg["rtol"]) if "rtol" in vcfg else None,
            "warmup": a.warmup, "repeat": a.repeat,
            "base_dir": str(store.path("device", op, "nki_profiles")),
            "index_path": str(store.path("device", op, "nki_neff_manifest.jsonl")),
            "out": str(out)}
    spec_path = out.with_suffix(".spec.json")
    spec_path.write_text(json.dumps(spec))
    st, rc = _run_child([py, "-m", "tilebench.neuron_diag.xla_device", "--spec", str(spec_path)],
                        overlay=overlay, env_extra=XLA_ENV, timeout=a.device_timeout,
                        log_path=store.path("logs", op, f"c{cid:03d}_xla_device.log"))
    res = json.loads(out.read_text()) if out.is_file() else {"error": f"device phase {st} rc={rc}"}
    for m, target in (("xla_torch", "torch"), ("xla_nki", "nki")):
        rec = recs.get(m)
        if rec is None or rec.get("status") not in ("pass", "cpu_fallback"):
            continue
        if "error" in res:
            rec["device_unavailable_reason"] = f"runtime-inspect phase failed: {res['error']}"[:2000]
            continue
        st_t = (res["targets"].get(target) or {}).get("stats")
        orch = res["orchestrator"]
        if not st_t:
            why = orch.get("torch_err") if target == "torch" else orch.get("nki_err")
            rec["device_unavailable_reason"] = f"runtime-inspect: {why or 'no stats'}"[:2000]
            continue
        rec.update(device_ms=st_t["mean"], device_samples_ms=st_t.get("per_iteration_ms", []),
                   timing_method="neuron_rt_inspect",
                   artifact_identity=st_t.get("neff_sha256s", []),
                   device_unavailable_reason="")
        rec["compile_info"]["device"] = {
            "metric": "per-iteration sum of execution durations; per-core rows of one "
                      "execution merged by flow_id (tilebench.core.nki_timer)",
            "executions_per_iteration": st_t.get("executions_per_iteration"),
            "per_model": st_t.get("per_model"),
            "manifest_path": orch.get("manifest_path"),
            "per_iteration_intervals": (res["targets"][target].get("per_iteration_intervals") or [])[:8],
        }


# ---------------------------------------------------------------- report
def cmd_report(a) -> None:
    from tilebench.neuron_diag import report

    store = _store(a.run_id)
    path = report.write_report(store, legacy_xla=a.legacy_xla)
    print(f"report -> {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("env")
    p.add_argument("--run-id", required=True)
    p.add_argument("--stack-label", default="native")
    p.add_argument("--resume", action="store_true", help="add a snapshot to an existing run")
    p.set_defaults(fn=cmd_env)

    p = sub.add_parser("sources")
    p.add_argument("--run-id", required=True)
    p.add_argument("--prs", help="JSON list of PRs (number, state, head, sha) for the manifest")
    p.set_defaults(fn=cmd_sources)

    p = sub.add_parser("compat-patch", help="apply a declared runtime compatibility patch")
    p.add_argument("--run-id", required=True)
    p.add_argument("--ops", required=True)
    p.set_defaults(fn=cmd_compat_patch)

    p = sub.add_parser("native-device", help="native device timing for recorded native rows")
    p.add_argument("--run-id", required=True)
    p.add_argument("--ops", required=True)
    p.add_argument("--cases", default="all")
    p.add_argument("--modes", default="native_torch_eager,native_nki")
    p.add_argument("--native-python", required=True)
    p.add_argument("--device-timeout", type=int, default=1800)
    p.set_defaults(fn=cmd_native_device)

    p = sub.add_parser("run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--stack-label", default="native", help="which env snapshot the rows refer to")
    p.add_argument("--ops", required=True, help="comma list or 'all'")
    p.add_argument("--cases", default="smoke", help="'smoke', 'pilot' or explicit ids '0,5'")
    p.add_argument("--modes", default=",".join(BENCHMARK_MODES))
    p.add_argument("--legacy-xla-diagnostics", action="store_true",
                   help="allow the legacy xla_* diagnostic modes (never part of the benchmark)")
    p.add_argument("--compile-diagnostics", action="store_true",
                   help="allow native_torch_compiled (torch.compile diagnostic, never part of the benchmark)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--smoke", action="store_true", help="mark rows as smoke measurements")
    p.add_argument("--no-device-timing", dest="device_timing", action="store_false")
    p.add_argument("--warm-recheck", action="store_true",
                   help="re-run each mode in a fresh process on the same compile cache (persistent-cache check)")
    p.add_argument("--profiling-crosscheck", action="store_true",
                   help="re-time the XLA wall phase with the runtime-inspect profiler enabled")
    p.add_argument("--no-resume-skip", dest="resume_skip", action="store_false")
    p.add_argument("--case-timeout", type=int, default=3600)
    p.add_argument("--device-timeout", type=int, default=5400)
    p.add_argument("--bundle-timeout", type=int, default=1800)
    p.add_argument("--xla-python", help="legacy XLA diagnostics only")
    p.add_argument("--native-python")
    p.add_argument("--bundle-python", help="interpreter that builds the CPU input/reference bundles "
                   "(default: this interpreter; CPU only, no device)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report")
    p.add_argument("--run-id", required=True)
    p.add_argument("--legacy-xla", action="store_true",
                   help="write the archived XLA diagnostic view to legacy_xla/report.md instead")
    p.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
