"""GH200-only supplementary INT8 MMA calibration (owner decision 2026-10-07).

The base GH200 calibration GH200-20261007T203829Z-a2fe6ae1 measured mma_int8_i32acc with the shared probe's
operands (torch._int_mm, A row-major x B row-major). On this host's cuBLAS that layout dispatches a legacy
sm_80 WMMA forward-compat kernel. This script re-runs ONLY that probe with A row-major x B column-major
(identical values, same seeds, sizes, warm-up, repetitions, timing, checks and validity rules: the shared
framework's own functions, imported unchanged) and writes a NEW derived calibration directory:

    raw/<mode>.json                      every other mode: byte-identical copies of the base calibration
    raw/mma_int8_i32acc.json             both layout variants' points (base points verbatim + the new ones)
    raw_supplement/mma_int8_i32acc__A_row_B_col.json   the new measurement as recorded
    environment.json                     the base calibration's environment (copied)
    environment_int8_supplement.json     the environment captured for the supplementary measurement
    protocol.json, profile.json (sealed), summary.json, SHA256SUMS

The int8 value is the shared selection rule (stats.select_mode_value: maximum over the valid, verified points)
applied to the union of both variants. Every other mode entry is asserted identical to the base profile.
The base calibration directory is never written.

    python scripts/llm_v2_gh200/int8_layout_supplement.py      (GH200 host, LD_LIBRARY_PATH unset, exclusive GPU)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tilebench.llm.v2.calibration import environment, protocol, run, schema, stats  # noqa: E402

DEVICE = "GH200"
MODE = "mma_int8_i32acc"
BASE_ID = "GH200-20261007T203829Z-a2fe6ae1"
BASE = run.CALIBRATION_ROOT / DEVICE / BASE_ID
VARIANT = "A_row_B_col"
BASE_VARIANT = "A_row_B_row"
DECISION = ("study owner, 2026-10-07: minimal GH200-only INT8 calibration correction; rerun only the INT8 MMA probe with "
            "row-major A x column-major B under the same protocol; ceiling = fastest verified sustained INT8 tensor-core "
            "throughput among the measured valid layout variants; both variants preserved")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def layout(t) -> dict:
    return {"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype),
            "order": "row-major" if t.stride()[-1] == 1 else ("column-major" if t.stride()[0] == 1 else "strided")}


def probe_int8_b_col(proto: dict, dev) -> dict:
    """gpu_probes.probe_gemm for mma_int8_i32acc with B column-major: same inputs (gpu_probes._gemm_inputs), the
    same check, measurement, trace and validity functions; only B's strides differ."""
    import torch
    from tilebench.llm.v2.calibration import gpu_probes as gp
    api, dt, fp32_path, rpr, _ = gp.GEMM_SPECS[MODE]
    points = []
    with gp.precision(fp32_path, rpr):
        flags = gp.effective_flags()
        for m in proto["gemm"]["sizes"]:
            pid = f"{MODE}/{VARIANT}/M{m}"
            try:
                a, b_row, _ = gp._gemm_inputs(MODE, m, dev)          # identical seeds and values as the base probe
                b = b_row.t().contiguous().t()                       # same values, column-major (K, N)
                assert torch.equal(b, b_row)
                del b_row
                fn = (lambda a=a, b=b: torch._int_mm(a, b))
                c = fn()
                gp._sync()
                check = gp._gemm_check(MODE, a, b, c, proto["gemm"]["check_block"])
                lay = {"A": layout(a), "B": layout(b), "C": layout(c)}
                del c
                meas = gp.measure(fn, proto)
                tr = gp.trace_check(fn, max(5, min(meas["inner"], 40)))
            except (RuntimeError, TypeError, ValueError) as e:
                points.append({"id": pid, "M": m, "layout": VARIANT, "valid": False, "error": f"{type(e).__name__}: {e}"})
                torch.cuda.empty_cache()
                continue
            t_s = stats.point_time(meas["batches_ms"]) / 1e3
            work = 2.0 * m * m * m
            dev_vs_event = (tr["device_us_per_launch"] / 1e6) / t_s if tr.get("available") else None
            valid = bool(check["ok"] and gp._device_bound(tr, dev_vs_event, proto))
            points.append({"id": pid, "M": m, "N": m, "K": m, "work": work, "work_unit": "OP", "layout": VARIANT,
                           "operand_layout": lay,
                           "inner": meas["inner"], "batches_ms": meas["batches_ms"],
                           "batch_medians_ms": stats.batch_medians(meas["batches_ms"]),
                           "point_time_s": t_s, "throughput": work / t_s,
                           "batch_spread": stats.batch_spread(meas["batches_ms"]),
                           "check": check, "trace": tr, "device_vs_event_time": dev_vs_event, "valid": valid})
            del a, b
            torch.cuda.empty_cache()
    return {"api": api, "call": "torch._int_mm(A, B), A row-major (M, K), B column-major (K, N) = B.t().contiguous().t()",
            "input_dtype": str(dt), "precision_flags_effective": flags, "points": points}


def main() -> int:
    import torch
    from tilebench import hardware
    from tilebench.llm.v2.orchestration.locks import device_lock

    proto = dict(protocol.DEFAULT)
    base_proto = json.loads((BASE / "protocol.json").read_text())
    if {k: v for k, v in base_proto.items() if k != "protocol_sha256"} != proto:
        raise SystemExit("protocol differs from the base calibration's protocol.json")
    base_profile = json.loads((BASE / "profile.json").read_text())
    if schema.validate_profile(base_profile) or base_profile["calibration_id"] != BASE_ID:
        raise SystemExit("base profile invalid")
    if hardware.detect_arch() != "hopper":
        raise SystemExit("not a hopper host")
    blas = environment.blas_stack_check()
    if not blas["ok"]:
        raise SystemExit(f"BLAS stack: {blas.get('error')}")
    code = run.code_record()                  # before anything is written; the script is committed
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    cal_id = f"{DEVICE}-{stamp}-{code['git_sha'][:8]}-int8layout"
    out = run.check_output_dir(run.CALIBRATION_ROOT / DEVICE / cal_id)
    base_sums_before = (BASE / "SHA256SUMS").read_text()

    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    with device_lock(DEVICE, timeout_s=600.0):
        env = environment.capture("cuda", 0)
        env["blas_stack_check"] = blas
        others = env.get("other_compute_processes")
        if isinstance(others, list):          # as run.run_calibration: this process's own context is not "other"
            others = [o for o in others if o.split(",")[0].strip() != str(os.getpid())]
            env["other_compute_processes"] = others
        if isinstance(others, list) and others:
            raise SystemExit(f"other compute processes are on the device: {others}; calibration is exclusive")
        if env.get("device_properties", {}).get("uuid") != base_profile["scope"]["uuid"]:
            raise SystemExit("device uuid differs from the base calibration's scope")
        print(f"[int8-supplement] {MODE} {VARIANT} ...", flush=True)
        before = environment.device_sample("cuda", 0)
        t0 = time.time()
        tele = environment.Telemetry("cuda", 0, proto.get("telemetry_interval_ms", 250))
        tele.start()
        try:
            res = probe_int8_b_col(proto, dev)
        finally:
            res_tele = tele.stop()
        res["telemetry"] = res_tele
        res["device_before"] = before
        res["device_after"] = environment.device_sample("cuda", 0)
        res["wall_s"] = time.time() - t0
    print(f"[int8-supplement] done in {res['wall_s']:.1f} s", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    run._write(out / "raw_supplement" / f"{MODE}__{VARIANT}.json", res)
    run._write(out / "environment_int8_supplement.json", env)
    shutil.copyfile(BASE / "protocol.json", out / "protocol.json")
    shutil.copyfile(BASE / "environment.json", out / "environment.json")
    base_raw = {}
    (out / "raw").mkdir(parents=True, exist_ok=True)
    for f in sorted((BASE / "raw").glob("*.json")):
        base_raw[f.stem] = json.loads(f.read_text())
        if f.stem != MODE:
            shutil.copyfile(f, out / "raw" / f.name)

    # int8: union of both layout variants; base points verbatim (+ their layout label)
    b = base_raw[MODE]
    base_points = [dict(p, layout=BASE_VARIANT) for p in b["points"]]
    combined = {
        "api": b["api"], "input_dtype": b["input_dtype"], "precision_flags_effective": b["precision_flags_effective"],
        "layout_variants": {
            BASE_VARIANT: {"call": "torch._int_mm(A, B), A and B row-major (shared gpu_probes._gemm_inputs)",
                           "source": f"artifacts/llm_v2/calibration/{DEVICE}/{BASE_ID}/raw/{MODE}.json",
                           "source_sha256": sha256_file(BASE / "raw" / f"{MODE}.json"),
                           "point_ids": [p["id"] for p in b["points"]],
                           "kernels": sorted({n for p in b["points"] for n in (p.get("trace") or {}).get("probe_names", [])})},
            VARIANT: {"call": res["call"],
                      "source": f"raw_supplement/{MODE}__{VARIANT}.json",
                      "source_sha256": sha256_file(out / "raw_supplement" / f"{MODE}__{VARIANT}.json"),
                      "environment": "environment_int8_supplement.json",
                      "point_ids": [p["id"] for p in res["points"]],
                      "kernels": sorted({n for p in res["points"] for n in (p.get("trace") or {}).get("probe_names", [])})},
        },
        "selection": "shared stats.select_mode_value over the valid points of both layout variants (maximum verified "
                     "sustained throughput); every point keeps its own check, trace and validity",
        "decision": DECISION,
        "points": base_points + res["points"],
        "telemetry": {"summary": {BASE_VARIANT: (b.get("telemetry") or {}).get("summary"),
                                  VARIANT: (res.get("telemetry") or {}).get("summary")}},
        "device_before": {BASE_VARIANT: b.get("device_before"), VARIANT: res.get("device_before")},
        "device_after": {BASE_VARIANT: b.get("device_after"), VARIANT: res.get("device_after")},
        "wall_s": {BASE_VARIANT: b.get("wall_s"), VARIANT: res.get("wall_s")},
    }
    run._write(out / "raw" / f"{MODE}.json", combined)

    results = {name: (combined if name == MODE else base_raw[name]) for name in base_raw}
    modes = {name: run._mode_entry(name, r, proto) for name, r in results.items()}
    for name, entry in modes.items():
        if name != MODE and entry != base_profile["modes"][name]:
            raise SystemExit(f"mode {name} does not reproduce the base profile entry")
    profile = {
        "schema": schema.PROFILE_SCHEMA, "ceiling_basis": schema.CEILING_BASIS, "calibration_id": cal_id,
        "device": DEVICE, "scope": base_profile["scope"], "protocol": base_profile["protocol"], "code": code,
        "environment": {"file": "environment.json", "sha256": sha256_file(out / "environment.json")},
        "modes": modes,
        "datasheet_reference": base_profile.get("datasheet_reference"),
        "legacy_reference": base_profile.get("legacy_reference"),
        "derivation": {"kind": "gh200_int8_layout_supplement", "decision": DECISION,
                       "base_calibration_id": BASE_ID,
                       "base_profile_file_sha256": sha256_file(BASE / "profile.json"),
                       "base_profile_sha256": base_profile["profile_sha256"],
                       "base_code": base_profile["code"],
                       "changed_modes": [MODE],
                       "unchanged_modes": "every other mode entry equals the base profile's (asserted); raw files copied byte-identical",
                       "supplement_environment": {"file": "environment_int8_supplement.json",
                                                  "sha256": sha256_file(out / "environment_int8_supplement.json")},
                       "script": {"path": "scripts/llm_v2_gh200/int8_layout_supplement.py",
                                  "sha256": sha256_file(Path(__file__))}},
        "created_at": time.time(),
    }
    for name in schema.MODES:
        profile["modes"].setdefault(name, base_profile["modes"][name])
    schema.seal(profile)
    errs = schema.validate_profile(profile)
    if errs:
        raise SystemExit(f"profile failed schema validation: {errs}")
    run._write(out / "profile.json", profile)
    summary = {"calibration_id": cal_id, "device": DEVICE, "profile_sha256": profile["profile_sha256"],
               "derived_from": BASE_ID,
               "modes": {n: {"status": m["status"], "unit": m["unit"], "value": m.get("value"),
                             "selected_point": m.get("selected_point"), "reason": m.get("reason")}
                         for n, m in profile["modes"].items()}}
    run._write(out / "summary.json", summary)
    lines = [f"{sha256_file(f)}  {f.relative_to(out)}" for f in sorted(p for p in out.rglob("*")
                                                                         if p.is_file() and p.name != "SHA256SUMS")]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    if (BASE / "SHA256SUMS").read_text() != base_sums_before:
        raise SystemExit("base calibration changed")
    print(json.dumps({"out": str(out), "calibration_id": cal_id, "profile_sha256": profile["profile_sha256"],
                      MODE: {"value": modes[MODE]["value"], "selected_point": modes[MODE]["selected_point"],
                             "flags": modes[MODE]["flags"]},
                      "points": [(p["id"], p.get("valid"), p.get("throughput")) for p in combined["points"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
