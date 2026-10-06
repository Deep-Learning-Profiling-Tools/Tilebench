"""Calibration orchestration and artifact writer.

Writes ONLY into a fresh calibration directory (default
artifacts/llm_v2/calibration/<device>/<calibration_id>/). It refuses any
output path inside the legacy peak table directory, results/, or the
package tree, and never touches tilebench/data/peak_performance/*.json.
There is no option to update the canonical peak table.

Layout of a calibration directory:
    protocol.json        the protocol (and its sha256)
    environment.json     device / toolchain / visible devices / default precision flags
    raw/<mode>.json      every point: raw per-batch samples (ms per launch), checks, trace summary
    profile.json         the empirical profile (schema.PROFILE_SCHEMA, sealed with profile_sha256)
    summary.json         compact per-mode table + validity
    SHA256SUMS           sha256 of every file above
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

from tilebench.llm.v2.calibration import environment, protocol, schema, stats
from tilebench.paths import PEAK_PERFORMANCE_ROOT, REPO_ROOT

CALIBRATION_ROOT = REPO_ROOT / "artifacts" / "llm_v2" / "calibration"
LEGACY_COMMIT = "db1176977759e30bec453677db54dee79b2e8bd7"
LEGACY_PATH = "results/peak_performance/peak_performance.json"
SOURCE_FILES = ("schema.py", "protocol.py", "stats.py", "environment.py", "gpu_probes.py", "run.py")

CUDA_MODES = ["hbm_stream_bw", "mma_fp16_f32acc", "mma_bf16_f32acc", "mma_tf32_f32acc", "gemm_fp32_ieee",
              "mma_fp8_e4m3_f32acc", "mma_int8_i32acc", "fp32_fma_vector", "fp16x2_fma_vector", "bf16x2_fma_vector"]
ROCM_MODES = ["hbm_stream_bw", "mma_fp16_f32acc", "mma_bf16_f32acc", "mma_xf32_f32acc", "gemm_fp32_ieee",
              "mma_fp8_e4m3fnuz_f32acc", "mma_int8_i32acc", "fp32_fma_vector", "fp16x2_fma_vector", "bf16x2_fma_vector"]


class CalibrationRefused(RuntimeError):
    pass


def detect_backend() -> str:
    try:
        import torch
    except ImportError:
        return "none"
    if getattr(torch.version, "hip", None):
        return "rocm"
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_neuronx  # noqa: F401
        return "neuron"
    except ImportError:
        return "none"


def default_modes(backend: str) -> list[str]:
    return {"cuda": CUDA_MODES, "rocm": ROCM_MODES}.get(backend, [])


def check_output_dir(out: Path) -> Path:
    out = out.resolve()
    forbidden = [PEAK_PERFORMANCE_ROOT.resolve(), (REPO_ROOT / "results").resolve(), (REPO_ROOT / "tilebench").resolve()]
    for f in forbidden:
        if out == f or f in out.parents:
            raise CalibrationRefused(f"refusing to write calibration output under {f} (legacy/canonical data)")
    if out.exists() and any(out.iterdir()):
        raise CalibrationRefused(f"{out} exists and is not empty; a calibration directory is written once")
    return out


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def code_record() -> dict:
    src = Path(__file__).parent
    return {"git_sha": (_git("rev-parse", "HEAD") or "unavailable").strip(),
            "dirty_paths": [l[3:] for l in (_git("status", "--porcelain") or "").splitlines() if l.strip()],
            "calibration_sources_sha256": {f: hashlib.sha256((src / f).read_bytes()).hexdigest() for f in SOURCE_FILES}}


def legacy_reference() -> dict | None:
    txt = _git("show", f"{LEGACY_COMMIT}:{LEGACY_PATH}")
    if not txt:
        return None
    d = json.loads(txt)
    s = d.get("summary", {})
    return {"source_kind": "measured_legacy", "commit": LEGACY_COMMIT, "path": LEGACY_PATH,
            "timestamp": d.get("timestamp"), "pytorch": d.get("pytorch"), "cuda": d.get("cuda"),
            "values": {"hbm_copy_GBs": s.get("peak_hbm_bw_GBs"), "tflops": s.get("peak_tflops")},
            "interpretation": "observed sustained rates of the 2026-04-08 benchmark (single run, medians only, "
                              "precision path of 'fp32' not recorded; int8 is a matrix dot in TOP/s). Comparison only."}


def datasheet_reference(device: str) -> dict | None:
    p = PEAK_PERFORMANCE_ROOT / f"{device}.json"
    if not p.is_file():
        return None
    raw = p.read_bytes()
    d = json.loads(raw)
    return {"source_kind": "datasheet", "path": str(p.relative_to(REPO_ROOT)), "sha256": hashlib.sha256(raw).hexdigest(),
            "peak_tflops": d.get("peak_tflops"), "source": d.get("source"),
            "note": "compute entries are datasheet dense values; the file's peak_bw_GBs is the 2026-04-08 measured "
                    "copy rate (see legacy_reference). Reference only; never a scoring ceiling."}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")


def _mode_entry(name: str, result: dict, proto: dict) -> dict:
    reg = schema.MODES[name]
    base = {"unit": reg["unit"], "source_kind": "measured", "role": reg["role"], "meaning": {k: v for k, v in reg.items()
                                                                                          if k not in ("unit", "role")}}
    if result.get("status") in ("unsupported", "not_measured"):
        return {**base, "status": result["status"], "value": None, "reason": result.get("reason")}
    pts = result.get("points", [])
    sel = stats.select_mode_value(pts, bandwidth=(name == "hbm_stream_bw"),
                                  plateau_tolerance=proto["hbm"]["plateau_tolerance"])
    flagged = [p["id"] for p in pts if p.get("batch_spread") is not None and p["batch_spread"] > proto["batch_spread_flag"]]
    entry = {**base, "status": sel["status"], "value": sel["value"], "selected_point": sel.get("selected"),
             "reason": sel.get("reason"), "probe": {k: v for k, v in result.items() if k != "points"},
             "points": [{k: p.get(k) for k in ("id", "valid", "throughput", "point_time_s", "batch_spread", "hbm_eligible",
                                               "primary", "role", "check", "device_vs_event_time", "error")}
                        | {"busy_fraction": (p.get("trace") or {}).get("busy_fraction")} for p in pts],
             "points_batch_spread_flagged": flagged}
    if "plateau" in sel:
        entry["plateau"] = sel["plateau"]
    if name == "gemm_fp32_ieee":
        entry["limitations"] = ["control only: library SGEMM rate, never a scoring ceiling and not a vector FMA rate"]
    if reg["kind"] == "vector":
        entry["limitations"] = ["dependent FMA chains in registers; the rate of other vector instructions "
                                "(exp, compare, convert) is not measured by this probe"]
    if reg["kind"] == "matrix" and name != "gemm_fp32_ieee":
        entry["limitations"] = ["library GEMM (vendor BLAS) on square shapes 4096..16384; the sustained rate of "
                                "this library on these shapes, not a proof of the hardware maximum"]
    return entry


def run_calibration(device: str, out: Path | None = None, *, modes: list[str] | None = None, quick: bool = False,
                    index: int = 0, lock_timeout_s: float = 600.0, log=print) -> dict:
    from tilebench import hardware
    from tilebench.llm.v2.manifests import schema as ms
    from tilebench.llm.v2.orchestration.locks import device_lock

    study = ms.load_study()
    if device not in study["support_matrix"]:
        raise CalibrationRefused(f"unknown device {device!r}")
    backend = detect_backend()
    if device == "Trn2" or backend == "neuron":
        raise CalibrationRefused(
            "Trn2/Neuron: the native calibration adapter is not implemented in the shared code. It is written in the "
            "authorized NKI window against this same profile schema (docs/llm_v2/CALIBRATION.md, 'Trn2'): fix the "
            "Neuron device, LNC, visible logical/physical cores, launch grid and HBM address space first; time real "
            "inputs with complete device execution; never divide a chip-level rate by a core count.")
    if backend not in ("cuda", "rocm"):
        raise CalibrationRefused(f"no supported accelerator backend detected ({backend})")
    expected = study["support_matrix"][device]["arch"]
    arch = hardware.detect_arch()
    if arch != expected:
        raise CalibrationRefused(f"this host detects arch {arch!r}; device {device} expects {expected!r}")
    proto = dict(protocol.QUICK if quick else protocol.DEFAULT)
    modes = modes or default_modes(backend)
    unknown = [m for m in modes if m not in schema.MODES]
    if unknown:
        raise CalibrationRefused(f"unregistered modes {unknown} (no aliases; see calibration/schema.py)")
    blas = environment.blas_stack_check()
    if not blas["ok"]:
        raise CalibrationRefused(
            f"inconsistent or failing BLAS stack ({blas.get('error')}); LD_LIBRARY_PATH={blas.get('LD_LIBRARY_PATH')}. "
            "Run with a consistent library stack (on dgx003: unset LD_LIBRARY_PATH so the wheel's cuBLAS/cuBLASLt pair "
            "is used).")
    code = code_record()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    cal_id = f"{device}-{stamp}-{code['git_sha'][:8]}" + ("-quick" if quick else "")
    out = check_output_dir(out or (CALIBRATION_ROOT / device / cal_id))

    import torch
    from tilebench.llm.v2.calibration import gpu_probes as gp
    dev = torch.device("cuda", index)
    torch.cuda.set_device(dev)
    with device_lock(device, timeout_s=lock_timeout_s):
        env = environment.capture(backend, index)
        env["blas_stack_check"] = blas
        others = env.get("other_compute_processes")
        if isinstance(others, list):
            import os
            others = [o for o in others if o.split(",")[0].strip() != str(os.getpid())]
            env["other_compute_processes"] = others
        if isinstance(others, list) and others:
            raise CalibrationRefused(f"other compute processes are on the device: {others}; calibration is exclusive")
        try:
            llc = hardware.last_level_cache_bytes()
        except Exception as e:  # noqa: BLE001  (uncalibrated LLC: HBM points are never eligible)
            llc = None
            env["llc_error"] = str(e)
        props = env.get("device_properties", {})
        scope = {"kind": "single_device_full", "device_index": index, "backend": backend,
                 "name": props.get("name"), "uuid": props.get("uuid"),
                 "sm_or_cu_count": props.get("multi_processor_count"),
                 "llc_bytes": llc, "mig_mode": env.get("nvidia_smi", {}).get("mig.mode.current", "unavailable"),
                 "visible_devices": env["visible_devices"],
                 "note": "the whole visible device as one evaluation scope (no MIG/partition); the same scope the "
                         "evaluator times candidates on"}
        out.mkdir(parents=True, exist_ok=True)
        _write(out / "protocol.json", {**proto, "protocol_sha256": protocol.protocol_hash(proto)})
        _write(out / "environment.json", env)
        results: dict[str, dict] = {}
        samples: dict[str, dict] = {}
        for name in modes:
            log(f"[calibrate] {name} ...")
            before = environment.device_sample(backend, index)
            t0 = time.time()
            try:
                if name == "hbm_stream_bw":
                    pts = []
                    for probe in proto["hbm"]["primary_probes"] + proto["hbm"]["diagnostic_probes"]:
                        for size in proto["hbm"]["sizes_mib"]:
                            pts.append(gp.probe_stream(probe, size, proto, dev, llc))
                    res = {"api": "torch copy_/fill_ + Triton SM streaming kernels", "points": pts,
                           "llc_bytes": llc, "selection": "max over primary probes at working set >= "
                           f"{proto['hbm']['min_working_set_over_llc']} x LLC, plateau required"}
                elif name in gp.GEMM_SPECS:
                    res = gp.probe_gemm(name, proto, dev)
                elif name in gp.FMA_SPECS:
                    res = gp.probe_fma(name, proto, dev, int(props.get("multi_processor_count") or 0), backend)
                else:
                    res = {"status": "not_measured", "reason": "no probe registered for this mode"}
            except Exception as e:  # noqa: BLE001  (recorded; the mode fails, others continue)
                res = {"status": "failed", "points": [], "error": f"{type(e).__name__}: {e}"}
            res["device_before"] = before
            res["device_after"] = environment.device_sample(backend, index)
            res["wall_s"] = time.time() - t0
            results[name] = res
            samples[name] = res
            _write(out / "raw" / f"{name}.json", res)
            log(f"[calibrate] {name}: done in {res['wall_s']:.1f} s")
    profile = {
        "schema": schema.PROFILE_SCHEMA, "ceiling_basis": schema.CEILING_BASIS, "calibration_id": cal_id,
        "device": device, "scope": scope,
        "protocol": {"id": proto["protocol"], "sha256": protocol.protocol_hash(proto), "quick": quick},
        "code": code,
        "environment": {"file": "environment.json",
                        "sha256": hashlib.sha256((out / "environment.json").read_bytes()).hexdigest()},
        "modes": {name: _mode_entry(name, res, proto) for name, res in results.items()},
        "datasheet_reference": datasheet_reference(device),
        "legacy_reference": legacy_reference() if device == "B200" else None,
        "created_at": time.time(),
    }
    for name in schema.MODES:
        profile["modes"].setdefault(name, {"unit": schema.MODES[name]["unit"], "source_kind": "measured",
                                           "role": schema.MODES[name]["role"], "status": "not_measured", "value": None,
                                           "reason": "not in this run's mode list"})
    schema.seal(profile)
    errs = schema.validate_profile(profile)
    if errs:
        raise CalibrationRefused(f"profile failed schema validation: {errs}")
    _write(out / "profile.json", profile)
    summary = {"calibration_id": cal_id, "device": device, "profile_sha256": profile["profile_sha256"],
               "modes": {n: {"status": m["status"], "unit": m["unit"], "value": m.get("value"),
                             "selected_point": m.get("selected_point"), "reason": m.get("reason")}
                         for n, m in profile["modes"].items()}}
    _write(out / "summary.json", summary)
    lines = []
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return {"out": str(out), **summary}
