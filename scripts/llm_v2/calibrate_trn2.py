#!/usr/bin/env python
"""Trn2 empirical calibration (native PyTorch Neuron + NKI) for the empirical Roofline of the LLM protocol.

The shared `calibrate` command refuses Trn2 by design (docs/llm_v2/CALIBRATION.md, "Trn2"); this standalone
adapter writes the same artifact layout and the same profile schema (calibration/schema.py
`tilebench-empirical-profile/1`, sealed and validated with the shared code, which is imported read-only).

Scope (fixed before measuring, recorded in the profile): one logical NeuronCore with LNC=2, i.e. two
physical NeuronCore-v3 of one Trainium2 device, NEURON_RT_NUM_CORES=1 -- the scope on which the NKI
candidates and the formal TileBench NKI results are timed. Per-physical-core points (one of the two cores
active) are recorded as diagnostics; whole-device (4 logical / 8 physical cores) rates are not measured.

Timing: device busy time of each execution from the Neuron runtime system trace (torch.profiler +
torch_neuronx.profiling.NeuronConfig(RUNTIME), `nc_exec_running` per execution, the per-core copies of one
execution merged), i.e. the formal NKI timing method. One profiler session per batch; the first call
(compilation) and the warmup calls are outside every session. Every timed call must be exactly one device
execution of one executable.

    python scripts/llm_v2/calibrate_trn2.py [--smoke] [--modes a,b] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PROTOCOL_ID = "tilebench-empirical-roofline/2+trn2-native"
SBUF_BYTES_PER_PCORE = 24 * 1024 * 1024        # NeuronCore-v3 SBUF (on-chip, software-managed; no HBM cache)

PROTOCOL = {
    "protocol": PROTOCOL_ID,
    "base_protocol": "tilebench-empirical-roofline/2 (calibration/protocol.py); Trn2 adaptation below",
    "timing": "device busy time per execution from the Neuron runtime system trace (nc_exec_running, per-core copies "
              "merged), the formal NKI timing method; no host interval; compilation and warmup outside the timed sessions",
    "batches": 3,
    "warmup": 30,
    "repeat": 50,
    "statistic": "per point: median of the executions within each batch, then median of the batch medians; "
                 "mode value: maximum over the valid points (stats.select_mode_value)",
    "batch_spread_flag": 0.03,
    "saturation_tolerance": 0.02,
    "validity": "correctness check passes AND every timed call is exactly one device execution of one executable",
    "hbm": {"primary_probes": ["nki_copy_via_sbuf", "nki_copy_dram_to_dram"], "diagnostic_probes": ["torch_copy"],
            "sizes_mib": [16, 64, 256, 1024, 2048], "min_working_set_over_llc": 4.0, "plateau_tolerance": 0.05,
            "llc_equivalent": "SBUF capacity of the scope (2 x 24 MiB): Trn2 has no hardware cache between HBM and SBUF; "
                              "working sets >= 4 x 48 MiB are HBM-scale",
            "work": "read + write bytes (2 x buffer size), random int32 payload, output compared in full"},
    "gemm": {"api": "torch.matmul on torch.device('neuron') (neuronx-cc compiled, the PyTorch eager library path)",
             "sizes": [4096, 8192, 12288, 16384], "check_block": 128,
             "check": "C[:128,:128] against a float64 product of the same (rounded) operands"},
    "vector": {"api": "NKI nisa.tensor_scalar(op0=multiply, op1=add) chains on SBUF-resident tiles, 2 FLOP per element",
               "partition": 128, "free": 2048, "chains_per_engine": 2, "iters": 400,
               "engine_sets": [["vector"], ["scalar"], ["gpsimd"], ["vector", "scalar"], ["vector", "scalar", "gpsimd"]],
               "scaling_check": "the same point at 2 x iters must take 2x the time within 10 %",
               "a": 0.999, "b": 0.001},
}
SMOKE = {**PROTOCOL, "protocol": PROTOCOL_ID + "+quick", "batches": 1, "warmup": 2, "repeat": 3,
         "hbm": {**PROTOCOL["hbm"], "sizes_mib": [16, 256]},
         "gemm": {**PROTOCOL["gemm"], "sizes": [2048]},
         "vector": {**PROTOCOL["vector"], "iters": 50, "engine_sets": [["vector"], ["vector", "scalar"]]}}


# ----------------------------------------------------------------------------- environment (before torch_neuronx)
def neuron_ls() -> str:
    try:
        return subprocess.run(["/opt/aws/neuron/bin/neuron-ls"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"unavailable: {e}"


def prepare_environment(cache_root: Path, extra_cc_flags: str = "") -> dict:
    """Same launch settings as tilebench/core/neuron_native.prepare_environment (main 1202eaf5)."""
    bindir = Path(sys.executable).parent
    if (bindir / "neuronx-cc").exists():
        os.environ["PATH"] = os.pathsep.join([str(bindir), os.environ.get("PATH", "")])
    os.environ["PATH"] = os.pathsep.join([os.environ.get("PATH", ""), "/opt/aws/neuron/bin"])
    lnc = re.search(r"logical-neuroncore-config:\s*(\d+)", neuron_ls())
    lnc = lnc.group(1) if lnc else None
    if lnc:
        os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", lnc)
        os.environ["NEURON_CC_FLAGS"] = (f"--target trn2 --lnc {lnc} " + extra_cc_flags).strip()
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
    for var, sub in (("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"), ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
                     ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "local"), ("NKI_TRACE_CACHE_URL", "nki_trace"),
                     ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TORCH_NEURONX_DEBUG_DIR", "debug")):
        os.environ.setdefault(var, str(cache_root / sub))
    return {k: os.environ[k] for k in sorted(os.environ) if k.startswith(("NEURON_", "TORCH_NEURONX_", "NKI_"))}


def capture_environment(env_vars: dict) -> dict:
    import importlib.metadata as md
    pk = {}
    for p in ("torch", "torch-neuronx", "nki", "neuronx-cc", "numpy"):
        try:
            pk[p] = md.version(p)
        except md.PackageNotFoundError:
            pk[p] = None
    debs = subprocess.run(["dpkg-query", "-W", "-f=${Package} ${Version}\n", "aws-neuronx-*"], capture_output=True,
                          text=True).stdout
    ls = neuron_ls()
    m = lambda pat: (re.search(pat, ls).group(1) if re.search(pat, ls) else None)
    row = re.search(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d-]+)\s*\|\s*([\d.]+\s*\w+)\s*\|", ls, re.M)
    try:
        driver = Path("/sys/module/neuron/version").read_text().strip()
    except OSError:
        driver = "unavailable"
    others = [l for l in ls.splitlines() if re.match(r"^\|\s*\d+\s*\|", l) and "neuron-ls" not in l]
    return {"instance_type": m(r"instance-type:\s*(\S+)"), "logical_neuroncore_config": m(r"logical-neuroncore-config:\s*(\d+)"),
            "neuron_devices": 1 if row else None, "logical_cores_per_device": int(row.group(2)) if row else None,
            "logical_core_ids": row.group(3) if row else None, "device_memory": row.group(4) if row else None,
            "physical_cores_per_device": (int(row.group(2)) * int(m(r"logical-neuroncore-config:\s*(\d+)") or 1)) if row else None,
            "packages": pk, "aws_neuronx_debs": {k: v for k, v in (l.split(" ", 1) for l in debs.splitlines() if " " in l) if v},
            "neuron_driver": driver, "python": sys.version.split()[0], "env": env_vars,
            "other_device_processes_at_start": others,
            "note": "instance id and host name deliberately not recorded"}


# ----------------------------------------------------------------------------- timing
class Timer:
    def __init__(self, workdir: Path, proto: dict):
        import torch_neuronx  # noqa: F401
        import torch
        self.torch = torch
        self.dev = torch.device("neuron")
        torch.empty(1, device=self.dev)
        self.sync = torch.neuron.synchronize if hasattr(torch, "neuron") else torch_neuronx.synchronize
        self.workdir = workdir
        self.proto = proto

    def batches(self, fn, tag: str) -> dict:
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
        from torch.profiler import ProfilerActivity, profile
        from torch_neuronx.profiling import NeuronConfig, ProfileMode
        from tilebench.core import nki_timer

        for _ in range(self.proto["warmup"]):
            fn()
            self.sync()
        out = []
        for b in range(self.proto["batches"]):
            d = self.workdir / f"{tag}_b{b}"
            shutil.rmtree(d, ignore_errors=True)
            cfg = NeuronConfig(modes=[ProfileMode.RUNTIME], profile_output_dir=str(d))
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1], experimental_config=cfg):
                self.sync()
                for _ in range(self.proto["repeat"]):
                    fn()
                    self.sync()
            sess = nki_timer.find_session_dir(str(d))
            execs, _ = nki_timer.load_session_executions(sess, data_path=str(d / "ne"), display_name=f"{tag}-{b}")
            out.append({"samples_ms": [(e["end_ns"] - e["start_ns"]) / 1e6 for e in execs],
                        "n_executions": len(execs), "model_ids": sorted({str(e["model_id"]) for e in execs}),
                        "pcores_per_execution": sorted({e["pcores"] for e in execs})})
            shutil.rmtree(d, ignore_errors=True)
        single = all(x["n_executions"] == self.proto["repeat"] and len(x["model_ids"]) == 1 for x in out) and \
            len({x["model_ids"][0] for x in out if x["model_ids"]}) == 1
        bm = [statistics.median(x["samples_ms"]) for x in out if x["samples_ms"]]
        return {"batches": out, "one_execution_per_call": single,
                "batch_medians_ms": bm, "point_time_s": statistics.median(bm) / 1e3 if bm else None,
                "batch_spread": ((max(bm) - min(bm)) / statistics.median(bm)) if len(bm) > 1 else None}


# ----------------------------------------------------------------------------- NKI kernels (module level: the NKI
# tracer resolves names from the kernel's module globals)
try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:  # pragma: no cover
    nki = None

ENGINE_CODES = {"vector": 0, "scalar": 1, "gpsimd": 2}

if nki is not None:
    _ENGINES = (nisa.engine.vector, nisa.engine.scalar, nisa.engine.gpsimd)

    @nki.jit
    def copy_via_sbuf(x, chunk_f, n_active):
        n = x.shape[0]
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        pid = nl.program_id(0)
        if pid < n_active:
            per = (n + n_active - 1) // n_active
            lo = pid * per
            hi = min(n, lo + per)
            chunk = 128 * chunk_f
            for j in range((hi - lo + chunk - 1) // chunk):
                start = lo + j * chunk
                f = min(chunk, hi - start) // 128
                tile = nl.ndarray((128, f), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=tile, src=x.ap(pattern=[[f, 128], [1, f]], offset=start))
                nisa.dma_copy(dst=out.ap(pattern=[[f, 128], [1, f]], offset=start), src=tile)
        return out

    @nki.jit
    def copy_dram_to_dram(x, chunk_f, n_active):
        n = x.shape[0]
        out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
        pid = nl.program_id(0)
        if pid < n_active:
            per = (n + n_active - 1) // n_active
            lo = pid * per
            hi = min(n, lo + per)
            chunk = 128 * chunk_f
            for j in range((hi - lo + chunk - 1) // chunk):
                start = lo + j * chunk
                f = min(chunk, hi - start) // 128
                nisa.dma_copy(dst=out.ap(pattern=[[f, 128], [1, f]], offset=start),
                              src=x.ap(pattern=[[f, 128], [1, f]], offset=start))
        return out

    @nki.jit
    def fma_chains(x0, iters, engines, a, b, n_active):
        F = x0.shape[1]
        out = nl.ndarray(x0.shape, dtype=x0.dtype, buffer=nl.shared_hbm)
        pid = nl.program_id(0)
        C = len(engines)
        active = pid < n_active
        # an inactive core runs the same instruction stream on 1x1 tiles (LNC=2 requires the same basic blocks on
        # both cores, NCC_IXGM002); its output rows are not written
        P, FF = (128, F) if active else (1, 1)
        tiles = []
        for c in range(C):
            t = nl.ndarray((P, FF), dtype=x0.dtype, buffer=nl.sbuf)
            r0 = (pid * C + c) * 128
            nisa.dma_copy(dst=t, src=x0[r0:r0 + P, 0:FF])
            tiles.append(t)
        for _ in nl.dynamic_range(iters):
            for c in range(C):
                nisa.tensor_scalar(dst=tiles[c], data=tiles[c], op0=nl.multiply, operand0=a,
                                   op1=nl.add, operand1=b, engine=_ENGINES[engines[c]])
        for c in range(C):
            r0 = (pid * C + c) * 128
            nisa.dma_copy(dst=out[r0:r0 + P, 0:FF], src=tiles[c])
        return out


def nki_kernels():
    return {"copy_via_sbuf": copy_via_sbuf, "copy_dram_to_dram": copy_dram_to_dram, "fma_chains": fma_chains}


# ----------------------------------------------------------------------------- probes
def probe_hbm(T: Timer, K: dict, probe: str, size_mib: int, proto: dict, llc_bytes: int, n_active: int = 2) -> dict:
    torch = T.torch
    nbytes = size_mib * 1024 * 1024
    pid = f"{probe}/{size_mib}MiB" + ("" if n_active == 2 else f"/pcores={n_active}")
    g = torch.Generator().manual_seed(77 + size_mib)
    src_cpu = torch.randint(-2**31, 2**31 - 1, (nbytes // 4,), dtype=torch.int32, generator=g)
    x = src_cpu.to(T.dev)
    T.sync()
    chunk_f = 4096                       # 128 x 4096 x 4 B = 2 MiB per DMA chunk
    if probe == "nki_copy_via_sbuf":
        k = K["copy_via_sbuf"][2]
        fn = lambda: k(x, chunk_f, n_active)
    elif probe == "nki_copy_dram_to_dram":
        k = K["copy_dram_to_dram"][2]
        fn = lambda: k(x, chunk_f, n_active)
    elif probe == "torch_copy":
        y = torch.empty_like(x)
        fn = lambda: y.copy_(x)
    else:
        raise ValueError(probe)
    res = {"id": pid, "probe": probe, "size_mib": size_mib, "physical_cores_active": n_active,
           "logical_bytes": 2 * nbytes, "working_set_bytes": 2 * nbytes,
           "primary": probe in proto["hbm"]["primary_probes"] and n_active == 2,
           "hbm_eligible": 2 * nbytes >= proto["hbm"]["min_working_set_over_llc"] * llc_bytes}
    res["role"] = ("hbm" if res["hbm_eligible"] else "cache_scale_diagnostic") if n_active == 2 else "per_physical_core_diagnostic"
    try:
        t0 = time.time()
        o = fn()
        T.sync()
        res["first_call_s"] = time.time() - t0
        o_cpu = (y if probe == "torch_copy" else o).cpu()
        ok = bool(torch.equal(o_cpu, src_cpu))
        res["check"] = {"ok": ok, "rule": "output equals the random int32 input (full comparison)"}
        meas = T.batches(fn, pid.replace("/", "_"))
        res.update({k: meas[k] for k in ("batches", "batch_medians_ms", "point_time_s", "batch_spread", "one_execution_per_call")})
        res["throughput"] = res["logical_bytes"] / meas["point_time_s"]
        res["valid"] = bool(ok and meas["one_execution_per_call"])
    except Exception as e:  # noqa: BLE001
        res.update(valid=False, throughput=None, error=f"{type(e).__name__}: {e}"[:600])
    del x
    return res


GEMM_DT = {"mma_bf16_f32acc": "bfloat16", "mma_fp16_f32acc": "float16", "gemm_fp32_ieee": "float32",
           "mma_tf32_f32acc": "float32", "mma_int8_i32acc": "int8", "mma_fp8_e4m3_f32acc": "float8_e4m3fn"}
# error window per mode; it also identifies the precision path: IEEE fp32 <= 1e-5, TF32 operand rounding
# (10-bit mantissa) lands between 5e-5 and 2e-3 (a TF32 point below 5e-5 did not take the TF32 path)
REL_TOL = {"mma_bf16_f32acc": (0.0, 1.6e-2), "mma_fp16_f32acc": (0.0, 2.5e-3), "gemm_fp32_ieee": (0.0, 1e-5),
           "mma_tf32_f32acc": (5e-5, 2e-3), "mma_fp8_e4m3_f32acc": (0.0, 1.6e-2), "mma_int8_i32acc": (0.0, 0.0)}


def probe_gemm(T: Timer, mode: str, n: int, proto: dict) -> dict:
    torch = T.torch
    dt = getattr(torch, GEMM_DT[mode])
    pid = f"{mode}/M={n}"
    g = torch.Generator().manual_seed(1000 + n)
    if mode == "mma_int8_i32acc":       # full int8 range: an inexact (fp32-accumulated) path cannot pass the exact check
        a_cpu = torch.randint(-128, 128, (n, n), generator=g, dtype=torch.int8)
        b_cpu = torch.randint(-128, 128, (n, n), generator=g, dtype=torch.int8)
    else:
        a_cpu = (torch.rand(n, n, generator=g) * 2 - 1).to(dt)
        b_cpu = (torch.rand(n, n, generator=g) * 2 - 1).to(dt)
    if mode == "mma_fp8_e4m3_f32acc":   # cuBLASLt convention of the B200 probe: B column-major, unit scales, bf16 out
        b_cpu = b_cpu.t().contiguous().t()
    a, b = a_cpu.to(T.dev), b_cpu.to(T.dev)
    T.sync()
    if mode == "mma_int8_i32acc":
        fn = lambda: torch._int_mm(a, b)
    elif mode == "mma_fp8_e4m3_f32acc":
        one = torch.tensor(1.0, device=T.dev)
        fn = lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    else:
        fn = lambda: torch.matmul(a, b)
    res = {"id": pid, "M": n, "N": n, "K": n, "dtype": GEMM_DT[mode], "flops": 2 * n ** 3,
           "api": {"mma_int8_i32acc": "torch._int_mm", "mma_fp8_e4m3_f32acc": "torch._scaled_mm(out_dtype=bf16)"}.get(mode, "torch.matmul")}
    try:
        t0 = time.time()
        c = fn()
        T.sync()
        res["first_call_s"] = time.time() - t0
        blk = proto["gemm"]["check_block"]
        if mode == "mma_int8_i32acc":
            ref = a_cpu[:blk].long() @ b_cpu[:, :blk].long()
            got_i = c[:blk, :blk].cpu().long()
            rel = float((got_i - ref).abs().max()) / float(ref.abs().max())      # exact path -> 0
            got = got_i.double()
        else:
            ref = a_cpu[:blk].double() @ b_cpu[:, :blk].double()
            got = c[:blk, :blk].cpu().double()
            rel = float((got - ref).abs().max() / ref.abs().max())
        lo, hi = REL_TOL[mode]
        res["check"] = {"ok": lo <= rel <= hi and bool(torch.isfinite(got).all()),
                        "max_abs_err_over_max_abs_ref": rel, "window": [lo, hi],
                        "rule": "C[:128,:128] vs float64 product of the same rounded operands"}
        meas = T.batches(fn, pid.replace("/", "_").replace("=", ""))
        res.update({k: meas[k] for k in ("batches", "batch_medians_ms", "point_time_s", "batch_spread", "one_execution_per_call")})
        if meas["point_time_s"] is None:
            res.update(valid=False, throughput=None, no_device_execution=True,
                       error="no device execution in the timed sessions: the call does not run on the NeuronCore "
                             "(host/CPU path); its result is correct but no device rate exists")
        else:
            res["throughput"] = res["flops"] / meas["point_time_s"]
            res["valid"] = bool(res["check"]["ok"] and meas["one_execution_per_call"])
    except Exception as e:  # noqa: BLE001
        res.update(valid=False, throughput=None, error=f"{type(e).__name__}: {e}"[:600])
    del a, b
    return res


VEC_DT = {"fp32_fma_vector": "float32", "fp16x2_fma_vector": "float16", "bf16x2_fma_vector": "bfloat16"}


def probe_fma(T: Timer, K: dict, mode: str, engines: list, proto: dict, n_active: int = 2, iters_mult: int = 1) -> dict:
    torch = T.torch
    v = proto["vector"]
    dt = getattr(torch, VEC_DT[mode])
    chains = tuple(ENGINE_CODES[e] for e in engines for _ in range(v["chains_per_engine"]))
    C, F = len(chains), v["free"]
    iters = v["iters"] * iters_mult
    pid = f"{mode}/{'+'.join(engines)}/iters={iters}" + ("" if n_active == 2 else f"/pcores={n_active}")
    g = torch.Generator().manual_seed(5)
    x0_cpu = torch.rand(2 * C * 128, F, generator=g).to(dt)
    x0 = x0_cpu.to(T.dev)
    T.sync()
    k = K["fma_chains"][2]
    fn = lambda: k(x0, iters, chains, v["a"], v["b"], n_active)
    res = {"id": pid, "engines": engines, "chains": list(chains), "iters": iters, "free": F, "partitions": 128,
           "physical_cores_active": n_active, "flops": 2 * 128 * F * C * iters * n_active,
           "role": "scoring" if n_active == 2 else "per_physical_core_diagnostic"}
    try:
        t0 = time.time()
        o = fn()
        T.sync()
        res["first_call_s"] = time.time() - t0
        rows = n_active * C * 128
        got = o.cpu()[:rows, :64].float()
        x = x0_cpu[:rows, :64].float()
        for _ in range(iters):           # reference: fp32 multiply-add, rounded to the tile dtype each step
            x = (x * v["a"] + v["b"]).to(dt).float()
        rel = float(((got - x).abs() / x.abs().clamp_min(1e-3)).max())
        tol = 1e-5 if dt == torch.float32 else 2e-2
        res["check"] = {"ok": rel <= tol, "max_rel_err": rel, "tolerance": tol,
                        "rule": f"{iters} dependent steps x <- a*x+b per element vs a CPU reference rounded per step"}
        meas = T.batches(fn, pid.replace("/", "_").replace("=", "").replace("+", "_"))
        res.update({k_: meas[k_] for k_ in ("batches", "batch_medians_ms", "point_time_s", "batch_spread", "one_execution_per_call")})
        res["throughput"] = res["flops"] / meas["point_time_s"]
        res["valid"] = bool(res["check"]["ok"] and meas["one_execution_per_call"])
    except Exception as e:  # noqa: BLE001
        res.update(valid=False, throughput=None, error=f"{type(e).__name__}: {e}"[:600])
    return res


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--modes", default="hbm_stream_bw,mma_bf16_f32acc,mma_fp16_f32acc,gemm_fp32_ieee,mma_int8_i32acc,"
                                         "mma_fp8_e4m3_f32acc,fp32_fma_vector,fp16x2_fma_vector,bf16x2_fma_vector")
    ap.add_argument("--out", default="")
    ap.add_argument("--extra-cc-flags", default="")
    ap.add_argument("--raw-only", action="store_true", help="write raw/<mode>.json only (used for sub-runs)")
    a = ap.parse_args()
    proto = SMOKE if a.smoke else PROTOCOL
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    cal_id = f"Trn2-{stamp}-{sha[:8]}" + ("-quick" if a.smoke else "")
    out = Path(a.out) if a.out else REPO / "artifacts/llm_v2/calibration/Trn2" / cal_id
    out.mkdir(parents=True, exist_ok=True)
    work = Path(os.environ.get("TRN2_CAL_WORK", "/tmp")) / f"trn2_cal_{cal_id}"
    env_vars = prepare_environment(work / "cache", a.extra_cc_flags)
    env = capture_environment(env_vars)
    T = Timer(work / "traces", proto)
    K = {k: {2: v[2]} for k, v in nki_kernels().items()}
    llc = 2 * SBUF_BYTES_PER_PCORE
    modes = [m for m in a.modes.split(",") if m]
    raw = {}
    for mode in modes:
        t0 = time.time()
        print(f"[trn2-cal] {mode}", flush=True)
        if mode == "hbm_stream_bw":
            pts = [probe_hbm(T, K, p, s, proto, llc) for p in proto["hbm"]["primary_probes"] + proto["hbm"]["diagnostic_probes"]
                   for s in proto["hbm"]["sizes_mib"]]
            pts.append(probe_hbm(T, K, "nki_copy_via_sbuf", 1024, proto, llc, n_active=1))
            raw[mode] = {"points": pts, "llc_equivalent_bytes": llc}
        elif mode in ("mma_int8_i32acc", "mma_fp8_e4m3_f32acc"):
            # device-path detection only: the smoke run showed these eager calls produce no device execution
            det = {**proto, "batches": 1, "warmup": 1, "repeat": 3}
            T.proto = det
            pt = probe_gemm(T, mode, 2048, det)
            T.proto = proto
            pt["role"] = "device_path_detection"
            raw[mode] = {"points": [pt], "detection_protocol": {"batches": 1, "warmup": 1, "repeat": 3, "size": 2048},
                         "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS")}
        elif mode in GEMM_DT:
            raw[mode] = {"points": [probe_gemm(T, mode, n, proto) for n in proto["gemm"]["sizes"]],
                         "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS")}
        elif mode in VEC_DT:
            pts = [probe_fma(T, K, mode, es, proto) for es in proto["vector"]["engine_sets"]]
            valid = [p for p in pts if p.get("valid")]
            if valid:                                    # scaling check on the best point; per-pcore diagnostic
                best = max(valid, key=lambda p: p["throughput"])
                dbl = probe_fma(T, K, mode, best["engines"], proto, iters_mult=2)
                dbl["role"] = "scaling_check"
                ratio = dbl["point_time_s"] / best["point_time_s"] if dbl.get("point_time_s") else None
                best["scaling"] = {"time_ratio_2x_iters": ratio, "ok": ratio is not None and abs(ratio - 2) / 2 <= 0.10}
                best["valid"] = bool(best["valid"] and best["scaling"]["ok"])
                pts.append(dbl)
                pts.append(probe_fma(T, K, mode, best["engines"], proto, n_active=1))
            raw[mode] = {"points": pts}
        raw[mode]["wall_s"] = time.time() - t0
        (out / "raw").mkdir(exist_ok=True)
        (out / "raw" / f"{mode}.json").write_text(json.dumps(raw[mode], indent=1, default=str) + "\n")
        for p in raw[mode]["points"]:
            print(f"   {p['id']:52s} valid={p.get('valid')} thr={p.get('throughput')} spread={p.get('batch_spread')} "
                  f"chk={(p.get('check') or {}).get('ok')} err={p.get('error', '')[:160]}", flush=True)
    if a.raw_only:
        (out / f"environment_subrun_{modes[0]}.json").write_text(json.dumps(env, indent=1) + "\n")
    if not a.raw_only:
        (out / "protocol.json").write_text(json.dumps(proto, indent=1) + "\n")
        (out / "environment.json").write_text(json.dumps(env, indent=1) + "\n")
    print(json.dumps({"out": str(out), "calibration_id": cal_id}))


# ----------------------------------------------------------------------------- finalize (profile / summary / sums)
SCORING_ROLES = {"scoring", "hbm", "cache_scale_diagnostic"}
UNSUPPORTED = {
    "mma_int8_i32acc": "NeuronCore-v3 TensorE has no int8 input type (installed nki 0.7 nc_matmul: float8_e4m3/float8_e5m2/"
                       "bfloat16/float16/tfloat32/float32); eager torch._int_mm returns a correct int32 result but produces no "
                       "device execution (host path), so no device rate exists",
    "mma_fp8_e4m3_f32acc": "NeuronCore-v3 TensorE takes the legacy float8_e4m3 format, not OCP float8_e4m3fn (installed nki 0.7 "
                           "nc_matmul docs: e4m3fn from NeuronCore-v4); eager torch._scaled_mm(e4m3fn) produces no device "
                           "execution (host path)",
    "mma_xf32_f32acc": "ROCm XF32 matrix path (AMD only)",
    "mma_fp8_e4m3fnuz_f32acc": "AMD float8_e4m3fnuz (AMD only)",
}
NOT_MEASURED = {
    "mma_fp8_e5m2_f32acc": "supported by TensorE (float8_e5m2, double_row) but declared by no Trn2 task; not measured",
    "int32_vector": "no probe registered (as on every device)",
}
LIMITS = {
    "matrix": ["eager torch.matmul on torch.device('neuron') (neuronx-cc compiled) on square shapes 4096..16384, one "
               "logical NeuronCore (2 physical cores); the sustained rate of this compiler path, not a proof of the "
               "hardware maximum; TensorE activity throttling, if any, is part of the sustained rate"],
    "vector": ["NKI nisa.tensor_scalar(multiply, add) chains on SBUF-resident tiles over VectorE / ScalarE / GpSimd "
               "(2 FLOP per element per instruction); Trn2 has no packed-pair FMA instruction, so the fp16x2/bf16x2 modes "
               "are FMA on fp16/bf16 tiles; exp/compare/convert/reduce instruction rates are not measured"],
    "bandwidth": ["NKI DMA copies (HBM->SBUF->HBM and HBM->HBM) and eager copy_ on one logical NeuronCore; read+write "
                  "bytes; the bandwidth one logical core can drive, not the device total"],
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(out: Path):
    from tilebench.llm.v2.calibration import schema, stats
    proto = json.loads((out / "protocol.json").read_text())
    env = json.loads((out / "environment.json").read_text())
    raws = {p.stem: json.loads(p.read_text()) for p in sorted((out / "raw").glob("*.json"))}
    cal_id = out.name
    modes = {}
    for name, r in raws.items():
        reg = schema.MODES[name]
        pts = r["points"]
        base = {"unit": reg["unit"], "source_kind": "measured", "role": reg["role"],
                "meaning": {k: v for k, v in reg.items() if k not in ("unit", "role")}}
        if name in UNSUPPORTED:
            ev = [{"id": p["id"], "check": p.get("check"), "no_device_execution": p.get("no_device_execution"),
                   "error": p.get("error")} for p in pts]
            modes[name] = {**base, "status": "unsupported", "value": None, "reason": UNSUPPORTED[name], "evidence": ev}
            continue
        cand = [p for p in pts if p.get("role", "scoring") in SCORING_ROLES]
        sel = stats.select_mode_value(cand, bandwidth=(name == "hbm_stream_bw"),
                                      plateau_tolerance=proto["hbm"]["plateau_tolerance"],
                                      saturation_tolerance=proto["saturation_tolerance"])
        flagged = [p["id"] for p in cand if p.get("batch_spread") is not None and p["batch_spread"] > proto["batch_spread_flag"]]
        keep = ("id", "valid", "throughput", "point_time_s", "batch_spread", "hbm_eligible", "primary", "role", "check",
                "physical_cores_active", "scaling", "error", "first_call_s")
        kind = reg["kind"]
        entry = {**base, "status": sel["status"], "value": sel["value"], "selected_point": sel.get("selected"),
                 "reason": sel.get("reason"), "points": [{k: p.get(k) for k in keep if k in p} for p in pts],
                 "points_batch_spread_flagged": flagged, "flags": sel.get("flags", []),
                 "limitations": LIMITS["bandwidth" if kind == "bandwidth" else kind],
                 "probe": {k: v for k, v in r.items() if k != "points"}}
        if "plateau" in sel:
            entry["plateau"] = sel["plateau"]
        if name == "gemm_fp32_ieee":
            entry["limitations"] = ["control only: IEEE fp32 eager matmul (error <= 1e-5 vs float64), never a scoring ceiling"]
        if name == "mma_tf32_f32acc":
            entry["precision_path"] = ("neuronx-cc --auto-cast matmult --auto-cast-type tf32 on eager fp32 torch.matmul; "
                                       "TF32 identified by the error window [5e-5, 2e-3] vs float64 (the default fp32 "
                                       "matmul is IEEE, error <= 1e-5)")
        per_core = [p for p in pts if p.get("role") == "per_physical_core_diagnostic"]
        if per_core:
            entry["per_physical_core_diagnostic"] = [{k: p.get(k) for k in ("id", "valid", "throughput", "check")} for p in per_core]
        modes[name] = entry
    for name, reg in schema.MODES.items():
        if name in modes:
            continue
        status, reason = ("unsupported", UNSUPPORTED[name]) if name in UNSUPPORTED else \
            ("not_measured", NOT_MEASURED.get(name, "not in this run's mode list"))
        modes[name] = {"unit": reg["unit"], "source_kind": "measured", "role": reg["role"], "status": status,
                       "value": None, "reason": reason}
    src = Path(__file__).resolve()
    cal_src = REPO / "tilebench/llm/v2/calibration"
    git = lambda *x: subprocess.run(["git", *x], cwd=REPO, capture_output=True, text=True).stdout
    dt_path = REPO / "tilebench/data/peak_performance/Trainium2.json"
    dt = json.loads(dt_path.read_text())
    profile = {
        "schema": schema.PROFILE_SCHEMA, "ceiling_basis": schema.CEILING_BASIS, "calibration_id": cal_id, "device": "Trn2",
        "scope": {"kind": "single_logical_neuroncore", "backend": "neuron",
                  "name": "AWS Trainium2 (NeuronCore-v3)", "instance_type": env.get("instance_type"),
                  "logical_neuroncore_config": env.get("logical_neuroncore_config"),
                  "physical_neuroncores_in_scope": 2, "logical_neuroncores_in_scope": 1,
                  "device_total": {"neuron_devices": env.get("neuron_devices"), "logical_neuroncores": env.get("logical_cores_per_device"),
                                   "physical_neuroncores": env.get("physical_cores_per_device"), "device_memory": env.get("device_memory")},
                  "visible_devices": {k: env["env"].get(k, "<unset>") for k in ("NEURON_RT_NUM_CORES", "NEURON_RT_VISIBLE_CORES",
                                                                                 "NEURON_LOGICAL_NC_CONFIG")},
                  "llc_bytes": 2 * SBUF_BYTES_PER_PCORE,
                  "llc_note": "SBUF capacity of the two physical cores used as the 'last-level' capacity for HBM eligibility; "
                              "Trn2 has no hardware cache between HBM and SBUF",
                  "note": "one logical NeuronCore (LNC=2: two physical NeuronCore-v3), NEURON_RT_NUM_CORES=1: the scope on "
                          "which NKI candidates and the formal TileBench NKI results are timed. Rates are per logical "
                          "NeuronCore. Per-physical-core values are diagnostics measured with one core active; the "
                          "whole device (4 logical / 8 physical cores) was not measured and is never obtained by scaling."},
        "protocol": {"id": proto["protocol"], "sha256": schema.sha256_json(proto), "quick": proto["protocol"].endswith("+quick")},
        "code": {"git_sha": git("rev-parse", "HEAD").strip(),
                 "dirty_paths": [l[3:] for l in git("status", "--porcelain").splitlines() if l.strip()],
                 "calibration_sources_sha256": {"scripts/llm_v2/calibrate_trn2.py": _sha(src),
                                                **{f"calibration/{f}": _sha(cal_src / f) for f in ("schema.py", "stats.py")},
                                                "tilebench/core/nki_timer.py": _sha(REPO / "tilebench/core/nki_timer.py")}},
        "environment": {"file": "environment.json", "sha256": _sha(out / "environment.json"),
                        "subruns": {p.name: _sha(p) for p in sorted(out.glob("environment_subrun_*.json"))}},
        "modes": modes,
        "datasheet_reference": {"source_kind": "datasheet", "path": str(dt_path.relative_to(REPO)), "sha256": _sha(dt_path),
                                "peak_tflops": dt.get("peak_tflops"), "note": "legacy per-logical-core compute entries "
                                "(not measured); reference only, never a scoring ceiling"},
        "legacy_reference": {"source_kind": "measured_legacy", "path": str(dt_path.relative_to(REPO)),
                             "values": {"hbm_GBs": dt.get("peak_bw_GBs")}, "method": dt.get("source"),
                             "interpretation": "torch-xla compiled mul2 reference on one logical core (older stack); "
                                               "comparison only"},
        "created_at": time.time(),
    }
    schema.seal(profile)
    errs = schema.validate_profile(profile)
    if errs:
        raise SystemExit(f"profile failed schema validation: {errs}")
    (out / "profile.json").write_text(json.dumps(profile, indent=1, sort_keys=True, allow_nan=False) + "\n")
    summary = {"calibration_id": cal_id, "device": "Trn2", "profile_sha256": profile["profile_sha256"],
               "scope": "one logical NeuronCore (LNC=2, two physical NeuronCore-v3), NEURON_RT_NUM_CORES=1",
               "modes": {n: {"status": m["status"], "unit": m["unit"], "value": m.get("value"),
                             "selected_point": m.get("selected_point"), "reason": m.get("reason"),
                             "per_physical_core_diagnostic": [(d["id"], d["throughput"]) for d in m.get("per_physical_core_diagnostic", [])]}
                         for n, m in profile["modes"].items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    lines = [f"{_sha(f)}  {f.relative_to(out)}" for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS")]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: (v["status"], v.get("value")) for k, v in summary["modes"].items()}, indent=0))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--finalize":
        finalize(Path(sys.argv[2]))
    else:
        main()
