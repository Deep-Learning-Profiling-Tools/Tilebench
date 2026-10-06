"""Task-independent calibration probes for CUDA and ROCm devices.

Timing path: CUDA/HIP events around `inner` back-to-back launches of the
probe (inner calibrated so a sample lasts >= target_sample_ms), samples are
event interval / inner. Allocation, input generation, compilation and
warm-up are outside the timed region. For every point a separate,
untimed profiler pass over back-to-back launches checks that the device
was busy for the whole window (`busy_fraction`) and that the per-launch
device time agrees with the event-based time, and records the kernel names
(precision path evidence). Profiled times are never used as rates.

Correctness / non-eliminable work:
- GEMMs: a 128x128 output block is compared with a float64 (exact int64
  for int8) reference; for fp32 inputs the error magnitude also tells the
  IEEE path from the TF32 path.
- copy/read/fill probes: full equality / per-block checksums / pattern check.
- FMA probes: closed-form expected output, a loop-length scaling check
  (2x iterations -> ~2x time) and the expected FMA instructions in the
  compiled SASS/AMDGCN.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import tempfile

import torch
import triton
import triton.language as tl

from tilebench.llm.v2.calibration import stats

# --------------------------------------------------------------------------- timing


def _sync():
    torch.cuda.synchronize()


def _event_ms(fn, n: int) -> float:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def measure(fn, proto: dict, inner: int | None = None) -> dict:
    for _ in range(proto["warmup"]):
        fn()
    _sync()
    if inner is None:
        single = max(_event_ms(fn, 1), 1e-4)
        inner = max(1, int(math.ceil(proto["target_sample_ms"] / single)))
    batches = []
    for b in range(proto["batches"]):
        if b:
            for _ in range(proto["warmup"]):
                fn()
            _sync()
        batches.append([_event_ms(fn, inner) / inner for _ in range(proto["repeat"])])
    return {"inner": inner, "batches_ms": batches}


def trace_check(fn, n: int) -> dict:
    """Untimed profiler pass: device-busy share of a window of n back-to-back
    launches, per-launch device time and the device activity names."""
    from torch.profiler import ProfilerActivity, profile
    _sync()
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                fn()
            _sync()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        prof.export_chrome_trace(path)
        with open(path) as fh:
            trace = json.load(fh)
        os.unlink(path)
    except Exception as e:  # noqa: BLE001
        return {"available": False, "error": f"{type(e).__name__}: {e}"}
    allev = [x for x in trace.get("traceEvents", []) if x.get("ph") == "X"
             and x.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    names: dict[str, int] = {}
    for x in allev:
        names[x["name"]] = names.get(x["name"], 0) + 1
    # The probe's own activities repeat once per launch; activities recorded
    # from before the profiled launches (input generation, checks) do not.
    probe_names = {k for k, c in names.items() if c >= n}
    ev = sorted((x for x in allev if x["name"] in probe_names), key=lambda x: x["ts"])
    if not ev:
        return {"available": False, "error": "no device activity repeating once per launch", "names": names}
    start = ev[0]["ts"]
    end = max(x["ts"] + x["dur"] for x in ev)
    inside = [x for x in allev if x["ts"] >= start and x["ts"] + x["dur"] <= end]
    busy = sum(x["dur"] for x in inside)
    return {"available": True, "launches": n, "activities": len(inside), "window_us": end - start,
            "device_busy_us": busy, "busy_fraction": busy / (end - start) if end > start else None,
            "device_us_per_launch": busy / n, "names": names, "probe_names": sorted(probe_names)}


def _device_bound(tr: dict, ratio: float | None, proto: dict) -> bool:
    """The point measures device work, not dispatch: either the profiled
    back-to-back window shows no idle device time (busy_fraction >=
    device_share_min), or the device time per launch from that window
    explains the event-timed time per launch (ratio >= device_share_min).
    A dispatch-bound point fails both. Both tests are needed because the
    profiler slows host enqueue (short probes show profiler-induced gaps) and
    the device clock drifts under sustained load (long probes show a ratio
    away from 1 between the timed and the profiled pass)."""
    if not tr.get("available"):
        return False
    busy = tr.get("busy_fraction")
    return bool((busy is not None and busy >= proto["device_share_min"])
                or (ratio is not None and ratio >= proto["device_share_min"]))


# --------------------------------------------------------------------------- precision control


@contextlib.contextmanager
def precision(fp32_path: str | None = None, reduced_precision_reduction: bool | None = None):
    """Set and restore the BLAS precision controls explicitly. fp32_path:
    'ieee' | 'tf32' | None. Uses torch's fp32_precision API when present,
    the legacy allow_tf32 flag otherwise."""
    m = torch.backends.cuda.matmul
    saved = {"allow_tf32": m.allow_tf32,
             "fp16_rpr": m.allow_fp16_reduced_precision_reduction,
             "bf16_rpr": m.allow_bf16_reduced_precision_reduction}
    new_api = hasattr(m, "fp32_precision")
    if new_api:
        saved["fp32_precision"] = m.fp32_precision
    try:
        if fp32_path is not None:
            if new_api:
                m.fp32_precision = fp32_path
            else:
                m.allow_tf32 = fp32_path == "tf32"
        if reduced_precision_reduction is not None:
            m.allow_fp16_reduced_precision_reduction = reduced_precision_reduction
            m.allow_bf16_reduced_precision_reduction = reduced_precision_reduction
        yield
    finally:
        if new_api:
            m.fp32_precision = saved["fp32_precision"]
        else:
            m.allow_tf32 = saved["allow_tf32"]
        m.allow_fp16_reduced_precision_reduction = saved["fp16_rpr"]
        m.allow_bf16_reduced_precision_reduction = saved["bf16_rpr"]


def effective_flags() -> dict:
    """Read back the controls in force. With torch's fp32_precision API only
    that API is read: reading the legacy allow_tf32 getter after setting the
    new API makes torch refuse later matmuls (mixed-API check)."""
    m = torch.backends.cuda.matmul
    out = {"allow_fp16_reduced_precision_reduction": m.allow_fp16_reduced_precision_reduction,
           "allow_bf16_reduced_precision_reduction": m.allow_bf16_reduced_precision_reduction}
    if hasattr(m, "fp32_precision"):
        out["api"] = "torch.backends.cuda.matmul.fp32_precision"
        out["fp32_precision"] = m.fp32_precision
        out["torch.backends.fp32_precision"] = getattr(torch.backends, "fp32_precision", "unavailable")
    else:
        out["api"] = "legacy allow_tf32"
        out["allow_tf32"] = m.allow_tf32
        out["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    return out


# --------------------------------------------------------------------------- GEMM probes

GEMM_SPECS = {
    # mode: (api, input dtype, fp32 path, reduced-precision reduction, rel-RMS window [lo, hi))
    "mma_fp16_f32acc": ("torch.matmul", torch.float16, None, False, (0.0, 2e-3)),
    "mma_bf16_f32acc": ("torch.matmul", torch.bfloat16, None, False, (0.0, 2e-2)),
    "mma_tf32_f32acc": ("torch.matmul", torch.float32, "tf32", None, (5e-5, 5e-3)),
    "mma_xf32_f32acc": ("torch.matmul", torch.float32, "tf32", None, (5e-5, 5e-3)),
    "gemm_fp32_ieee": ("torch.matmul", torch.float32, "ieee", None, (0.0, 2e-5)),
    "mma_fp8_e4m3_f32acc": ("torch._scaled_mm", torch.float8_e4m3fn, None, None, (0.0, 2e-2)),
    "mma_fp8_e5m2_f32acc": ("torch._scaled_mm", torch.float8_e5m2, None, None, (0.0, 2e-2)),
    "mma_fp8_e4m3fnuz_f32acc": ("torch._scaled_mm", getattr(torch, "float8_e4m3fnuz", None), None, None, (0.0, 2e-2)),
    "mma_int8_i32acc": ("torch._int_mm", torch.int8, None, None, None),
}


def _gemm_inputs(mode: str, m: int, dev):
    api, dt, _, _, _ = GEMM_SPECS[mode]
    g = torch.Generator(device=dev).manual_seed(1234 + m)
    if api == "torch._int_mm":
        a = torch.randint(-128, 128, (m, m), dtype=torch.int8, device=dev, generator=g)
        b = torch.randint(-128, 128, (m, m), dtype=torch.int8, device=dev, generator=g)
        return a, b, (lambda: torch._int_mm(a, b))
    if api == "torch._scaled_mm":
        a = torch.randn(m, m, device=dev, generator=g).to(dt)
        b = torch.randn(m, m, device=dev, generator=g).to(dt).t()          # column-major (K, N)
        one = torch.ones((), device=dev, dtype=torch.float32)
        return a, b, (lambda: torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16,
                                               use_fast_accum=False))
    a = torch.randn(m, m, device=dev, generator=g, dtype=torch.float32).to(dt)
    b = torch.randn(m, m, device=dev, generator=g, dtype=torch.float32).to(dt)
    return a, b, (lambda: torch.matmul(a, b))


def _gemm_check(mode: str, a, b, c, blk: int) -> dict:
    api, _, _, _, window = GEMM_SPECS[mode]
    if api == "torch._int_mm":
        ref = a[:blk].cpu().long() @ b[:, :blk].cpu().long()
        ok = bool(torch.equal(c[:blk, :blk].cpu().long(), ref))
        return {"ok": ok, "exact": True, "rule": "int64 CPU reference, exact equality"}
    ref = a[:blk].double() @ b[:, :blk].double()
    got = c[:blk, :blk].double()
    rel = float(torch.sqrt(((got - ref) ** 2).mean()) / torch.sqrt((ref ** 2).mean()))
    lo, hi = window
    return {"ok": lo <= rel < hi and math.isfinite(rel), "rel_rms_error": rel, "window": [lo, hi],
            "rule": "rel RMS error of a block vs float64; the window identifies the precision path"}


def probe_gemm(mode: str, proto: dict, dev) -> dict:
    api, dt, fp32_path, rpr, _ = GEMM_SPECS[mode]
    if dt is None:
        return {"status": "unsupported", "reason": f"torch build has no dtype for {mode}"}
    points = []
    with precision(fp32_path, rpr):
        flags = effective_flags()
        for m in proto["gemm"]["sizes"]:
            pid = f"{mode}/M{m}"
            try:
                a, b, fn = _gemm_inputs(mode, m, dev)
                c = fn()
                _sync()
                check = _gemm_check(mode, a, b, c, proto["gemm"]["check_block"])
                del c
                meas = measure(fn, proto)
                tr = trace_check(fn, max(5, min(meas["inner"], 40)))
            except (RuntimeError, TypeError, ValueError) as e:
                points.append({"id": pid, "M": m, "valid": False, "error": f"{type(e).__name__}: {e}"})
                torch.cuda.empty_cache()
                continue
            t_s = stats.point_time(meas["batches_ms"]) / 1e3
            work = 2.0 * m * m * m
            dev_vs_event = (tr["device_us_per_launch"] / 1e6) / t_s if tr.get("available") else None
            valid = bool(check["ok"] and _device_bound(tr, dev_vs_event, proto))
            points.append({"id": pid, "M": m, "N": m, "K": m, "work": work,
                           "work_unit": "OP" if api == "torch._int_mm" else "FLOP",
                           "inner": meas["inner"], "batches_ms": meas["batches_ms"],
                           "batch_medians_ms": stats.batch_medians(meas["batches_ms"]),
                           "point_time_s": t_s, "throughput": work / t_s,
                           "batch_spread": stats.batch_spread(meas["batches_ms"]),
                           "check": check, "trace": tr, "device_vs_event_time": dev_vs_event, "valid": valid})
            del a, b
            torch.cuda.empty_cache()
    return {"api": api, "input_dtype": str(dt), "precision_flags_effective": flags, "points": points}


# --------------------------------------------------------------------------- streaming probes


@triton.jit
def _sm_copy_kernel(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    msk = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=msk), mask=msk)


@triton.jit
def _sm_read_kernel(src, partial, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(src + offs, mask=offs < n, other=0)
    tl.store(partial + pid, tl.sum(x, axis=0))


STREAM_BLOCK = 4096


def probe_stream(probe: str, size_mib: int, proto: dict, dev, llc_bytes: int | None) -> dict:
    nbytes = size_mib * 1024 * 1024
    g = torch.Generator(device=dev).manual_seed(77 + size_mib)
    pid = f"{probe}/{size_mib}MiB"
    src = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=dev, generator=g)
    if probe == "d2d_copy":
        dst = torch.empty_like(src)
        fn = (lambda: dst.copy_(src))
        work, ws = 2 * nbytes, 2 * nbytes
        fn(); _sync()
        ok = bool(torch.equal(dst, src)); rule = "dst equals src (full)"
    elif probe == "sm_copy":
        s32, d32 = src.view(torch.int32), torch.empty(nbytes // 4, dtype=torch.int32, device=dev)
        n = s32.numel(); grid = (triton.cdiv(n, STREAM_BLOCK),)
        fn = (lambda: _sm_copy_kernel[grid](s32, d32, n, BLOCK=STREAM_BLOCK, num_warps=8))
        work, ws = 2 * nbytes, 2 * nbytes
        fn(); _sync()
        ok = bool(torch.equal(d32, s32)); rule = "dst equals src (full)"
    elif probe == "sm_read":
        s32 = src.view(torch.int32); n = s32.numel(); grid = (triton.cdiv(n, STREAM_BLOCK),)
        part = torch.empty(grid[0], dtype=torch.int32, device=dev)
        fn = (lambda: _sm_read_kernel[grid](s32, part, n, BLOCK=STREAM_BLOCK, num_warps=8))
        work, ws = nbytes + part.numel() * 4, nbytes
        fn(); _sync()
        ref = s32.view(-1, STREAM_BLOCK).sum(dim=1, dtype=torch.int32)
        ok = bool(torch.equal(part, ref)); rule = "per-block int32 sums equal the torch reference"
    elif probe == "fill_write":
        dst = torch.empty_like(src)
        fn = (lambda: dst.fill_(0x5A))
        work, ws = nbytes, nbytes
        fn(); _sync()
        ok = bool((dst == 0x5A).all()); rule = "every byte equals the non-zero pattern 0x5A"
    else:
        raise ValueError(probe)
    meas = measure(fn, proto)
    tr = trace_check(fn, max(5, min(meas["inner"], 40)))
    t_s = stats.point_time(meas["batches_ms"]) / 1e3
    dev_vs_event = (tr["device_us_per_launch"] / 1e6) / t_s if tr.get("available") else None
    eligible = stats.hbm_eligible(ws, llc_bytes, proto["hbm"]["min_working_set_over_llc"])
    valid = bool(ok and _device_bound(tr, dev_vs_event, proto))
    out = {"id": pid, "probe": probe, "size_mib": size_mib, "logical_bytes": work, "working_set_bytes": ws,
           "primary": probe in proto["hbm"]["primary_probes"], "hbm_eligible": eligible,
           "role": "hbm" if eligible else "cache_scale_diagnostic",
           "inner": meas["inner"], "batches_ms": meas["batches_ms"],
           "batch_medians_ms": stats.batch_medians(meas["batches_ms"]), "point_time_s": t_s,
           "throughput": work / t_s, "batch_spread": stats.batch_spread(meas["batches_ms"]),
           "check": {"ok": ok, "rule": rule, "source": "random bytes (not compressible zeros)"},
           "trace": tr, "device_vs_event_time": dev_vs_event, "valid": valid}
    del src
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- FMA micro-benchmarks


@triton.jit
def _fma_kernel(out_ptr, a, b, iters, BLOCK: tl.constexpr, DT: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    base = (offs % 97).to(tl.float32) * 0.01
    x0 = base.to(DT)
    x1 = (base + 0.1).to(DT)
    x2 = (base + 0.2).to(DT)
    x3 = (base + 0.3).to(DT)
    x4 = (base + 0.4).to(DT)
    x5 = (base + 0.5).to(DT)
    x6 = (base + 0.6).to(DT)
    x7 = (base + 0.7).to(DT)
    av = tl.full((BLOCK,), a, DT)
    bv = tl.full((BLOCK,), b, DT)
    for _ in range(iters):            # runtime trip count: the loop cannot be folded
        x0 = tl.fma(x0, av, bv)
        x1 = tl.fma(x1, av, bv)
        x2 = tl.fma(x2, av, bv)
        x3 = tl.fma(x3, av, bv)
        x4 = tl.fma(x4, av, bv)
        x5 = tl.fma(x5, av, bv)
        x6 = tl.fma(x6, av, bv)
        x7 = tl.fma(x7, av, bv)
    s = (x0.to(tl.float32) + x1.to(tl.float32) + x2.to(tl.float32) + x3.to(tl.float32)
         + x4.to(tl.float32) + x5.to(tl.float32) + x6.to(tl.float32) + x7.to(tl.float32))
    tl.store(out_ptr + offs, s)


FMA_CHAINS = 8
FMA_SPECS = {  # mode: (triton dtype, CUDA SASS opcode regex, ROCm AMDGCN regex, lanes per instruction)
    "fp32_fma_vector": (tl.float32, r"^FFMA2?(\.|$)", r"^v_(pk_)?fma(c|k|mk)?_f32", None),
    "fp16x2_fma_vector": (tl.float16, r"^HFMA2(\.|$)(?!.*BF16)", r"^v_pk_fma_f16", 2),
    "bf16x2_fma_vector": (tl.bfloat16, r"^HFMA2\.BF16", r"^v_pk_fma_bf16", 2),
}


def _opcodes(asm: dict) -> tuple[str, dict]:
    if "sass" in asm or "cubin" in asm:
        try:
            text = asm["sass"]
        except Exception as e:  # noqa: BLE001
            return "sass", {"error": str(e)}
        ops: dict[str, int] = {}
        for line in text.splitlines():
            s = line.strip()
            if "\t" in s:                                   # "<control codes>\t<OPCODE> operands;"
                s = s.split("\t", 1)[1].strip()
            s = re.sub(r"^@!?U?P[T\d]+\s+", "", s)          # predicate guard
            m = re.match(r"([A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*)\b", s)
            if not m:
                continue
            op = m.group(1)
            if op.startswith("HFMA2") and "-RZ, RZ" in s:  # constant materialisation, not arithmetic
                op = op + "<const-mov>"
            ops[op] = ops.get(op, 0) + 1
        return "sass", ops
    text = asm.get("amdgcn", "")
    ops = {}
    for line in text.splitlines():
        m = re.match(r"^\s+(v_[a-z0-9_]+|s_[a-z0-9_]+)\b", line)
        if m:
            ops[m.group(1)] = ops.get(m.group(1), 0) + 1
    return "amdgcn", ops


def probe_fma(mode: str, proto: dict, dev, sm_count: int, backend: str) -> dict:
    dt, sass_re, gcn_re, _ = FMA_SPECS[mode]
    vp = proto["vector"]
    block, warps, iters = vp["block"], vp["num_warps"], vp["iters"]
    a, b = 0.5, 0.5          # exact in fp32/fp16/bf16; x -> 1 is the fixed point, every FMA still executes
    points = []
    for pps in vp["programs_per_sm"]:
        programs = sm_count * pps
        n = programs * block
        out = torch.empty(n, device=dev, dtype=torch.float32)
        pid = f"{mode}/programs_per_sm{pps}"

        def launch(it, out=out, programs=programs):
            return _fma_kernel[(programs,)](out, a, b, it, BLOCK=block, DT=dt, num_warps=warps)

        k = launch(iters); _sync()
        ok = bool(torch.allclose(out, torch.full_like(out, float(FMA_CHAINS)), rtol=1e-3, atol=1e-3))
        kind, ops = _opcodes(k.asm)
        rx = re.compile(sass_re if kind == "sass" else gcn_re)
        fma_instr = sum(c for op, c in ops.items() if rx.search(op) and "<const-mov>" not in op)
        per_thread_fmas = FMA_CHAINS * block // (32 * warps if kind == "sass" else 64 * warps)
        instr_ok = fma_instr >= per_thread_fmas // 2            # packed forms carry 2 lanes per instruction
        fn = (lambda: launch(iters))
        meas = measure(fn, proto)
        t1 = stats.point_time(meas["batches_ms"]) / 1e3
        t2 = stats.point_time(measure(lambda: launch(2 * iters), {**proto, "batches": 1, "repeat": 10,
                                                                   "warmup": 2}, inner=1)["batches_ms"]) / 1e3
        ratio = t2 / t1
        scaling_ok = abs(ratio - 2.0) <= 2.0 * vp["scaling_tolerance"]
        tr = trace_check(fn, max(5, min(meas["inner"], 20)))
        dev_vs_event = (tr["device_us_per_launch"] / 1e6) / t1 if tr.get("available") else None
        work = 2.0 * n * FMA_CHAINS * iters
        valid = bool(ok and instr_ok and scaling_ok and _device_bound(tr, dev_vs_event, proto))
        points.append({"id": pid, "programs": programs, "programs_per_sm": pps, "block": block, "num_warps": warps,
                       "chains": FMA_CHAINS, "iters": iters, "work": work, "work_unit": "FLOP",
                       "inner": meas["inner"], "batches_ms": meas["batches_ms"],
                       "batch_medians_ms": stats.batch_medians(meas["batches_ms"]), "point_time_s": t1,
                       "throughput": work / t1, "batch_spread": stats.batch_spread(meas["batches_ms"]),
                       "check": {"ok": ok, "rule": f"every output equals {FMA_CHAINS} (fixed point 1.0 per chain)"},
                       "instructions": {"listing": kind, "fma_opcode_regex": rx.pattern, "fma_instructions": fma_instr,
                                        "fmas_per_thread_per_iteration": per_thread_fmas, "ok": instr_ok,
                                        "top_opcodes": dict(sorted(ops.items(), key=lambda x: -x[1])[:15])},
                       "scaling": {"time_2x_over_1x": ratio, "ok": scaling_ok},
                       "trace": tr, "device_vs_event_time": dev_vs_event, "valid": valid})
        del out
    return {"api": "triton micro-benchmark (tl.fma, 8 independent chains, runtime trip count)",
            "dtype": str(dt), "points": points, "a": a, "b": b}
