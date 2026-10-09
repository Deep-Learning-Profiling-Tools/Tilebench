"""ROCm (MI300X, gfx942) additions to the calibration probes. Used only when the backend is ROCm; the CUDA
probes and their results are unchanged.

Two registered modes need a second measurement path on gfx942, found on the hardware (2026-10-09):

- fp16x2_fma_vector: the shared tl.fma probe compiles to mostly scalar v_fma_f16 on gfx942 (28 v_fma_f16 +
  2 v_pk_fma_f16 per iteration), i.e. half of the packed fp16 rate; its instruction check rejects those points.
  This probe keeps each chain as two fp16 values packed in one 32-bit register and issues v_pk_fma_f16 through
  tl.inline_asm_elementwise, so the loop body is only v_pk_fma_f16 (no per-iteration pack/unpack).
- mma_int8_i32acc: torch._int_mm (hipBLASLt Cijk_..._I8II_..., and rocBLAS alike) sustains about half of a plain
  Triton int8 tl.dot GEMM (v_mfma_i32_*_i8) at 8192^3. The library point is kept; a Triton GEMM probe adds
  points, and the mode value is the maximum over all valid points (same rule as the two primary HBM probes).

Every point uses the shared timing, device-bound and correctness rules of gpu_probes.py.
"""
from __future__ import annotations

import re

import torch
import triton
import triton.language as tl

from tilebench.llm.v2.calibration import gpu_probes as gp
from tilebench.llm.v2.calibration import stats

# --------------------------------------------------------------------------- packed fp16 FMA


@triton.jit
def _pk_fma_f16(x, a, b):
    return tl.inline_asm_elementwise("v_pk_fma_f16 $0, $1, $2, $3", "=v,v,v,v", [x, a, b], dtype=tl.int32,
                                     is_pure=True, pack=1)


@triton.jit
def _h2(v):  # fp32 value -> int32 holding that fp16 value in both halves
    h = v.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    return h | (h << 16)


@triton.jit
def _sum_halves(x):
    lo = (x & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    hi = ((x >> 16) & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    return lo + hi


@triton.jit
def _pk_fma_kernel(out_ptr, a, b, iters, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    base = (offs % 97).to(tl.float32) * 0.01
    x0 = _h2(base)
    x1 = _h2(base + 0.1)
    x2 = _h2(base + 0.2)
    x3 = _h2(base + 0.3)
    x4 = _h2(base + 0.4)
    x5 = _h2(base + 0.5)
    x6 = _h2(base + 0.6)
    x7 = _h2(base + 0.7)
    av = _h2(tl.full((BLOCK,), a, tl.float32))
    bv = _h2(tl.full((BLOCK,), b, tl.float32))
    for _ in range(iters):            # runtime trip count: the loop cannot be folded
        x0 = _pk_fma_f16(x0, av, bv)
        x1 = _pk_fma_f16(x1, av, bv)
        x2 = _pk_fma_f16(x2, av, bv)
        x3 = _pk_fma_f16(x3, av, bv)
        x4 = _pk_fma_f16(x4, av, bv)
        x5 = _pk_fma_f16(x5, av, bv)
        x6 = _pk_fma_f16(x6, av, bv)
        x7 = _pk_fma_f16(x7, av, bv)
    s = (_sum_halves(x0) + _sum_halves(x1) + _sum_halves(x2) + _sum_halves(x3)
         + _sum_halves(x4) + _sum_halves(x5) + _sum_halves(x6) + _sum_halves(x7))
    tl.store(out_ptr + offs, s)


def probe_fp16x2_packed(proto: dict, dev, sm_count: int) -> list[dict]:
    """Points of fp16x2_fma_vector measured with explicit v_pk_fma_f16 (2 fp16 lanes per instruction)."""
    vp = proto["vector"]
    block, warps, iters, chains = vp["block"], vp["num_warps"], vp["iters"], gp.FMA_CHAINS
    a, b = 0.5, 0.5
    expected = float(2 * chains)      # both halves of every chain converge to 1.0
    points = []
    for pps in vp["programs_per_sm"]:
        programs = sm_count * pps
        n = programs * block
        out = torch.empty(n, device=dev, dtype=torch.float32)
        pid = f"fp16x2_fma_vector/rocm_pk_fma_f16_asm/programs_per_sm{pps}"

        def launch(it, out=out, programs=programs):
            return _pk_fma_kernel[(programs,)](out, a, b, it, BLOCK=block, num_warps=warps)

        k = launch(iters)
        gp._sync()
        ok = bool(torch.allclose(out, torch.full_like(out, expected), rtol=1e-3, atol=1e-3))
        _, ops = gp._opcodes(k.asm)
        pk = ops.get("v_pk_fma_f16", 0)
        per_thread = chains * block // (64 * warps)            # packed instructions per thread per iteration
        loop_ops = _loop_opcodes(k.asm.get("amdgcn", ""))
        instr_ok = pk >= per_thread and set(loop_ops) <= {"v_pk_fma_f16", "s_add_i32", "s_cmp_lg_u32", "s_cbranch_scc1",
                                                           "s_cbranch_scc0", "s_nop"}
        fn = (lambda: launch(iters))
        meas = gp.measure(fn, proto)
        t1 = stats.point_time(meas["batches_ms"]) / 1e3
        t2 = stats.point_time(gp.measure(lambda: launch(2 * iters), {**proto, "batches": 1, "repeat": 10, "warmup": 2},
                                         inner=1)["batches_ms"]) / 1e3
        ratio = t2 / t1
        scaling_ok = abs(ratio - 2.0) <= 2.0 * vp["scaling_tolerance"]
        tr = gp.trace_check(fn, max(5, min(meas["inner"], 20)))
        dev_vs_event = (tr["device_us_per_launch"] / 1e6) / t1 if tr.get("available") else None
        work = 2.0 * 2 * n * chains * iters                      # 2 FLOP x 2 fp16 lanes per v_pk_fma_f16
        valid = bool(ok and instr_ok and scaling_ok and gp._device_bound(tr, dev_vs_event, proto))
        points.append({"id": pid, "probe": "rocm_pk_fma_f16_inline_asm", "programs": programs, "programs_per_sm": pps,
                       "block": block, "num_warps": warps, "chains": chains, "lanes_per_element": 2, "iters": iters,
                       "work": work, "work_unit": "FLOP", "inner": meas["inner"], "batches_ms": meas["batches_ms"],
                       "batch_medians_ms": stats.batch_medians(meas["batches_ms"]), "point_time_s": t1,
                       "throughput": work / t1, "batch_spread": stats.batch_spread(meas["batches_ms"]),
                       "check": {"ok": ok, "rule": f"every output equals {expected:g} (both fp16 halves of 8 chains reach 1.0)"},
                       "instructions": {"listing": "amdgcn", "v_pk_fma_f16": pk, "packed_per_thread_per_iteration": per_thread,
                                        "loop_body_opcodes": loop_ops, "ok": instr_ok},
                       "scaling": {"time_2x_over_1x": ratio, "ok": scaling_ok},
                       "trace": tr, "device_vs_event_time": dev_vs_event, "valid": valid})
        del out
    return points


def _loop_opcodes(text: str) -> list[str]:
    """Distinct opcodes of the innermost loop that contains v_pk_fma_f16 (between its label and back-branch)."""
    lines = text.split(".Lfunc_end")[0].splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(\.LBB\d+_\d+):", line)
        if not m:
            continue
        body = []
        for s in lines[i + 1:]:
            s = s.split(";")[0].strip()
            if not s or s.startswith("."):
                if s.startswith(".LBB"):
                    break
                continue
            op = s.split()[0]
            body.append(op)
            if op.startswith("s_cbranch") and m.group(1) in s:
                if "v_pk_fma_f16" in body:
                    return sorted(set(body))
                break
    return []


# --------------------------------------------------------------------------- Triton int8 GEMM


@triton.jit
def _gemm_i8_kernel(a, b, c, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    gid = pid // (G * npn)
    fm = gid * G
    gs = min(npm - fm, G)
    pm = fm + (pid % (G * npn)) % gs
    pn = (pid % (G * npn)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, K, BK):              # shapes are multiples of the tile: no masks
        x = tl.load(a + rm[:, None] * K + (k + rk)[None, :])
        y = tl.load(b + (k + rk)[:, None] * N + rn[None, :])
        acc += tl.dot(x, y, out_dtype=tl.int32)
    tl.store(c + rm[:, None] * N + rn[None, :], acc)


# Task-independent configurations chosen from a small sweep (tiles 128..256, BK 32..128, stages 1/2,
# matrix_instr_nonkdim 16/32, kpack 1/2, waves_per_eu 0/2) at 8192^3 before this run; recorded per point.
INT8_GEMM_CONFIGS = [
    {"BM": 256, "BN": 256, "BK": 64, "num_warps": 8, "num_stages": 2, "matrix_instr_nonkdim": 16, "kpack": 1, "waves_per_eu": 2},
    {"BM": 256, "BN": 256, "BK": 64, "num_warps": 8, "num_stages": 2, "matrix_instr_nonkdim": 16, "kpack": 1, "waves_per_eu": 0},
    {"BM": 256, "BN": 256, "BK": 64, "num_warps": 8, "num_stages": 2, "matrix_instr_nonkdim": 32, "kpack": 1, "waves_per_eu": 0},
]


def probe_int8_triton(proto: dict, dev) -> list[dict]:
    points = []
    for ci, cfg in enumerate(INT8_GEMM_CONFIGS):
        for m in proto["gemm"]["sizes"]:
            pid = f"mma_int8_i32acc/triton_tl_dot_cfg{ci}/M{m}"
            kw = dict(cfg)
            BM, BN, BK = kw.pop("BM"), kw.pop("BN"), kw.pop("BK")
            if m % BM or m % BN or m % BK:
                continue
            try:
                g = torch.Generator(device=dev).manual_seed(1234 + m)
                a = torch.randint(-128, 128, (m, m), dtype=torch.int8, device=dev, generator=g)
                b = torch.randint(-128, 128, (m, m), dtype=torch.int8, device=dev, generator=g)
                c = torch.empty((m, m), dtype=torch.int32, device=dev)
                grid = (triton.cdiv(m, BM) * triton.cdiv(m, BN),)
                fn = (lambda: _gemm_i8_kernel[grid](a, b, c, m, m, m, BM=BM, BN=BN, BK=BK, G=8, **kw))
                k = fn()
                gp._sync()
                check = gp._gemm_check("mma_int8_i32acc", a, b, c, proto["gemm"]["check_block"])
                _, ops = gp._opcodes(k.asm)
                mfma = sorted(o for o in ops if o.startswith("v_mfma"))
                check["mfma_opcodes"] = mfma
                check["ok"] = bool(check["ok"] and mfma and all(o.endswith("_i8") for o in mfma))
                meas = gp.measure(fn, proto)
                tr = gp.trace_check(fn, max(5, min(meas["inner"], 40)))
            except (RuntimeError, TypeError, ValueError) as e:
                points.append({"id": pid, "M": m, "probe": "triton_int8_gemm", "config": cfg, "valid": False,
                               "error": f"{type(e).__name__}: {e}"})
                torch.cuda.empty_cache()
                continue
            t_s = stats.point_time(meas["batches_ms"]) / 1e3
            work = 2.0 * m * m * m
            dev_vs_event = (tr["device_us_per_launch"] / 1e6) / t_s if tr.get("available") else None
            valid = bool(check["ok"] and gp._device_bound(tr, dev_vs_event, proto))
            points.append({"id": pid, "probe": "triton_int8_gemm", "config": cfg, "M": m, "N": m, "K": m, "work": work,
                           "work_unit": "OP", "inner": meas["inner"], "batches_ms": meas["batches_ms"],
                           "batch_medians_ms": stats.batch_medians(meas["batches_ms"]), "point_time_s": t_s,
                           "throughput": work / t_s, "batch_spread": stats.batch_spread(meas["batches_ms"]),
                           "check": check, "trace": tr, "device_vs_event_time": dev_vs_event, "valid": valid})
            del a, b, c
            torch.cuda.empty_cache()
    return points
