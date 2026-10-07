"""Smoke tests for SKILL.md (TileLang 0.1.11 reference).

The code snippets are taken VERBATIM from SKILL.md (located by section heading), executed
with tilelang 0.1.11 on the local GPU, and compared with PyTorch via
torch.testing.assert_close. A few additional checks exercise behavioural claims made in
the text (streams, host-side checks, out-of-bounds zero fill, integer division, kernel-scope
statements, GEMM instruction selection on sm_100).

Run:  env -u LD_LIBRARY_PATH <tilebench_env python> smoke_tests.py
Writes smoke_results.json next to this file and the extracted snippets to ./smoke_snippets/.
TILELANG_CACHE_DIR is set to ./tilelang_cache (wiped at start so every kernel is compiled by
this run). This file deliberately does not use `from __future__ import annotations` (see
SKILL.md 18.12).
"""

import importlib.metadata as md
import json
import os
import platform
import re
import runpy
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "tilelang_cache")
shutil.rmtree(CACHE, ignore_errors=True)
os.makedirs(CACHE, exist_ok=True)
os.environ["TILELANG_CACHE_DIR"] = CACHE  # must be set before importing tilelang

import torch  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

SKILL = open(os.path.join(HERE, "SKILL.md"), encoding="utf-8").read()
PRELUDE = "import torch\nimport tilelang\nimport tilelang.language as T\n"


def snippet(heading_prefix: str, must_contain: str = "@tilelang.jit") -> str:
    """Return the first ```python block after the heading starting with `heading_prefix`
    (within that section) that contains `must_contain`."""
    m = re.search(r"^#{2,4} " + re.escape(heading_prefix) + r".*$", SKILL, flags=re.M)
    if m is None:
        raise KeyError(f"heading not found: {heading_prefix}")
    nxt = re.search(r"^#{2,4} ", SKILL[m.end():], flags=re.M)
    section = SKILL[m.end(): m.end() + nxt.start()] if nxt else SKILL[m.end():]
    for block in re.findall(r"```python\n(.*?)```", section, flags=re.S):
        if must_contain in block:
            return block
    raise KeyError(f"no python block containing {must_contain!r} under {heading_prefix!r}")


SNIPPET_DIR = os.path.join(HERE, "smoke_snippets")
shutil.rmtree(SNIPPET_DIR, ignore_errors=True)
os.makedirs(SNIPPET_DIR, exist_ok=True)
_counter = [0]


def run_snippet(code: str, pre: str = "") -> dict:
    """Write the snippet to a real .py file (TileLang's frontend reads function source via
    inspect, which does not work for exec'd strings) and run it; return its globals."""
    _counter[0] += 1
    path = os.path.join(SNIPPET_DIR, f"snippet_{_counter[0]:02d}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PRELUDE + pre + code)
    return runpy.run_path(path, run_name="skill_snippet")


RESULTS: list[dict] = []


def test(name: str, source: str):
    def deco(fn):
        t0 = time.time()
        rec = {"name": name, "source": source}
        try:
            details = fn()
            rec["status"] = "pass"
            if details:
                rec["details"] = details
        except Exception as e:  # noqa: BLE001
            rec["status"] = "fail"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-3000:]
        rec["seconds"] = round(time.time() - t0, 2)
        RESULTS.append(rec)
        print(f"[{rec['status'].upper()}] {name} ({rec['seconds']}s)" + (f"  {rec.get('error', '')}" if rec["status"] == "fail" else ""), flush=True)
        return fn

    return deco


# ---------------------------------------------------------------- snippets from SKILL.md

@test("vector_add_lazy_out_idx", "SKILL.md 17.1")
def _():
    ns = run_snippet(snippet("17.1"))
    torch.testing.assert_close(ns["c"], ns["a"] + ns["b"])
    assert type(ns["kernel"]).__name__ == "JITKernel"
    assert ns["vector_add"](1000) is ns["kernel"], "JITKernel not cached per argument tuple"
    return {"N": 1000, "returned": type(ns["c"]).__name__}


@test("row_max_dynamic_M_masked_partial_tile", "SKILL.md 17.2")
def _():
    ns = run_snippet(snippet("17.2"))
    torch.testing.assert_close(ns["y"], ns["x"].max(dim=1).values)
    kern = ns["row_max"](1000)
    out = {}
    for M in (64, 129):  # same compiled kernel, other M (T.dynamic)
        x = -torch.rand(M, 1000, device="cuda") - 1.0  # all values negative: zero padding would be wrong
        y = torch.empty(M, device="cuda")
        r = kern(x, y)
        torch.testing.assert_close(y, x.max(dim=1).values)
        out[f"M={M}"] = "ok"
    out["return_without_out_idx"] = repr(r)
    assert r == [], "kernel without out_idx should return []"
    return out


@test("gemm_eager_fp16", "SKILL.md 17.3")
def _():
    ns = run_snippet(snippet("17.3"))
    a, b, c = ns["a"], ns["b"], ns["c"]
    torch.testing.assert_close(c, (a.float() @ b.float()).half(), rtol=1e-2, atol=1e-2)
    src = ns["matmul"].get_kernel_source(a, b)
    assert "tl::mma_sync" in src, "expected warp-level MMA for a fragment accumulator on sm_100"
    # non-multiple sizes reuse the same snippet with a new T.const specialization
    a2 = torch.randn(200, 100, device="cuda", dtype=torch.float16)
    b2 = torch.randn(100, 130, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(ns["matmul"](a2, b2), (a2.float() @ b2.float()).half(), rtol=1e-2, atol=1e-2)
    return {"shapes": [[256, 192, 320], [200, 100, 130]], "mma_sync_in_source": True}


@test("gemm_blackwell_tmem_tcgen05", "SKILL.md 9.4")
def _():
    ns = run_snippet(snippet("9.4"))
    kern = ns["matmul_tmem"](256, 256, 256)
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    c = kern(a, b)
    torch.testing.assert_close(c, (a.float() @ b.float().T).bfloat16(), rtol=1e-2, atol=1e-2)
    src = kern.get_kernel_source()
    assert "tcgen05" in src, "expected TCGEN5MMA in generated source"
    return {"M,N,K": [256, 256, 256], "tcgen05_in_source": True}


@test("lazy_and_eager_styles", "SKILL.md 3.2")
def _():
    ns = run_snippet(snippet("3.2"), pre='a = torch.randn(4096, device="cuda")\n')
    torch.testing.assert_close(ns["b"], ns["a"] + 1)
    assert type(ns["kernel"]).__name__ == "JITKernel"
    torch.testing.assert_close(ns["kernel"](ns["a"]), ns["a"] + 1)
    return {"eager_compile_returns": type(ns["kernel"]).__name__}


@test("plain_python_wrapper_dynamic_N_scalar_arg", "SKILL.md 4.4")
def _():
    ns = run_snippet(snippet("4.4"))
    for n in (1000, 3001):
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        torch.testing.assert_close(ns["run"](x, y, 2.0), 2.0 * x + y)
    # non-contiguous input goes through .contiguous() in the wrapper
    xs = torch.randn(2000, device="cuda")[::2]
    ys = torch.randn(1000, device="cuda")
    torch.testing.assert_close(ns["run"](xs, ys, 0.5), 0.5 * xs + ys)
    return {"sizes": [1000, 3001, "strided view 1000"]}


@test("row_cumsum", "SKILL.md 10.2")
def _():
    ns = run_snippet(snippet("10.2"))
    x = torch.randn(8, 256, device="cuda")
    torch.testing.assert_close(ns["row_cumsum"](8, 256)(x), x.cumsum(dim=1), rtol=1e-4, atol=1e-4)
    return {"M,N": [8, 256]}


@test("column_sums_atomic_add", "SKILL.md 11")
def _():
    ns = run_snippet(snippet("11. Atomics"))
    x = torch.randn(100, 200, device="cuda")
    y = torch.zeros(200, device="cuda")
    ns["column_sums"](100, 200)(x, y)
    torch.testing.assert_close(y, x.sum(dim=0), rtol=1e-4, atol=1e-4)
    return {"M,N": [100, 200]}


@test("macro_square", "SKILL.md 14", )
def _():
    ns = run_snippet(snippet("14. Macros", must_contain="@T.macro"))
    x = torch.randn(300, device="cuda")
    torch.testing.assert_close(ns["squares"](300)(x), x * x)
    return {"N": 300}


# ---------------------------------------------------------------- behavioural claims

@test("claim_current_torch_stream", "SKILL.md 4.2")
def _():
    ns = run_snippet(snippet("17.1"))
    n = 1 << 20
    kern = ns["vector_add"](n)
    src = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    a = torch.zeros(n, device="cuda")
    kern(a, b)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        torch.cuda._sleep(200_000_000)  # delay stream s
        a.copy_(src)                    # a is valid only after the delay, on stream s
        c = kern(a, b)                  # correct only if launched on s
    s.synchronize()
    torch.cuda.synchronize()
    torch.testing.assert_close(c, src + b)
    return {"launched_on_current_stream": True}


@test("claim_host_checks", "SKILL.md 4.1 / 4.5")
def _():
    ns = run_snippet(snippet("17.1"))
    kern = ns["vector_add"](1000)
    a = torch.randn(1000, device="cuda")
    out = {}
    try:
        kern(a)
        raise AssertionError("missing argument not rejected")
    except ValueError as e:
        out["arg_count"] = str(e)
    try:
        kern(torch.randn(2000, device="cuda")[::2], a)
        raise AssertionError("non-contiguous tensor not rejected")
    except RuntimeError as e:
        out["non_contiguous"] = str(e)[:160]
    try:
        kern(a.half(), a)
        raise AssertionError("dtype mismatch not rejected")
    except RuntimeError as e:
        out["dtype"] = str(e)[:160]
    return out


@test("claim_oob_global_load_reads_zero", "SKILL.md 6.2")
def _():
    @tilelang.jit(out_idx=[1])
    def row_max_unmasked(M, N, BN=128):
        @T.prim_func
        def main(X: T.Tensor((M, N), T.float32), Y: T.Tensor((M,), T.float32)):
            with T.Kernel(M, threads=128) as bm:
                xf = T.alloc_fragment((1, BN), T.float32)
                acc = T.alloc_fragment((1,), T.float32)
                T.fill(acc, -T.infinity(T.float32))
                for kn in T.serial(T.ceildiv(N, BN)):
                    T.copy(X[bm, kn * BN], xf)
                    T.reduce_max(xf, acc, dim=1, clear=False)
                Y[bm] = acc[0]
        return main

    x = -torch.rand(4, 1000, device="cuda") - 1.0
    y = row_max_unmasked(4, 1000)(x)
    assert torch.all(y == 0), f"expected zero padding to dominate the max, got {y.tolist()}"
    return {"unmasked_row_max_of_negative_rows": y.tolist()}


@test("claim_integer_division", "SKILL.md 8.1")
def _():
    @tilelang.jit(out_idx=[1, 2, 3, 4])
    def intops(N):
        @T.prim_func
        def main(X: T.Tensor((N,), T.int32), Q: T.Tensor((N,), T.int32), R: T.Tensor((N,), T.int32),
                 TQ: T.Tensor((N,), T.int32), TR: T.Tensor((N,), T.int32)):
            with T.Kernel(1, threads=32) as bx:
                for i in T.Parallel(N):
                    Q[i] = X[i] // 3
                    R[i] = X[i] % 3
                    TQ[i] = T.truncdiv(X[i], 3)
                    TR[i] = T.truncmod(X[i], 3)
        return main

    x = torch.tensor([-7, -6, -1, 0, 1, 5, 7, 8], device="cuda", dtype=torch.int32)
    q, r, tq, tr = intops(8)(x)
    torch.testing.assert_close(q, torch.div(x, 3, rounding_mode="floor"))
    torch.testing.assert_close(r, torch.remainder(x, 3))
    torch.testing.assert_close(tq, torch.div(x, 3, rounding_mode="trunc"))
    torch.testing.assert_close(tr, torch.fmod(x, 3))

    @tilelang.jit(out_idx=[1])
    def intdiv(N):
        @T.prim_func
        def main(X: T.Tensor((N,), T.int32), Y: T.Tensor((N,), T.int32)):
            with T.Kernel(1, threads=32) as bx:
                for i in T.Parallel(N):
                    Y[i] = X[i] / 3
        return main

    msg = None
    try:
        intdiv(8)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
    assert msg is not None, "integer / was not rejected"
    assert "integer divisions" in msg, msg[:200]
    return {"floor_div": q.tolist(), "floor_mod": r.tolist(), "integer_slash": "rejected"}


@test("claim_kernel_scope_statement_runs_on_every_thread", "SKILL.md 3.1 / 18.1")
def _():
    @tilelang.jit
    def counter(threads):
        @T.prim_func
        def main(Y: T.Tensor((2,), T.int32)):
            with T.Kernel(1, threads=threads) as bx:
                T.atomic_add(Y[0], 1)
                if T.get_thread_binding() == 0:
                    T.atomic_add(Y[1], 1)
        return main

    y = torch.zeros(2, device="cuda", dtype=torch.int32)
    counter(128)(y)
    assert y.tolist() == [128, 1], y.tolist()
    return {"counts": y.tolist()}


@test("claim_fp32_gemm_uses_tf32", "SKILL.md 9.2")
def _():
    @tilelang.jit(out_idx=[-1])
    def mm(M, N, K, bM=64, bN=64, bK=32):
        @T.prim_func
        def main(A: T.Tensor((M, K), T.float32), B: T.Tensor((K, N), T.float32), C: T.Tensor((M, N), T.float32)):
            with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=128) as (bx, by):
                As = T.alloc_shared((bM, bK), T.float32)
                Bs = T.alloc_shared((bK, bN), T.float32)
                Cf = T.alloc_fragment((bM, bN), T.float32)
                T.clear(Cf)
                for k in T.Pipelined(T.ceildiv(K, bK), num_stages=2):
                    T.copy(A[by * bM, k * bK], As)
                    T.copy(B[k * bK, bx * bN], Bs)
                    T.gemm(As, Bs, Cf)
                T.copy(Cf, C[by * bM, bx * bN])
        return main

    kern = mm(128, 128, 128)
    src = kern.get_kernel_source()
    assert "kTensorFloat32" in src, "expected TF32 MMA operands"
    a = torch.randn(128, 128, device="cuda")
    b = torch.randn(128, 128, device="cuda")
    torch.testing.assert_close(kern(a, b), a @ b, rtol=2e-2, atol=2e-1)
    return {"kTensorFloat32_in_source": True}


@test("claim_postponed_annotations_break_prim_func", "SKILL.md 18.12")
def _():
    code = (
        "from __future__ import annotations\n" + PRELUDE +
        "@tilelang.jit(out_idx=[-1])\n"
        "def k(N, block=128):\n"
        "    @T.prim_func\n"
        "    def main(A: T.Tensor((N,), T.float32), B: T.Tensor((N,), T.float32)):\n"
        "        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:\n"
        "            for i in T.Parallel(block):\n"
        "                B[bx * block + i] = A[bx * block + i] + 1\n"
        "    return main\n"
        "kern = k(256)\n"
    )
    path = os.path.join(SNIPPET_DIR, "postponed_annotations.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    msg = None
    try:
        runpy.run_path(path, run_name="skill_snippet")
    except TypeError as e:
        msg = str(e)
    assert msg is not None and "Forward references must evaluate to types" in msg, msg
    return {"python": platform.python_version(), "error": msg}


def main() -> None:
    props = torch.cuda.get_device_properties(0)
    out = {
        "generated_by": "smoke_tests.py",
        "versions": {
            "python": platform.python_version(),
            "tilelang": tilelang.__version__,
            "apache-tvm-ffi": md.version("apache-tvm-ffi"),
            "torch": torch.__version__,
            "cuda_runtime_torch": torch.version.cuda,
        },
        "gpu": {"name": props.name, "compute_capability": f"{props.major}.{props.minor}"},
        "tilelang_cache_dir": CACHE,
        "summary": {
            "total": len(RESULTS),
            "passed": sum(r["status"] == "pass" for r in RESULTS),
            "failed": [r["name"] for r in RESULTS if r["status"] != "pass"],
        },
        "tests": RESULTS,
    }
    with open(os.path.join(HERE, "smoke_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))
    sys.exit(0 if not out["summary"]["failed"] else 1)


if __name__ == "__main__":
    main()
