# Task: improve operator `flash_attention` for TileBench (iteration 8)

Your previous iteration (7) did not yet meet the stopping criterion (`stop_score` ≥ 80%, computed on the **largest 3 cases per dtype**). Read the trajectory below carefully — if your last iteration **regressed** vs the best verify-clean iter so far, you should consider going back to that approach as your starting point and trying a different optimization. Then re-emit BOTH files.

## Iteration trajectory so far

| iter | stop_score | report_geo | verify | autotune | notes |
|---|---:|---:|---|---|---|
| 0 | 19.7% | 18.9% | ✗ 20 fails | ok |  |
| 1 | 46.2% | 40.5% | ✗ 20 fails | ok |  |
| 2 | 57.1% | 50.6% | ✗ 20 fails | ok |  |
| 3 | 50.7% | 49.7% | ✗ 31 fails | ok |  |
| 4 | 0.0% | 0.0% | ✗ 40 fails | ok |  |
| 5 | 0.0% | 0.0% | ✗ 40 fails | ok |  |
| 6 | 0.0% | 0.0% | ✗ 40 fails | ok |  |
| 7 | 36.2% | 34.0% | ✗ 20 fails | ok |  |

**No verify-clean iteration yet — first achieve correctness on ALL (backend, dtype, case) combinations before optimising.**

## Feedback from iteration 7

### ❌ Verification failures (20 cases)
- `cutile` / dtype=`fp16` / params={'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 1024, 'dtype': 'fp16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 215, in _run_one_case
    output = impl.run(*inputs, autotune=True)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/flash_attention/gpt-5.5/high/iter_7/impl_cutile.py", line 173, in run
    cfg = _tuner.tune_or_cached(
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/core/cutile_autotune.py", line 109, in tune_or_cached
    result = ct.tune.exhaustive_search(
  File "/projects/kzhou6/bcui2/env_software/miniconda3/envs/tilebench_env/lib/python3.10/site-packages/cuda/tile/tune/_tune.py", line 264, in exhaustive_search
    raise ValueError(f"No valid config found in search space."
ValueError: No valid config found in search space.
Config: namespace(bm=64, bn=64, occupancy=4)
TileTypeError: No such attribute 'to' for object of type Tile[float32,(64,64)]
  "/projects/kzhou6/bcui2/research/tilebench/Tileben
- `cutile` / dtype=`fp16` / params={'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 2048, 'dtype': 'fp16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 215, in _run_one_case
    output = impl.run(*inputs, autotune=True)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/flash_attention/gpt-5.5/high/iter_7/impl_cutile.py", line 173, in run
    cfg = _tuner.tune_or_cached(
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/core/cutile_autotune.py", line 109, in tune_or_cached
    result = ct.tune.exhaustive_search(
  File "/projects/kzhou6/bcui2/env_software/miniconda3/envs/tilebench_env/lib/python3.10/site-packages/cuda/tile/tune/_tune.py", line 264, in exhaustive_search
    raise ValueError(f"No valid config found in search space."
ValueError: No valid config found in search space.
Config: namespace(bm=64, bn=64, occupancy=4)
TileTypeError: No such attribute 'to' for object of type Tile[float32,(64,64)]
  "/projects/kzhou6/bcui2/research/tilebench/Tileben
- `cutile` / dtype=`fp16` / params={'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 3072, 'dtype': 'fp16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 215, in _run_one_case
    output = impl.run(*inputs, autotune=True)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/flash_attention/gpt-5.5/high/iter_7/impl_cutile.py", line 173, in run
    cfg = _tuner.tune_or_cached(
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/core/cutile_autotune.py", line 109, in tune_or_cached
    result = ct.tune.exhaustive_search(
  File "/projects/kzhou6/bcui2/env_software/miniconda3/envs/tilebench_env/lib/python3.10/site-packages/cuda/tile/tune/_tune.py", line 264, in exhaustive_search
    raise ValueError(f"No valid config found in search space."
ValueError: No valid config found in search space.
Config: namespace(bm=64, bn=64, occupancy=4)
TileTypeError: No such attribute 'to' for object of type Tile[float32,(64,64)]
  "/projects/kzhou6/bcui2/research/tilebench/Tileben
- `cutile` / dtype=`fp16` / params={'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 4096, 'dtype': 'fp16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 215, in _run_one_case
    output = impl.run(*inputs, autotune=True)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/flash_attention/gpt-5.5/high/iter_7/impl_cutile.py", line 173, in run
    cfg = _tuner.tune_or_cached(
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/core/cutile_autotune.py", line 109, in tune_or_cached
    result = ct.tune.exhaustive_search(
  File "/projects/kzhou6/bcui2/env_software/miniconda3/envs/tilebench_env/lib/python3.10/site-packages/cuda/tile/tune/_tune.py", line 264, in exhaustive_search
    raise ValueError(f"No valid config found in search space."
ValueError: No valid config found in search space.
Config: namespace(bm=64, bn=64, occupancy=4)
TileTypeError: No such attribute 'to' for object of type Tile[float32,(64,64)]
  "/projects/kzhou6/bcui2/research/tilebench/Tileben
- `cutile` / dtype=`fp16` / params={'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 5120, 'dtype': 'fp16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 215, in _run_one_case
    output = impl.run(*inputs, autotune=True)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/flash_attention/gpt-5.5/high/iter_7/impl_cutile.py", line 173, in run
    cfg = _tuner.tune_or_cached(
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/core/cutile_autotune.py", line 109, in tune_or_cached
    result = ct.tune.exhaustive_search(
  File "/projects/kzhou6/bcui2/env_software/miniconda3/envs/tilebench_env/lib/python3.10/site-packages/cuda/tile/tune/_tune.py", line 264, in exhaustive_search
    raise ValueError(f"No valid config found in search space."
ValueError: No valid config found in search space.
Config: namespace(bm=64, bn=64, occupancy=4)
TileTypeError: No such attribute 'to' for object of type Tile[float32,(64,64)]
  "/projects/kzhou6/bcui2/research/tilebench/Tileben
- `cutile` / dtype=`fp16` / params={'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 6144, 'dtype': 'fp16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 215, in _run_one_case
    output = impl.run(*inputs, autotune=True)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/flash_attention/gpt-5.5/high/iter_7/impl_cutile.py", line 173, in run
    cfg = _tuner.tune_or_cached(
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/core/cutile_autotune.py", line 109, in tune_or_cached
    result = ct.tune.exhaustive_search(
  File "/projects/kzhou6/bcui2/env_software/miniconda3/envs/tilebench_env/lib/python3.10/site-packages/cuda/tile/tune/_tune.py", line 264, in exhaustive_search
    raise ValueError(f"No valid config found in search space."
ValueError: No valid config found in search space.
Config: namespace(bm=64, bn=64, occupancy=4)
TileTypeError: No such attribute 'to' for object of type Tile[float32,(64,64)]
  "/projects/kzhou6/bcui2/research/tilebench/Tileben
- `cutile` / dtype=`fp16` / params={'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 7168, 'dtype': 'fp16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 215, in _run_one_case
    output = impl.run(*inputs, autotune=True)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/flash_attention/gpt-5.5/high/iter_7/impl_cutile.py", line 173, in run
    cfg = _tuner.tune_or_cached(
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/core/cutile_autotune.py", line 109, in tune_or_cached
    result = ct.tune.exhaustive_search(
  File "/projects/kzhou6/bcui2/env_software/miniconda3/envs/tilebench_env/lib/python3.10/site-packages/cuda/tile/tune/_tune.py", line 264, in exhaustive_search
    raise ValueError(f"No valid config found in search space."
ValueError: No valid config found in search space.
Config: namespace(bm=64, bn=64, occupancy=4)
TileTypeError: No such attribute 'to' for object of type Tile[float32,(64,64)]
  "/projects/kzhou6/bcui2/research/tilebench/Tileben
- `cutile` / dtype=`fp16` / params={'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 8192, 'dtype': 'fp16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 215, in _run_one_case
    output = impl.run(*inputs, autotune=True)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/flash_attention/gpt-5.5/high/iter_7/impl_cutile.py", line 173, in run
    cfg = _tuner.tune_or_cached(
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/core/cutile_autotune.py", line 109, in tune_or_cached
    result = ct.tune.exhaustive_search(
  File "/projects/kzhou6/bcui2/env_software/miniconda3/envs/tilebench_env/lib/python3.10/site-packages/cuda/tile/tune/_tune.py", line 264, in exhaustive_search
    raise ValueError(f"No valid config found in search space."
ValueError: No valid config found in search space.
Config: namespace(bm=64, bn=64, occupancy=4)
TileTypeError: No such attribute 'to' for object of type Tile[float32,(64,64)]
  "/projects/kzhou6/bcui2/research/tilebench/Tileben
- `cutile` / dtype=`fp16` / params={'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 9216, 'dtype': 'fp16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 215, in _run_one_case
    output = impl.run(*inputs, autotune=True)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/flash_attention/gpt-5.5/high/iter_7/impl_cutile.py", line 173, in run
    cfg = _tuner.tune_or_cached(
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/core/cutile_autotune.py", line 109, in tune_or_cached
    result = ct.tune.exhaustive_search(
  File "/projects/kzhou6/bcui2/env_software/miniconda3/envs/tilebench_env/lib/python3.10/site-packages/cuda/tile/tune/_tune.py", line 264, in exhaustive_search
    raise ValueError(f"No valid config found in search space."
ValueError: No valid config found in search space.
Config: namespace(bm=64, bn=64, occupancy=4)
TileTypeError: No such attribute 'to' for object of type Tile[float32,(64,64)]
  "/projects/kzhou6/bcui2/research/tilebench/Tileben
- `cutile` / dtype=`fp16` / params={'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 10240, 'dtype': 'fp16'}: Traceback (most recent call last):
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/tools/llm_codegen/evaluator_runner.py", line 215, in _run_one_case
    output = impl.run(*inputs, autotune=True)
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/benchmarks/llm_generated/flash_attention/gpt-5.5/high/iter_7/impl_cutile.py", line 173, in run
    cfg = _tuner.tune_or_cached(
  File "/projects/kzhou6/bcui2/research/tilebench/Tilebench/core/cutile_autotune.py", line 109, in tune_or_cached
    result = ct.tune.exhaustive_search(
  File "/projects/kzhou6/bcui2/env_software/miniconda3/envs/tilebench_env/lib/python3.10/site-packages/cuda/tile/tune/_tune.py", line 264, in exhaustive_search
    raise ValueError(f"No valid config found in search space."
ValueError: No valid config found in search space.
Config: namespace(bm=64, bn=64, occupancy=4)
TileTypeError: No such attribute 'to' for object of type Tile[float32,(64,64)]
  "/projects/kzhou6/bcui2/research/tilebench/Tileben
  ... and 10 more.

### Performance (geo-mean roofline = **34.0%**, target ≥ 80%)

Per-(backend,dtype,case) detail (sorted by worst first):
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 1024, 'dtype': 'fp16'}: 23.0% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 2048, 'dtype': 'fp16'}: 29.0% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 3072, 'dtype': 'fp16'}: 31.5% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 4096, 'dtype': 'fp16'}: 32.8% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 5120, 'dtype': 'fp16'}: 33.6% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 6144, 'dtype': 'fp16'}: 34.2% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 7168, 'dtype': 'fp16'}: 34.6% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 8192, 'dtype': 'fp16'}: 34.9% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 9216, 'dtype': 'fp16'}: 35.1% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 10240, 'dtype': 'fp16'}: 35.3% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 11264, 'dtype': 'fp16'}: 35.5% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 12288, 'dtype': 'fp16'}: 35.6% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 13312, 'dtype': 'fp16'}: 35.7% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 14336, 'dtype': 'fp16'}: 35.8% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 15360, 'dtype': 'fp16'}: 35.9% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 16384, 'dtype': 'fp16'}: 36.0% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 17408, 'dtype': 'fp16'}: 36.1% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 18432, 'dtype': 'fp16'}: 36.1% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 19456, 'dtype': 'fp16'}: 36.2% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 20480, 'dtype': 'fp16'}: 36.2% of roofline (compute)

→ Action: focus on the worst-performing combinations. If the case is bandwidth-bound, optimize memory coalescing / tile layout / async copies. If compute-bound, ensure Tensor Cores (tl.dot/ct.mma) and high arithmetic intensity per load.

---

## Your previous `impl_triton.py` (iteration 7)

```python
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    base = pid_bh * N_CTX * HEAD_DIM

    q = tl.load(
        q_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    qk_scale = SM_SCALE * 1.4426950408889634

    m_i = tl.full((BLOCK_M,), -1.0e20, tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.full((BLOCK_M, BLOCK_D), 0.0, tl.float32)

    if CAUSAL:
        loop_end = (pid_m + 1) * BLOCK_M
    else:
        loop_end = N_CTX

    for start_n in tl.range(0, loop_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cols = start_n + offs_n

        k_t = tl.load(
            k_ptr + base + cols[None, :] * HEAD_DIM + offs_d[:, None],
            mask=(cols[None, :] < N_CTX) & (offs_d[:, None] < HEAD_DIM),
            other=0.0,
        )

        qk = tl.dot(q, k_t, out_dtype=tl.float32) * qk_scale

        valid = cols[None, :] < N_CTX
        if CAUSAL:
            valid = valid & (cols[None, :] <= offs_m[:, None])
        qk = tl.where(valid, qk, -1.0e20)

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)

        v = tl.load(
            v_ptr + base + cols[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(cols[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        acc = acc * alpha[:, None] + tl.dot(
            p.to(tl.float16),
            v,
            out_dtype=tl.float32,
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]

    tl.store(
        o_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


_flash_attention_fwd_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    ],
    key=["N_CTX", "HEAD_DIM", "CAUSAL", "BLOCK_D"],
)(_flash_attention_fwd_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch_size, n_heads, seq_len, head_dim = q.shape
    output = torch.empty_like(q)

    batch_heads = batch_size * n_heads
    block_d = max(32, _next_power_of_2(head_dim))
    sm_scale = 1.0 / math.sqrt(head_dim)

    grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_M"]), batch_heads)

    _flash_attention_fwd_kernel_autotuned[grid](
        q,
        k,
        v,
        output,
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        SM_SCALE=sm_scale,
        CAUSAL=bool(causal),
        BLOCK_D=block_d,
    )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_flash_attention_fwd_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_M": cfg.kwargs["BLOCK_M"],
        "BLOCK_N": cfg.kwargs["BLOCK_N"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }

```

## Your previous `impl_cutile.py` (iteration 7)

```python
from types import SimpleNamespace
import math
import torch
import cuda.tile as ct
from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]

_last_autotune_config: dict = {}

_SEARCH_SPACE = [
    SimpleNamespace(bm=64, bn=64, occupancy=4),
]


@ct.kernel
def _flash_attention_fwd_kernel(
    q,
    k,
    v,
    out,
    N_CTX: ConstInt,
    HEAD_DIM: ConstInt,
    SM_SCALE: ConstFloat,
    CAUSAL: ConstBool,
    BM: ConstInt,
    BN: ConstInt,
    BD: ConstInt,
):
    pid_m = ct.bid(0)
    pid_bh = ct.bid(1)

    q_tile_id = pid_bh * (N_CTX // BM) + pid_m

    q_tile = ct.load(
        q,
        index=(q_tile_id, 0),
        shape=(BM, BD),
        padding_mode=ct.PaddingMode.ZERO,
    )

    acc = ct.full((BM, BD), 0.0, ct.float32)
    m_i = ct.full((BM,), -1.0e20, ct.float32)
    l_i = ct.full((BM,), 0.0, ct.float32)

    rows = pid_m * BM + ct.arange(BM, dtype=ct.int32)
    qk_scale = SM_SCALE * 1.4426950408889634

    if CAUSAL:
        for start_n in range(0, N_CTX, BN):
            kv_tile_id = pid_bh * (N_CTX // BN) + (start_n // BN)

            k_tile = ct.load(
                k,
                index=(kv_tile_id, 0),
                shape=(BN, BD),
                padding_mode=ct.PaddingMode.ZERO,
            )

            scores = ct.mma(
                q_tile,
                ct.transpose(k_tile),
                ct.full((BM, BN), 0.0, ct.float32),
            )
            scores = scores * qk_scale

            cols = start_n + ct.arange(BN, dtype=ct.int32)
            valid = (cols[None, :] <= rows[:, None]) & (cols[None, :] < N_CTX)
            scores = ct.where(valid, scores, -1.0e20)

            m_ij = ct.max(scores, axis=1)
            m_new = ct.maximum(m_i, m_ij)

            p = ct.exp2(scores - m_new[:, None])
            alpha = ct.exp2(m_i - m_new)

            v_tile = ct.load(
                v,
                index=(kv_tile_id, 0),
                shape=(BN, BD),
                padding_mode=ct.PaddingMode.ZERO,
            )

            acc = acc * alpha[:, None]
            acc = ct.mma(
                p.to(ct.float16),
                v_tile,
                acc,
            )
            l_i = l_i * alpha + ct.sum(p, axis=1)
            m_i = m_new
    else:
        for start_n in range(0, N_CTX, BN):
            kv_tile_id = pid_bh * (N_CTX // BN) + (start_n // BN)

            k_tile = ct.load(
                k,
                index=(kv_tile_id, 0),
                shape=(BN, BD),
                padding_mode=ct.PaddingMode.ZERO,
            )

            scores = ct.mma(
                q_tile,
                ct.transpose(k_tile),
                ct.full((BM, BN), 0.0, ct.float32),
            )
            scores = scores * qk_scale

            cols = start_n + ct.arange(BN, dtype=ct.int32)
            scores = ct.where(cols[None, :] < N_CTX, scores, -1.0e20)

            m_ij = ct.max(scores, axis=1)
            m_new = ct.maximum(m_i, m_ij)

            p = ct.exp2(scores - m_new[:, None])
            alpha = ct.exp2(m_i - m_new)

            v_tile = ct.load(
                v,
                index=(kv_tile_id, 0),
                shape=(BN, BD),
                padding_mode=ct.PaddingMode.ZERO,
            )

            acc = acc * alpha[:, None]
            acc = ct.mma(
                p.to(ct.float16),
                v_tile,
                acc,
            )
            l_i = l_i * alpha + ct.sum(p, axis=1)
            m_i = m_new

    out_tile = acc / l_i[:, None]

    ct.store(
        out,
        index=(q_tile_id, 0),
        tile=out_tile,
    )


_tuner = CutileAutotuner(_flash_attention_fwd_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch_size, n_heads, seq_len, head_dim = q.shape
    batch_heads = batch_size * n_heads

    output = torch.empty_like(q)

    q2 = q.reshape(batch_heads * seq_len, head_dim)
    k2 = k.reshape(batch_heads * seq_len, head_dim)
    v2 = v.reshape(batch_heads * seq_len, head_dim)
    out2 = output.reshape(batch_heads * seq_len, head_dim)

    bd = max(32, _next_power_of_2(head_dim))
    sm_scale = 1.0 / math.sqrt(head_dim)

    stream = torch.cuda.current_stream()

    cfg = _tuner.tune_or_cached(
        shape_key=(batch_heads, seq_len, head_dim, bool(causal)),
        search_space=_SEARCH_SPACE,
        stream=stream,
        grid_fn=lambda cfg: (ct.cdiv(seq_len, cfg.bm), batch_heads, 1),
        args_fn=lambda cfg: (
            q2,
            k2,
            v2,
            out2,
            seq_len,
            head_dim,
            sm_scale,
            bool(causal),
            cfg.bm,
            cfg.bn,
            bd,
        ),
        hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
    )

    _last_autotune_config.clear()
    _last_autotune_config.update(
        {
            "bm": cfg.bm,
            "bn": cfg.bn,
            "occupancy": cfg.occupancy,
        }
    )

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)

    ct.launch(
        stream,
        (ct.cdiv(seq_len, cfg.bm), batch_heads, 1),
        kernel,
        (
            q2,
            k2,
            v2,
            out2,
            seq_len,
            head_dim,
            sm_scale,
            bool(causal),
            cfg.bm,
            cfg.bn,
            bd,
        ),
    )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None

```

---

# Reminders: TileBench Framework Conventions

# TileBench Framework Guide

Read this carefully. Your generated kernels MUST follow this contract or the
benchmark harness will fail to load / run / evaluate them.

## Files you must produce

For operator `<op>`, write exactly two files:

1. **`impl_triton.py`** — Triton kernel + Python `run()` wrapper
2. **`impl_cutile.py`** — cuTile kernel + Python `run()` wrapper

You do NOT generate `impl_torch.py` — it is copied from the human-written
reference implementation. Your `run()` function's signature MUST match
`impl_torch.run()` exactly (same positional args, same keyword args).

## Required exports (both impls)

```python
def run(*args, autotune: bool = False, **kwargs):
    """Execute the operator. Must produce a single torch.Tensor (or tuple
    of tensors) that bit-equivalently matches impl_torch.run() within the
    tolerances in config.yaml's `verify:` section."""

def get_last_config() -> dict | None:
    """Return the autotune-winner config as a flat dict, or None if no
    autotune has run yet. Engine uses this to log per-case best cfgs."""
```

When called with `autotune=True`, `run()` must perform autotune internally.
When called with `autotune=False`, `run()` must use SOME default config —
since you are NOT required to write `_DEFAULT_CONFIG`, the simplest path
is to ALWAYS run autotune. The harness's autotune cache makes repeated
runs cheap (autotune sweeps once per shape, then caches).

## Autotune convention

### Triton

```python
import triton
import triton.language as tl

@triton.jit
def _op_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # ... kernel body ...

_op_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [512, 1024, 2048]      # keep cfg space small (<= 30 cfgs)
        for nw in [2, 4, 8]              # autotune budget is 15min per backend
        for ns in [2, 3]
    ],
    key=["n_elements"],                  # cache key: which problem dims trigger re-tune
)(_op_kernel)

def run(x, autotune: bool = False):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _op_kernel_autotuned[grid](x, output, n_elements)
    return output

def get_last_config() -> dict | None:
    cfg = getattr(_op_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps":  cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
```

**Rules:**
- `triton.autotune` caches `best_config` on the kernel object automatically.
- Do NOT use a module-level `_last_config` global — read `kernel.best_config`.
- Keep total cfg space ≤ 30; bigger spaces blow the 15-minute autotune cap.
- `num_warps` and `num_stages` are **optional** — include them only if your
  kernel actually benefits from a non-default value (e.g. a software-pipelined
  matmul gains from `num_stages` sweep; a one-pass elementwise rarely does).
  Reasonable ranges when you do sweep them: `num_warps` ∈ {2, 4, 8},
  `num_stages` ∈ {1, 2, 3, 4}.

### cuTile

```python
from types import SimpleNamespace
import cuda.tile as ct
from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

# Module-level dict — DO NOT use `global` keyword; use .clear() + .update().
_last_autotune_config: dict = {}

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [512, 1024, 2048]            # keep search space small
    for occ in [4, 8, 16]                  # nw * occ ≈ 64 on B200
]

@ct.kernel
def _op_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,))
    ct.store(output, index=(bid,), tile=x_tile)   # ... your compute ...

_tuner = CutileAutotuner(_op_kernel)

def run(x, autotune: bool = False):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()
    cfg = _tuner.tune_or_cached(
        shape_key=(n_elements,),
        search_space=_SEARCH_SPACE,
        stream=stream,
        grid_fn=lambda cfg: (ct.cdiv(n_elements, cfg.tile), 1, 1),
        args_fn=lambda cfg: (x, output, cfg.tile),
        hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
    )
    _last_autotune_config.clear()
    _last_autotune_config.update({"tile": cfg.tile, "occupancy": cfg.occupancy})
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream,
              (ct.cdiv(n_elements, cfg.tile), 1, 1),
              kernel,
              (x, output, cfg.tile))
    return output

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
```

**Rules:**
- `_last_autotune_config` must be a module-level `dict` mutated via
  `.clear()` + `.update()`. Never use `global _last_autotune_config`.
- cuTile tile-shape dimensions MUST be powers of 2.
- For non-power-of-2 problem dims: use `padding_mode=ct.PaddingMode.ZERO`
  on `ct.load`, and `ct.store` silently drops OOB writes.
- For max-reduction kernels needing to neutralize OOB: use
  `padding_mode=ct.PaddingMode.NEG_INF`.
- `ct.store()` only accepts static indices. For runtime-computed scatter
  indices, use `ct.scatter()`.

## Choosing meaningful tile / BLOCK sizes

A common LLM mistake is to throw every small tile size into the autotune
search space "to be thorough". Tiles below the thresholds below are
provably suboptimal on B200 (each thread loads / computes too few elements
to amortise launch overhead, miss memory coalescing windows, or fall off
the Tensor Core path). Including them wastes autotune budget on configs
that cannot win.

**Lower bounds — do NOT sweep below these.** All values must be powers of 2.

| Operator shape | Triton `BLOCK_SIZE` / cuTile `tile` | Triton matmul `BLOCK_M`,`BLOCK_N` | Triton matmul `BLOCK_K` | cuTile matmul `tm`,`tn` | cuTile matmul `tk` |
|---|---|---|---|---|---|
| 1D elementwise / pointwise | ≥ **512** | — | — | — | — |
| Row-reduction / softmax / norm (per-row inner tile) | ≥ **256** along reduction dim | — | — | — | — |
| Stencil / 2D conv (sliding window) | ≥ **256** along the spatial inner dim | — | — | — | — |
| Matmul / attention (Tensor Core) | — | ≥ **64** | ≥ **32** | ≥ **64** | ≥ **32** |

**Upper bounds — do NOT sweep above these.** Larger tiles run out of
registers / shared memory on B200 (sm_100, 228 KB shmem, 64 K registers
per CTA).

| Operator shape | Reasonable upper bound |
|---|---|
| 1D elementwise / pointwise | `BLOCK_SIZE` ≤ **8192** for fp16/fp32; ≤ **16384** for int8 |
| Matmul tile area | `BLOCK_M × BLOCK_N` ≤ **256 × 256** for fp16, ≤ **128 × 256** for fp32, ≤ **256 × 256** for fp8/int8 |
| Matmul K-tile | `BLOCK_K` ≤ **128** for fp16/fp8/int8, ≤ **64** for fp32 |

**Recommended sweep ranges (start here, prune if 30-cfg budget is tight):**

| Use case | Recommended config space |
|---|---|
| 1D pointwise (Triton) | `BLOCK_SIZE` ∈ {512, 1024, 2048, 4096}, `num_warps` ∈ {4, 8} |
| 1D pointwise (cuTile) | `tile` ∈ {512, 1024, 2048, 4096}, `occupancy` ∈ {4, 8, 16} |
| Per-row reduction (Triton, 1 CTA per row) | `BLOCK_N` ∈ {256, 512, 1024, 2048}, `num_warps` ∈ {2, 4, 8} |
| Matmul (Triton, fp16/bf16) | `BLOCK_M`,`BLOCK_N` ∈ {64, 128, 256}, `BLOCK_K` ∈ {32, 64, 128}, `num_warps` ∈ {4, 8}, `num_stages` ∈ {2, 3, 4} — prune to ≤ 30 cfgs |
| Matmul (cuTile, fp16/bf16) | `tm`,`tn` ∈ {64, 128, 256}, `tk` ∈ {32, 64, 128}, `occupancy` ∈ {4, 8, 16}, `group_size_m` ∈ {8} |

**Why this matters.** A `BLOCK_SIZE=128` config for a 20M-element fp32
pointwise has only 32 elements per warp (1 element per thread with
`num_warps=4`), missing vectorised loads, missing memory coalescing,
and forcing 156× more CTAs than `BLOCK_SIZE=2048`. The autotune
sweep will dutifully benchmark this config and reject it, but every
small config you include costs roughly 1-15 seconds of compile +
benchmark time per problem shape — for matmul, that adds tens of
minutes per iteration with no chance of a win.

## Multi-dtype support

If `config.yaml`'s `case_grid.dtype` lists multiple dtypes (e.g.
`["fp16", "bf16", "fp32"]`), your `run()` function must work for **all
of them** in a single `run()` call. Two acceptable patterns:

1. **Single kernel, dtype-polymorphic** — Triton's `tl.dot` and cuTile's
   `ct.mma` adapt to input dtype automatically. The cleanest path.
2. **Per-dtype branches inside `run()`** — if e.g. fp8 needs different
   cast logic, branch on `a.dtype` in the Python wrapper, NOT in two
   separate kernels.

For mixed-dtype matmul (e.g. fp32 + fp16 + fp8 in one op), look at
`benchmarks/operators/matmul_fp32_fp16_fp8/impl_triton.py` for reference.

## Hardware constraints to remember (B200, sm_100)

- **Tensor Memory Accelerator (TMA)** via `tl.make_tensor_descriptor`:
  the descriptor's `block_shape` last dimension must occupy ≥ 16 bytes
  (i.e. ≥ 8 elements for fp16, ≥ 4 for fp32, ≥ 16 for int8). If your op
  is 1D pointwise, do NOT use TMA descriptors — use plain `tl.load`.
- **Tensor Core**: `tl.dot` / `ct.mma` triggers MMA hardware. Triton's
  default `input_precision="tf32"` promotes fp32 inputs to TF32 TC
  (1100 TFLOPS dense ceiling). Use this unless strict IEEE-754 fp32 is
  required.
- **Powers of two**: cuTile tile dims must be powers of 2; non-pow2
  cases use padding.

## ⛔️ FORBIDDEN: delegating the actual computation to PyTorch / cuDNN / cuBLAS

Your `run()` function in `impl_triton.py` MUST perform the operator's
computation through your `@triton.jit` kernel, and your `run()` in
`impl_cutile.py` MUST perform it through your `@ct.kernel`. You are
NOT allowed to compute the output by calling any of the following from
inside `run()` (this list is non-exhaustive but covers the common cheats
the harness detects):

- `torch.nn.functional.*` — `scaled_dot_product_attention`, `softmax`,
  `layer_norm`, `linear`, `conv1d`/`conv2d`/`conv3d`, `cross_entropy`,
  `max_pool2d`, `batch_norm`, `dropout`, etc.
- `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.einsum`,
  `torch.linalg.*`
- `torch.softmax`, `torch.argmax`, `torch.sort`, `torch.topk`,
  `torch.bincount`, `torch.flip`, `torch.transpose` *as a substitute
  for the actual operator's computation*
- Any function whose name on its own is a literal description of the
  operator (e.g. you cannot use `torch.norm` inside the `l2_norm`
  operator's `run`).
- Importing the reference (`from impl_torch import run`).

The harness will reject any `impl_triton.py` or `impl_cutile.py` whose
`run()` body contains a call to a forbidden pattern, even if the kernel
file *also* defines a `@triton.jit`/`@ct.kernel` function. A "decoy"
kernel that is defined but not invoked on the real workload, or invoked
only on a placeholder one-element input, will be detected and rejected.

Valid uses of `torch.*` inside `run()` are limited to: tensor allocation
(`torch.empty`, `torch.empty_like`, `torch.zeros`, `torch.zeros_like`),
shape manipulation that does not compute (`.contiguous()`, `.view()`,
`.reshape()`, `.transpose()`, `.unsqueeze()`, `.permute()` — but
\textbf{not} as a substitute for the operator), stream / event
management (`torch.cuda.current_stream`, `torch.cuda.synchronize`), and
dtype-only casts on metadata (\textbf{not} on the data path that should
be computed by your kernel).

## ⛔️ FORBIDDEN: caching outputs across `run()` calls

Do NOT add a Python-level cache that returns a previously-computed output when
the inputs look identical. The harness will pass the same input objects (or
freshly-regenerated tensors with the same values) to `run()` many times during
warmup, verify, and timing — and an output cache would make the kernel appear
to run in ~0 ms, inflating the measured throughput far past the hardware
roofline. The harness detects this (it regenerates fresh inputs between
warmup and the timed loop) and will flag any implementation whose measured
throughput exceeds the hardware roofline as a cache cheat.

Every `run()` call must perform the actual GPU work — no Python shortcuts,
no `_OUTPUT_CACHE`-style dicts, no `is`/`data_ptr`/`_version` shortcuts that
skip the kernel launch. Triton's per-cfg compilation cache and cuTile's
autotune cache are fine (they cache the **compiled kernel + best cfg**, not
the **output tensor**).

## Common pitfalls (the harness rejects these)

1. **Importing modules you didn't list**: stay within `torch`, `triton`,
   `triton.language`, `cuda.tile`, `cuda.tile_experimental` (optional),
   `core.cutile_autotune`, `math`, `numpy as np`.
2. **Wrong `run()` signature**: if `impl_torch.run(x, y, BATCH, M, N, K)`
   takes 6 positional args, your `run()` must take the same 6.
3. **Returning a list when impl_torch returns a tensor** (or vice versa).
4. **Not handling OOB tail tiles**: input sizes from `config.yaml` may
   not be exact multiples of your `BLOCK_SIZE`. Always mask.
5. **Mutating inputs in-place**: the harness re-uses input tensors
   across the verify + timing repetitions. Clone if you need to.
6. **`get_last_config()` returning wrong type**: must return `dict` or
   `None`. Do not return `Config` objects or `SimpleNamespace`.

## What the engine does with your code

For each `(case, dtype)` from `config.yaml`'s `case_grid`:
1. Generates inputs via `data/tensors.py`'s `GENERATORS[<op>]`.
2. Calls `impl_torch.run(*inputs)` → reference output.
3. Calls `impl_triton.run(*inputs, autotune=True)` → triton output.
4. `verify(triton_output, ref_output, atol, rtol)` with per-dtype tolerances
   (see `core/verifier.py`). On failure, no timing is collected for that backend.
5. If verify passes: times `impl_triton.run(*inputs, autotune=True)` over
   `repeat=100` iterations using NVIDIA Proton. Mean latency is recorded.
6. Same flow for `impl_cutile.run()`.

The metric for stopping is the **geometric mean of `roofline_pct` across
all `(backend, dtype, case)` combinations** ≥ 0.80, where `roofline_pct`
= measured_FLOPS_per_sec / `min(peak_compute_dtype, peak_bw × AI)`.

## TL;DR checklist before you return code

- [ ] Two files: `impl_triton.py` and `impl_cutile.py`
- [ ] Each exports `run(...)` and `get_last_config() -> dict | None`
- [ ] `run()` signature matches `impl_torch.run()` exactly
- [ ] Triton autotune via `triton.autotune` decorator, ≤ 30 cfgs
- [ ] cuTile autotune via `CutileAutotuner`, ≤ 30 cfgs
- [ ] No `_DEFAULT_CONFIG`, no `global` keyword for `_last_autotune_config`
- [ ] Multi-dtype handled in a single `run()`
- [ ] OOB / non-pow-2 handled with masks (Triton) or `padding_mode` (cuTile)
- [ ] No in-place mutation of inputs


---

# Operator description: `flash_attention`

# Flash Attention (Forward, Causal)

Implement a program that performs scaled dot-product attention with causal masking, using the FlashAttention algorithm (Dao 2022) — i.e. tile-by-tile online softmax with running max and denominator so the full attention matrix is never materialized.

The input consists of:

- `q`: A 4D array of shape $(batch\_size, n\_heads, seq\_len, head\_dim)$.
- `k`: A 4D array of shape $(batch\_size, n\_heads, seq\_len, head\_dim)$ (same shape as `q`).
- `v`: A 4D array of shape $(batch\_size, n\_heads, seq\_len, head\_dim)$.
- `causal`: A boolean flag; when `True`, position $i$ may only attend to positions $j \leq i$.

The output should be written to the `output` array of shape $(batch\_size, n\_heads, seq\_len, head\_dim)$, with the same dtype as `q`.

Let $s = 1 / \sqrt{head\_dim}$. The operation is defined mathematically as:

$$
attn[b, h, i, j] = \frac{q[b, h, i, :] \cdot k[b, h, j, :]}{\sqrt{head\_dim}}
$$

When `causal=True`, set $attn[b, h, i, j] = -\infty$ for $j > i$. Then softmax over $j$ and weighted-sum the values:

$$
output[b, h, i, :] = \sum_{j=0}^{seq\_len-1} \text{softmax}_j\!\big( attn[b, h, i, :] \big) \cdot v[b, h, j, :]
$$

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- The kernel **must use the FlashAttention online-softmax algorithm**: tile-by-tile K/V loading with running $(m_i, \ell_i, acc_i)$ state — do NOT materialize a full $(seq\_len, seq\_len)$ attention matrix in global memory
- Tensor cores must be used for both the QK and PV matmuls
- Accumulators (running max, denominator, output sum) in fp32; final cast to input dtype

## Example 1

Conceptual single-head, head_dim=2, seq_len=2, causal=True:

```text
q = [[1, 0], [0, 1]], k = [[1, 0], [0, 1]], v = [[10, 20], [30, 40]]
scaled qk = [[1/sqrt(2), 0], [0, 1/sqrt(2)]]
After causal mask: qk[0,:] = [1/sqrt(2), -inf]; qk[1,:] = [0, 1/sqrt(2)]
softmax(qk[0]) = [1.0, 0.0]; softmax(qk[1]) ≈ [0.330, 0.670]
output[0] = 1.0 * [10, 20] = [10, 20]
output[1] ≈ 0.330 * [10, 20] + 0.670 * [30, 40] ≈ [23.4, 33.4]
```

## Constraints

- $batch\_size \geq 1$, $n\_heads \geq 1$, $seq\_len \geq 1$, $head\_dim \geq 1$
- `causal` is the typical case (autoregressive generation)
- Q, K, V have identical shape (full attention, not GQA)


---

# `config.yaml` for `flash_attention`

```yaml
verify:
  atol: 5e-3
  rtol: 5e-3

benchmark:
  warmup: 20
  repeat: 100
  use_cuda_graph: true
  flush_l2: true
  autotune: false

case_defaults:
  batch_size: 4
  n_heads: 32
  head_dim: 128
  causal: true

case_grid:
  seq_len:
    expr: "[1024 * i for i in range(1, 21)]"
  dtype: ["fp16"]

metrics:
  flops_expr: "4 * batch_size * n_heads * seq_len * seq_len * head_dim"
  bytes_expr: "4 * batch_size * n_heads * seq_len * head_dim * dtype_size"
  plots:
    - latency_ms
    - bandwidth_GBs
    - speedup

```

---

# PyTorch reference: `impl_torch.py`

```python
import torch
from torch.nn.functional import scaled_dot_product_attention

def run(q, k, v, causal=True, **kwargs):
    return scaled_dot_product_attention(q, k, v, is_causal=causal)
```

---

## Output format (STRICT)

Return EXACTLY two fenced code blocks, in this order, with these titles:

    ```python title="impl_triton.py"
    # full Python file content here
    ```

    ```python title="impl_cutile.py"
    # full Python file content here
    ```

No other code blocks. No prose between or after the two blocks beyond a 1-2
sentence summary of your approach. Do NOT include test code, do NOT include
PyTorch reference code (that file is provided by the framework).
