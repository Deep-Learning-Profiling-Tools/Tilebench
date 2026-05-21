# Task: improve operator `flash_attention` for TileBench (iteration 5)

Your previous iteration (4) did not yet meet the stopping criterion (`stop_score` ≥ 80%, computed on the **largest 3 cases per dtype**). Read the trajectory below carefully — if your last iteration **regressed** vs the best verify-clean iter so far, you should consider going back to that approach as your starting point and trying a different optimization. Then re-emit BOTH files.

## Iteration trajectory so far

| iter | stop_score | report_geo | verify | autotune | notes |
|---|---:|---:|---|---|---|
| 0 | 46.1% | 44.1% | ✓ clean | ok |  |
| 1 | 54.1% | 49.8% | ✓ clean | ok |  |
| 2 | 45.9% | 41.1% | ✗ 6 fails | ok |  |
| 3 | 53.7% | 49.4% | ✓ clean | ok |  |
| 4 | 55.7% | 51.1% | ✓ clean | ok |  |

**Best verify-clean iter so far: iter 4 with stop_score=55.7% and report_geo=51.1%.**

## Feedback from iteration 4

### Performance (mean roofline = **51.1%**, target ≥ 80%)

Per-(backend,dtype,case) detail (sorted by worst first):
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 1024, 'dtype': 'fp16'}: 27.4% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 1024, 'dtype': 'fp16'}: 30.0% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 2048, 'dtype': 'fp16'}: 37.7% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 2048, 'dtype': 'fp16'}: 40.1% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 3072, 'dtype': 'fp16'}: 43.0% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 3072, 'dtype': 'fp16'}: 44.9% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 4096, 'dtype': 'fp16'}: 46.6% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 4096, 'dtype': 'fp16'}: 47.3% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 5120, 'dtype': 'fp16'}: 49.1% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 5120, 'dtype': 'fp16'}: 49.2% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 6144, 'dtype': 'fp16'}: 50.4% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 6144, 'dtype': 'fp16'}: 50.9% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 7168, 'dtype': 'fp16'}: 51.3% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 8192, 'dtype': 'fp16'}: 51.9% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 7168, 'dtype': 'fp16'}: 52.2% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 9216, 'dtype': 'fp16'}: 52.4% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 10240, 'dtype': 'fp16'}: 52.9% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 11264, 'dtype': 'fp16'}: 53.0% of roofline (compute)
- `triton` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 8192, 'dtype': 'fp16'}: 53.1% of roofline (compute)
- `cutile` / `fp16` / {'batch_size': 4, 'n_heads': 32, 'head_dim': 128, 'causal': True, 'seq_len': 12288, 'dtype': 'fp16'}: 53.3% of roofline (compute)
  ... and 20 more cases.

→ Action: focus on the worst-performing combinations. If the case is bandwidth-bound, optimize memory coalescing / tile layout / async copies. If compute-bound, ensure Tensor Cores (tl.dot/ct.mma) and high arithmetic intensity per load.

---

## Your previous `impl_triton.py` (iteration 4)

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
    scale_log2,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    base = pid_bh * N_CTX * D_HEAD

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + base
        + offs_m[:, None] * D_HEAD
        + offs_d[None, :]
    )

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    if CAUSAL:
        full_end = (pid_m * BLOCK_M // BLOCK_N) * BLOCK_N

        for start_n in tl.range(0, full_end, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                k_ptr
                + base
                + offs_d[:, None]
                + offs_n[None, :] * D_HEAD,
                eviction_policy="evict_last",
            )

            qk = tl.dot(q, k, out_dtype=tl.float32) * scale_log2

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)

            v = tl.load(
                v_ptr
                + base
                + offs_n[:, None] * D_HEAD
                + offs_d[None, :],
                eviction_policy="evict_last",
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

        hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)

        for start_n in tl.range(full_end, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                k_ptr
                + base
                + offs_d[:, None]
                + offs_n[None, :] * D_HEAD,
                eviction_policy="evict_last",
            )

            qk = tl.dot(q, k, out_dtype=tl.float32) * scale_log2
            qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)

            v = tl.load(
                v_ptr
                + base
                + offs_n[:, None] * D_HEAD
                + offs_d[None, :],
                eviction_policy="evict_last",
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij
    else:
        for start_n in tl.range(0, N_CTX, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                k_ptr
                + base
                + offs_d[:, None]
                + offs_n[None, :] * D_HEAD,
                eviction_policy="evict_last",
            )

            qk = tl.dot(q, k, out_dtype=tl.float32) * scale_log2

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)

            v = tl.load(
                v_ptr
                + base
                + offs_n[:, None] * D_HEAD
                + offs_d[None, :],
                eviction_policy="evict_last",
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

    out = acc / l_i[:, None]

    tl.store(
        o_ptr
        + base
        + offs_m[:, None] * D_HEAD
        + offs_d[None, :],
        out,
    )


_flash_attention_fwd_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=3),
    ],
    key=["N_CTX", "D_HEAD", "CAUSAL"],
)(_flash_attention_fwd_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    B, H, N_CTX, D_HEAD = q.shape
    output = torch.empty_like(q)

    block_d = max(32, _next_power_of_2(int(D_HEAD)))
    scale_log2 = (1.0 / math.sqrt(float(D_HEAD))) * 1.4426950408889634
    causal = bool(causal)

    grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H)

    _flash_attention_fwd_autotuned[grid](
        q,
        k,
        v,
        output,
        scale_log2,
        N_CTX=N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_D=block_d,
        CAUSAL=causal,
    )
    return output


def get_last_config() -> dict | None:
    cfg = getattr(_flash_attention_fwd_autotuned, "best_config", None)
    if cfg is None:
        return None
    out = dict(cfg.kwargs)
    out["num_warps"] = cfg.num_warps
    out["num_stages"] = cfg.num_stages
    return out

```

## Your previous `impl_cutile.py` (iteration 4)

```python
from types import SimpleNamespace
import math
import torch
import cuda.tile as ct
import numpy as np
from cuda.tile import RoundingMode as RMd

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_last_autotune_config: dict = {}


@ct.kernel
def _flash_attention_kernel(
    q,
    k,
    v,
    out,
    scale_log2: float,
    N: ConstInt,
    D: ConstInt,
    CAUSAL: ConstBool,
    BM: ConstInt,
    BN: ConstInt,
    BD: ConstInt,
):
    bid_m = ct.bid(0)
    bid_bh = ct.bid(1)

    offs_m = (bid_m * BM + ct.arange(BM, dtype=np.int32))[:, None]
    offs_n_base = ct.arange(BN, dtype=np.int32)[None, :]

    q_tile = ct.load(
        q,
        index=(bid_bh, bid_m, 0),
        shape=(1, BM, BD),
    ).reshape((BM, BD))

    m_i = ct.full((BM, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((BM, 1), 0.0, dtype=np.float32)
    acc = ct.full((BM, BD), 0.0, dtype=np.float32)

    if CAUSAL:
        for j in range(0, ct.cdiv(N, BN)):
            if (j + 1) * BN <= bid_m * BM:
                k_tile = ct.load(
                    k,
                    index=(bid_bh, 0, j),
                    shape=(1, BD, BN),
                    order=(0, 2, 1),
                ).reshape((BD, BN))

                qk = ct.mma(
                    q_tile,
                    k_tile,
                    ct.full((BM, BN), 0.0, dtype=np.float32),
                )
                qk = qk * scale_log2

                m_ij = ct.maximum(m_i, ct.max(qk, axis=1, keepdims=True))
                p = ct.exp2(qk - m_ij, flush_to_zero=True)
                alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

                l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)
                acc = acc * alpha

                v_tile = ct.load(
                    v,
                    index=(bid_bh, j, 0),
                    shape=(1, BN, BD),
                ).reshape((BN, BD))

                acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
                m_i = m_ij
            elif j * BN <= bid_m * BM + (BM - 1):
                offs_n = j * BN + offs_n_base

                k_tile = ct.load(
                    k,
                    index=(bid_bh, 0, j),
                    shape=(1, BD, BN),
                    order=(0, 2, 1),
                ).reshape((BD, BN))

                qk = ct.mma(
                    q_tile,
                    k_tile,
                    ct.full((BM, BN), 0.0, dtype=np.float32),
                )
                qk = qk * scale_log2
                qk = ct.where(offs_m >= offs_n, qk, -np.inf)

                m_ij = ct.maximum(m_i, ct.max(qk, axis=1, keepdims=True))
                p = ct.exp2(qk - m_ij, flush_to_zero=True)
                alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

                l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)
                acc = acc * alpha

                v_tile = ct.load(
                    v,
                    index=(bid_bh, j, 0),
                    shape=(1, BN, BD),
                ).reshape((BN, BD))

                acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
                m_i = m_ij
    else:
        for j in range(0, ct.cdiv(N, BN)):
            k_tile = ct.load(
                k,
                index=(bid_bh, 0, j),
                shape=(1, BD, BN),
                order=(0, 2, 1),
            ).reshape((BD, BN))

            qk = ct.mma(
                q_tile,
                k_tile,
                ct.full((BM, BN), 0.0, dtype=np.float32),
            )
            qk = qk * scale_log2

            m_ij = ct.maximum(m_i, ct.max(qk, axis=1, keepdims=True))
            p = ct.exp2(qk - m_ij, flush_to_zero=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

            l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)
            acc = acc * alpha

            v_tile = ct.load(
                v,
                index=(bid_bh, j, 0),
                shape=(1, BN, BD),
            ).reshape((BN, BD))

            acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
            m_i = m_ij

    out_tile = ct.truediv(
        acc,
        l_i,
        flush_to_zero=True,
        rounding_mode=RMd.APPROX,
    )
    out_tile = ct.astype(out_tile, q.dtype)
    ct.store(
        out,
        index=(bid_bh, bid_m, 0),
        tile=ct.reshape(out_tile, (1, BM, BD)),
    )


_SEARCH_SPACE = [
    SimpleNamespace(tm=64, tn=128, occupancy=1),
    SimpleNamespace(tm=64, tn=128, occupancy=2),
    SimpleNamespace(tm=64, tn=128, occupancy=4),
    SimpleNamespace(tm=64, tn=256, occupancy=1),
    SimpleNamespace(tm=64, tn=256, occupancy=2),
    SimpleNamespace(tm=128, tn=64, occupancy=1),
    SimpleNamespace(tm=128, tn=64, occupancy=2),
    SimpleNamespace(tm=128, tn=64, occupancy=4),
    SimpleNamespace(tm=128, tn=128, occupancy=1),
    SimpleNamespace(tm=128, tn=128, occupancy=2),
    SimpleNamespace(tm=128, tn=128, occupancy=4),
]

_tuner = CutileAutotuner(_flash_attention_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    B, H, N, D = q.shape
    output = torch.empty_like(q)

    q3 = q.view(B * H, N, D)
    k3 = k.view(B * H, N, D)
    v3 = v.view(B * H, N, D)
    out3 = output.view(B * H, N, D)

    bd = max(32, _next_power_of_2(int(D)))
    scale_log2 = (1.0 / math.sqrt(float(D))) * 1.4426950408889634
    causal = bool(causal)

    stream = torch.cuda.current_stream()

    cfg = _tuner.tune_or_cached(
        shape_key=(B, H, N, D, causal, str(q.dtype)),
        search_space=_SEARCH_SPACE,
        stream=stream,
        grid_fn=lambda cfg: (ct.cdiv(N, cfg.tm), B * H, 1),
        args_fn=lambda cfg: (
            q3,
            k3,
            v3,
            out3,
            scale_log2,
            N,
            D,
            causal,
            cfg.tm,
            cfg.tn,
            bd,
        ),
        hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
    )

    _last_autotune_config.clear()
    _last_autotune_config.update(
        {
            "tm": cfg.tm,
            "tn": cfg.tn,
            "bd": bd,
            "occupancy": cfg.occupancy,
        }
    )

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream,
        (ct.cdiv(N, cfg.tm), B * H, 1),
        kernel,
        (
            q3,
            k3,
            v3,
            out3,
            scale_log2,
            N,
            D,
            causal,
            cfg.tm,
            cfg.tn,
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

# cuTile (cuda.tile) API Reference

The cuTile DSL is newer than Triton and likely sparse in your training data. Treat the API reference below as authoritative; do NOT invent attributes by analogy with Triton.

# cuTile (cuda.tile) Comprehensive Programming Guide

The `cuda.tile` Python DSL compiles Python into optimized CUDA via MLIR, targeting NVIDIA GPUs (Blackwell B200 / sm_100). All computation operates on immutable, multi-dimensional **Tiles** with power-of-2 dimensions.

---

## 1. Import Boilerplate

Every `impl_cutile.py` MUST follow this structure:

```python
from types import SimpleNamespace
import torch
import cuda.tile as ct
import numpy as np
from cuda.tile import RoundingMode as RMd

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_last_autotune_config: dict | None = None
```

- `ct_experimental` is optional; always guard with try/except
- `_last_autotune_config` is module-level state for the engine to read via `get_last_config()`
- Inside kernels: use `np.float32`, `np.int32`, `np.bool` etc. for dtype args, or `x.dtype` to reference a tensor's dtype

---

## 2. Data Types

### 2.1 DType Constants

| Integer Types | Float Types | Special Types |
|---------------|-------------|---------------|
| `ct.bool_` (8-bit) | `ct.float16` (IEEE 754 half) | `ct.tfloat32` (tensor float, 19-bit) |
| `ct.int8` / `ct.uint8` | `ct.float32` (IEEE 754 single) | `ct.float8_e4m3fn` (FP8, 1+4+3) |
| `ct.int16` / `ct.uint16` | `ct.float64` (IEEE 754 double) | `ct.float8_e5m2` (FP8, 1+5+2) |
| `ct.int32` / `ct.uint32` | `ct.bfloat16` (1+8+7 bits) | |
| `ct.int64` / `ct.uint64` | | |

DType properties: `.name` (str), `.bitwidth` (int).

Inside kernels, you can also use numpy dtypes: `np.float32`, `np.float16`, `np.int32`, etc.

### 2.2 DType Usage in Kernels

```python
# Use ct.* types or np.* types interchangeably in kernels
tile = ct.full((64,), 0.0, dtype=ct.float32)    # ct type
tile = ct.full((64,), 0.0, dtype=np.float32)     # numpy type
tile = ct.astype(x, ct.bfloat16)                 # explicit cast
tile = ct.astype(x, x_array.dtype)               # match input dtype
```

---

## 3. Core Classes

### 3.1 Array (Global Memory)

`Array` wraps a PyTorch tensor passed to the kernel. Available properties:

```python
array.dtype     # DType — element data type (compile-time constant)
array.shape     # tuple[int,...] — dimensions (compile-time per axis)
array.strides   # tuple[int,...] — strides per dimension
array.ndim      # int — number of dimensions (compile-time constant)
```

**Array.slice()** — create a sub-view:

```python
sub = array.slice(axis, start, stop)
# axis: const int (negative counts from last dim)
# start/stop: int or 0D tile (inclusive/exclusive)
# Returns: Array view (same memory, restricted range)
```

### 3.2 Tile (Local Data)

`Tile` is an immutable multi-dimensional collection local to a block. Every dimension must be a **power of 2**.

Properties:
```python
tile.dtype    # DType — element type
tile.shape    # tuple[const int,...] — shape
tile.ndim     # int — number of dimensions
```

Methods:
```python
tile.item()                         # extract scalar (0D Tile) from 1-element tile
tile.reshape(shape)                 # reshape
tile.permute(axes)                  # permute dimensions
tile.transpose(axis0=None, axis1=None)  # transpose two dims (auto for 2D)
tile.astype(dtype)                  # type cast
tile.extract(index, shape)          # extract sub-tile
tile[np.newaxis, :]                 # expand_dims via indexing
tile[:, None]                       # add dim via slicing
```

Operator overloads on Tile:
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Bitwise: `&`, `|`, `^`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Reverse ops: `radd`, `rsub`, `rmul`, etc. (for `scalar op tile`)

### 3.3 Scalar

`Scalar = int | float` — Python scalars auto-promote to tiles in binary operations.

---

## 4. Enumerations

### 4.1 PaddingMode

Controls out-of-bounds behavior for `ct.load()`:

| Value | Description | Use Case |
|-------|-------------|----------|
| `ct.PaddingMode.UNDETERMINED` | Undefined OOB (default) | When OOB never happens |
| `ct.PaddingMode.ZERO` | OOB = 0 | Sum, L2 norm, RMSNorm, LayerNorm |
| `ct.PaddingMode.NEG_ZERO` | OOB = -0.0 | Rare; IEEE semantics |
| `ct.PaddingMode.NAN` | OOB = NaN | Detection of OOB access |
| `ct.PaddingMode.POS_INF` | OOB = +inf | min reductions |
| `ct.PaddingMode.NEG_INF` | OOB = -inf | max, argmax, softmax |

### 4.2 RoundingMode

Controls float rounding for `ct.truediv`, `ct.sqrt`, `ct.sum`, `ct.add`, etc.:

| Value | Alias | Description |
|-------|-------|-------------|
| `RMd.RN` | `NEAREST_EVEN` | Round to nearest, ties to even (default) |
| `RMd.RZ` | `ZERO` | Round towards zero (truncate) |
| `RMd.RM` | `NEGATIVE_INF` | Round towards -inf |
| `RMd.RP` | `POSITIVE_INF` | Round towards +inf |
| `RMd.APPROX` | — | Approximate (faster, less accurate) |
| `RMd.FULL` | — | Full precision |

### 4.3 MemoryOrder (for atomics)

| Value | Description |
|-------|-------------|
| `ct.MemoryOrder.RELAXED` | No ordering guarantee |
| `ct.MemoryOrder.ACQUIRE` | Acquire semantics |
| `ct.MemoryOrder.RELEASE` | Release semantics |
| `ct.MemoryOrder.ACQ_REL` | Combined acquire+release (default for atomics) |

### 4.4 MemoryScope (for atomics)

| Value | Description |
|-------|-------------|
| `ct.MemoryScope.BLOCK` | Same CTA block |
| `ct.MemoryScope.DEVICE` | Same GPU (default for atomics) |
| `ct.MemoryScope.SYS` | Entire system (all GPUs + host) |

---

## 5. Kernel Definition & Launch

### 5.1 @ct.kernel Decorator

```python
@ct.kernel
def my_kernel(x, y, output, N: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    # ...

# With optimization hints:
@ct.kernel(occupancy=2, num_ctas=4, opt_level=3)
def my_kernel(x, y, output, N: ConstInt, TILE: ConstInt):
    # ...

# Target-specific hints:
from cuda.tile import ByTarget
@ct.kernel(num_ctas=ByTarget(sm_100=8, sm_120=4, default=2))
def my_kernel(x, y, output, N: ConstInt, TILE: ConstInt):
    # ...
```

Parameters:
- `num_ctas` — CTAs per CGA, power of 2 in [1, 16]. Default: auto.
- `occupancy` — Active CTAs per SM, [1, 32]. Default: auto.
- `opt_level` — Optimization level [0, 3]. Default: 3.

Kernel parameter rules:
- No annotation → runtime value (torch tensor, Python scalar like `float`, `int`)
- `ConstInt` / `ConstBool` → compile-time constant (embedded in binary, triggers recompilation on change)
- `ct.Constant[int]`, `ct.Constant[bool]`, `ct.Constant[float]` are all available

### 5.2 @ct.function Decorator

For helper functions callable from kernels:

```python
@ct.function
def helper(x, y):
    return x + y

@ct.function(host=True)       # callable from both host and tile code
def cdiv_helper(a, b):
    return (a + b - 1) // b
```

### 5.3 ct.launch()

```python
ct.launch(stream, grid, kernel, kernel_args)
```

- `stream` — CUDA stream (`torch.cuda.current_stream()`)
- `grid` — `(x,)`, `(x, y)`, or `(x, y, z)` tuple of block counts
- `kernel` — `@ct.kernel` decorated function
- `kernel_args` — tuple matching kernel signature order

### 5.4 Block/Grid Information

```python
ct.bid(axis)          # int32 — block ID along axis (0, 1, or 2)
ct.num_blocks(axis)   # int32 — total blocks along axis

ct.num_tiles(array, axis, shape, order="C")
# Number of tiles in the tile space of array along axis.
# shape: tile shape tuple.
# order: "C" (default), "F" (reversed), or tuple permutation.
```

---

## 6. Memory Operations

### 6.1 ct.load()

```python
tile = ct.load(array, index, shape, *,
               order="C",
               padding_mode=PaddingMode.UNDETERMINED,
               latency=None,
               allow_tma=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `array` | Array | Source array (torch tensor) |
| `index` | tuple[int\|Tile,...] | Tile-space coordinates |
| `shape` | tuple[const int,...] | Tile shape (all dims must be power of 2); `()` for 0D scalar tile |
| `order` | `"C"` / `"F"` / tuple | Axis mapping: `"C"`=identity, `"F"`=reversed, tuple=explicit permutation |
| `padding_mode` | PaddingMode | OOB element handling |
| `latency` | const int | DRAM traffic hint, 1 (low) to 10 (high) |
| `allow_tma` | const bool | Enable/disable TMA (default True) |

```python
# Examples
tile = ct.load(x, index=(bid,), shape=(TILE,))                           # 1D
tile = ct.load(x, index=(row, j), shape=(1, TILE), padding_mode=ct.PaddingMode.ZERO)  # 2D padded
tile = ct.load(Q, index=(b, h, i, 0), shape=(1,1,M,D))                  # 4D
tile = ct.load(K, index=(b, h, 0, j), shape=(1,1,D,N), order=(0,1,3,2)) # transposed load
scalar = ct.load(x, index=(i,), shape=())                                # scalar (0D tile)
```

### 6.2 ct.store()

```python
ct.store(array, index, tile, *,
         order="C",
         latency=None,
         allow_tma=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `array` | Array | Destination array |
| `index` | tuple[int\|Tile,...] | Tile-space coordinates |
| `tile` | Tile or scalar | Data to store |
| `order` | `"C"` / `"F"` / tuple | Axis mapping |
| `latency` | const int | Latency hint |
| `allow_tma` | const bool | Enable/disable TMA |

OOB writes are silently ignored (no-op).

**CRITICAL**: indices must be **static** (compile-time known or block-uniform). For runtime-computed destinations, use `ct.scatter()`.

### 6.3 ct.gather()

```python
result = ct.gather(array, indices, *,
                   padding_value=0,
                   check_bounds=True,
                   latency=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `array` | Array | Source array |
| `indices` | Tile or tuple[Tile\|int,...] | Per-element indices (length = array rank); single Tile for 1D arrays |
| `padding_value` | scalar | Value for OOB indices (default 0) |
| `check_bounds` | bool | Enable bounds checking (default True) |
| `latency` | int | Latency hint |

- All index tiles must be same shape or broadcastable. Result shape = broadcasted shape.
- Negative indices treated as OOB.

```python
# 1D gather
val = ct.gather(dest_loc, token_id)                     # scalar
# Multi-dim gather
vals = ct.gather(kv, (token_id, head_id, offsets))       # tile
# With custom padding
vals = ct.gather(x, (idx,), padding_value=-1)            # -1 for OOB
```

### 6.4 ct.scatter()

```python
ct.scatter(array, indices, value, *,
           check_bounds=True,
           latency=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `array` | Array | Destination array |
| `indices` | Tile or tuple[Tile\|int,...] | Per-element destination indices |
| `value` | Tile or scalar | Data to scatter (broadcastable to indices shape) |
| `check_bounds` | bool | Enable bounds checking (default True) |
| `latency` | int | Latency hint |

OOB writes are silently ignored.

```python
ct.scatter(out, (dest_index, head_id, offsets), kv_vals)
```

---

## 7. Atomic Operations

All atomics return the **old value** before the operation. All support these common keyword args:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `check_bounds` | bool | True | Bounds checking |
| `memory_order` | MemoryOrder | ACQ_REL | Ordering semantics |
| `memory_scope` | MemoryScope | DEVICE | Scope of ordering |

```python
ct.atomic_add(array, indices, update)     # array[idx] += update; return old
ct.atomic_max(array, indices, update)     # array[idx] = max(old, update); return old
ct.atomic_min(array, indices, update)     # array[idx] = min(old, update); return old
ct.atomic_and(array, indices, update)     # array[idx] &= update; return old
ct.atomic_or(array, indices, update)      # array[idx] |= update; return old
ct.atomic_xor(array, indices, update)     # array[idx] ^= update; return old
ct.atomic_xchg(array, indices, update)    # array[idx] = update; return old
ct.atomic_cas(array, indices, expected, desired)
    # if array[idx] == expected: array[idx] = desired; return old
```

OOB atomics: no-op, returns `expected` (for CAS) or `update`.

```python
# Example: histogram
old = ct.atomic_add(histogram, (bin_idx,), 1,
                    memory_order=ct.MemoryOrder.RELAXED,
                    memory_scope=ct.MemoryScope.DEVICE)
```

---

## 8. Tile Creation (Factory Functions)

```python
ct.full(shape, fill_value, dtype)       # filled tile
ct.zeros(shape, dtype)                   # all zeros
ct.ones(shape, dtype)                    # all ones
ct.arange(size, dtype=...)               # [0, 1, ..., size-1]
```

```python
# Examples
acc = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
m_i = ct.full((TILE_M, 1), -np.inf, dtype=np.float32)
zero = ct.zeros((TILE,), dtype=ct.float16)
one = ct.ones((64,), dtype=ct.int32)
idx = ct.arange(128, dtype=np.int32)

# Common indexing patterns
offs_m = ct.arange(TILE_M, dtype=np.int32)[:, None]   # [TILE_M, 1] column
offs_n = ct.arange(TILE_N, dtype=np.int32)[None, :]   # [1, TILE_N] row
```

---

## 9. Shape & View Operations

```python
ct.reshape(x, shape)              # reshape; one dim can be -1
ct.permute(x, axes)               # permute dimensions
ct.transpose(x, axis0=None, axis1=None)  # transpose two axes; auto for 2D
ct.expand_dims(x, axis)           # insert size-1 dimension
ct.broadcast_to(x, shape)         # broadcast (NumPy rules)
ct.cat((x, y), axis)              # concatenate two tiles along axis
ct.extract(x, index, shape)       # extract sub-tile from tile
ct.bitcast(x, dtype)              # reinterpret bits as different type (same bitwidth)
```

```python
# Examples
q = ct.load(Q, ..., shape=(1,1,M,D)).reshape((M, D))   # 4D -> 2D
acc = ct.reshape(acc, (1, 1, M, D))                      # 2D -> 4D
k_t = ct.transpose(k)                                    # transpose 2D tile
p = ct.permute(tile, (0, 2, 1))                          # reorder 3D axes
offs = ct.expand_dims(ct.arange(M), 1)                   # [M] -> [M, 1]
big = ct.cat((left, right), axis=1)                       # concat along cols
sub = ct.extract(tile, (0, 0), shape=(32, 32))            # top-left 32x32
i32_bits = ct.bitcast(float_tile, ct.int32)               # reinterpret float as int
```

---

## 10. Type Conversion

```python
ct.astype(x, dtype)               # cast tile to dtype
x.astype(dtype)                   # method form

# Inside kernels, numpy dtypes or ct dtypes both work:
y = ct.astype(x, np.float32)
y = ct.astype(x, ct.bfloat16)
y = x.astype(array.dtype)         # match input tensor's dtype
```

Common pattern — **compute in fp32, store in original dtype**:
```python
xj = ct.astype(ct.load(x, ...), np.float32)    # upcast for accumulation
yj = ct.astype(result, x.dtype)                 # downcast for storage
```

---

## 11. Arithmetic Operations

### 11.1 Operator Overloads (preferred for simple cases)

```python
c = a + b          c = a - b          c = a * b
c = a / b          c = a // b         c = a % b
c = a ** b
c = a & b          c = a | b          c = a ^ b
c = ~a                                c = -a
```

Python scalars auto-promote: `x * 2`, `x + 0.5`, `3 - x`.

### 11.2 Explicit Functions (when you need extra control)

All support `rounding_mode` and/or `flush_to_zero` where applicable:

```python
ct.add(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.sub(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.mul(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.truediv(x, y, *, rounding_mode=None, flush_to_zero=False)
ct.floordiv(x, y)
ct.mod(x, y)
ct.pow(x, y)
ct.negative(x)              # unary negate (same as -x)
ct.abs(x)                   # absolute value

ct.minimum(x, y, *, flush_to_zero=False)   # element-wise min
ct.maximum(x, y, *, flush_to_zero=False)   # element-wise max

ct.cdiv(x, y)     # ceiling division: ceil(x/y). Works on host AND in kernels.
```

```python
# Example: controlled division
acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
```

### 11.3 Bitwise Operations

```python
ct.bitwise_and(x, y)        # x & y
ct.bitwise_or(x, y)         # x | y
ct.bitwise_xor(x, y)        # x ^ y
ct.bitwise_not(x)           # ~x
ct.bitwise_lshift(x, n)     # x << n
ct.bitwise_rshift(x, n)     # x >> n
```

---

## 12. Comparison Operations

### 12.1 Operator Overloads (return bool tiles)

```python
mask = x > y       mask = x >= y      mask = x < y
mask = x <= y      mask = x == y      mask = x != y
```

### 12.2 Explicit Functions

```python
ct.greater(x, y)             # x > y
ct.greater_equal(x, y)       # x >= y
ct.less(x, y)                # x < y
ct.less_equal(x, y)          # x <= y
ct.equal(x, y)               # x == y
ct.not_equal(x, y)           # x != y
```

---

## 13. Math Functions

### 13.1 Exponential & Logarithmic

```python
ct.exp(x)           # e^x
ct.exp2(x, *, flush_to_zero=False)   # 2^x (faster; use for softmax with log2 scaling)
ct.log(x)           # ln(x)
ct.log2(x)          # log2(x)
ct.log10(x)         # log10(x)
ct.log1p(x)         # log(1 + x) — more accurate near x=0
```

### 13.2 Trigonometric

```python
ct.sin(x)           ct.cos(x)           ct.tan(x)
ct.sinh(x)          ct.cosh(x)          ct.tanh(x)
```

### 13.3 Power & Root

```python
ct.sqrt(x, *, rounding_mode=None, flush_to_zero=False)    # sqrt(x)
ct.rsqrt(x, *, flush_to_zero=False)                        # 1/sqrt(x)
ct.pow(x, y)                                                # x^y
```

### 13.4 Rounding

```python
ct.floor(x)         # round down
ct.ceil(x)          # round up
```

---

## 14. Reduction Operations

All reductions support:
- `axis=None` — reduce all elements to scalar
- `axis=int` — reduce along one axis
- `axis=tuple` — reduce along multiple axes
- `keepdims=False` — whether to preserve reduced dimensions

```python
ct.sum(x, axis=None, *, keepdims=False, rounding_mode=None, flush_to_zero=False)
ct.prod(x, axis=None, *, keepdims=False, rounding_mode=None, flush_to_zero=False)
ct.max(x, axis=None, *, keepdims=False, flush_to_zero=False)
ct.min(x, axis=None, *, keepdims=False, flush_to_zero=False)
ct.argmax(x, axis=None, *, keepdims=False)
ct.argmin(x, axis=None, *, keepdims=False)
```

```python
# Examples
total = ct.sum(tile)                                # scalar sum
row_sum = ct.sum(tile, axis=1, keepdims=False)      # sum along cols
max_val = ct.max(tile, axis=-1, keepdims=True)      # keep dim for broadcast
idx = ct.argmax(tile)                                # index of max element
```

### 14.1 Custom Reduction

```python
ct.reduce(x, axis, func, identity, *, keepdims=False)
```

- `x` — Tile or tuple of Tiles
- `func(a, b) -> combined` — reduction function (binary associative)
- `identity` — identity element (scalar or tuple of scalars)

```python
# Custom sum (equivalent to ct.sum)
result = ct.reduce(x, axis=0, func=lambda a, b: a + b, identity=0)

# Multi-output reduction: simultaneous min and argmin
min_val, min_idx = ct.reduce(
    (values, indices), axis=0,
    func=lambda v1, i1, v2, i2: (
        ct.where(v1 < v2, v1, v2),
        ct.where(v1 < v2, i1, i2)
    ),
    identity=(float('inf'), 0)
)
```

---

## 15. Scan (Prefix) Operations

```python
ct.cumsum(x, axis=0, *, reverse=False, rounding_mode=None, flush_to_zero=False)
ct.cumprod(x, axis=0, *, reverse=False, rounding_mode=None, flush_to_zero=False)
```

- `axis` — axis along which to scan (default 0)
- `reverse` — if True, scan from end to start

```python
# Prefix sum
prefix = ct.cumsum(tile, axis=1)           # [a, a+b, a+b+c, ...]
# Suffix product
suffix = ct.cumprod(tile, axis=0, reverse=True)
```

---

## 16. Conditional & Selection

```python
ct.where(condition, x, y)
# condition: bool Tile of shape S
# x, y: Tiles of shape S and same dtype (or scalars)
# Returns: element from x where True, y where False
```

```python
# ReLU
y = ct.where(x >= 0, x, 0.0)

# Causal mask for attention
mask = ct.where(offs_m >= offs_n, 0.0, -np.inf)
qk = qk + mask

# Clamp
y = ct.maximum(ct.minimum(x, upper), lower)
```

---

## 17. Matrix Multiply

### 17.1 ct.mma() — Matrix Multiply-Accumulate (Fused)

```python
result = ct.mma(x, y, acc)
# x: [M, K] or [B, M, K]
# y: [K, N] or [B, K, N]
# acc: [M, N] or [B, M, N]
# Returns: (x @ y) + acc with acc's dtype preserved
```

Supported input/accumulator dtype combinations:

| Input (x, y) | Accumulator (acc) |
|---------------|-------------------|
| float16 | float16 or float32 |
| bfloat16 | float32 |
| float32 | float32 |
| float64 | float64 |
| tfloat32 | float32 |
| float8_e4m3fn | float16 or float32 |
| float8_e5m2 | float16 or float32 |
| int8 / uint8 | int32 |

```python
# GEMM pattern
acc = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)
for k_block in range(num_k_tiles):
    a = ct.load(A, index=(bid_m, k_block), shape=(TILE_M, TILE_K))
    b = ct.load(B, index=(k_block, bid_n), shape=(TILE_K, TILE_N))
    acc = ct.mma(a, b, acc)
```

### 17.2 ct.matmul() — Simple Matrix Multiply

```python
result = ct.matmul(x, y)   # or x @ y
# Auto-promotes dtypes and infers result type
# Supports 1D, 2D, 3D (batch) inputs
```

Difference from `ct.mma()`: `matmul` doesn't require an accumulator, auto-promotes types, and returns a fresh result. `mma` is fused and preserves accumulator dtype — preferred for performance-critical paths.

---

## 18. Control Flow

### 18.1 Python for-loop (compile-time unrolled)

```python
num_tiles = ct.cdiv(N, TILE_SIZE)
for j in range(0, num_tiles):
    xj = ct.load(x, index=(row, j), shape=(1, TILE_SIZE), ...)
    # ... process tile
```

The loop is fully unrolled at compile time. `range()` bounds can use `ct.cdiv()` or any expression yielding a compile-time int.

### 18.2 Python if/else (compile-time)

```python
if CAUSAL:   # ConstBool — resolved at compile time, generates only one branch
    mask = offs_m >= offs_n
```

### 18.3 Runtime Conditionals

`ct.where()` for element-wise (see Section 16).

For block-level branching, use standard Python `if` on runtime scalars — but note this creates both branches in the compiled code and selects at runtime.

---

## 19. Synchronization (Advanced)

Token-based synchronization for explicit memory ordering (low-level API, rarely needed for typical kernels):

```python
from cuda_tile.dialects.cuda_tile_ops import (
    make_token, join_tokens,
    load_ptr_tko, store_ptr_tko,
    load_view_tko, store_view_tko,
)

token = make_token()
data, token = load_ptr_tko(result_type, ptr,
                            memory_ordering_semantics=MemoryOrderingSemantics.WEAK,
                            input_token=token, return_token=True)
token = store_ptr_tko(ptr, data, input_token=token)
combined = join_tokens(token1, token2)
```

These are the low-level MLIR bindings. The high-level `ct.load`/`ct.store` handle synchronization automatically in most cases.

---

## 20. TensorView & PartitionView (Advanced)

For complex memory access patterns beyond simple tile-space indexing:

```python
from cuda_tile.dialects.cuda_tile_ops import (
    make_tensor_view, make_partition_view,
    load_view_tko, store_view_tko,
    get_tensor_shape,
)

# Create tensor view with dynamic shape
tv = make_tensor_view(base_ptr, Float32,
                       shape=[N, M],       # int or scalar Tile for dynamic
                       strides=[M, 1])

# Partition into tiles
pv = make_partition_view(tv, tile_shape=[64, 64],
                          padding_value=PaddingValue.ZERO)

# Load/store via views
data = load_view_tko(pv, indices=[i, j], return_token=False)
store_view_tko(data, pv, indices=[i, j])
```

---

## 21. Debug & Utility

### 21.1 printf

```python
ct.printf(format_str, *tile_args)
# format_str: C-style printf format (specifiers: %d %u %x %f %e %g etc.)
# *tile_args: only Tile values (not Python scalars)
```

**Significant overhead** — use for debugging only. Multi-block outputs interleave. Use `opt_level=0` for serial output.

```python
ct.printf("bid=%d val=%f\n", ct.bid(0), tile)
ct.printf("x[0]=%f x[1]=%f\n", tile)  # prints per-element
```

### 21.2 assert_

```python
ct.assert_(cond_tile)
ct.assert_(cond_tile, "message")
# Asserts all elements of cond_tile are True. Significant overhead.
```

---

## 22. Compiler Hints (Advanced)

Optimization hints that help the compiler generate better code:

```python
# Hint: value is divisible by divisor
x = ct.assume_div_by(x, divisor=16)
x = ct.assume_div_by(x, divisor=4, every=2, along=0)  # every 2nd element along axis 0

# Hint: elements within groups are identical
x = ct.assume_same_elements(x, group_size=[1, 32])

# Hint: value is bounded
x = ct.assume_bounded(x, lb=0, ub=1023)

# Prevent compiler from optimizing across this point
x = ct.optimization_barrier(x)
```

---

## 23. Autotune Pattern

### 23.1 autotune_launch()

```python
result = ct_experimental.autotune_launch(
    stream,                                              # CUDA stream
    grid_fn=lambda cfg: (grid_x, grid_y, 1),            # cfg -> grid tuple
    kernel=my_kernel,                                     # @ct.kernel function
    args_fn=lambda cfg: (x, y, output, cfg.tile),        # cfg -> kernel args tuple
    hints_fn=lambda cfg: {"occupancy": cfg.occupancy},   # cfg -> hints dict
    search_space=_SEARCH_SPACE,                          # iterable of config objects
    # Optional:
    key=None,                   # hashable cache key (auto-generated from tensor shapes/dtypes)
    max_iter=60,                # max configs to try
    compiler_time_limit_sec=10, # timeout per config compilation
    seed=None,                  # random seed for sampling
    force_retune=False,         # ignore cache and retune
)
```

Returns `TunedResult`:
- `result.tuned_config` — winning `SimpleNamespace`
- `result.grid` — grid tuple used
- `result.cache_hit` — bool
- `result.tuning_record` — list of `(config, time_ms)` pairs

### 23.2 clear_autotune_cache()

```python
ct_experimental.clear_autotune_cache()                    # clear all
ct_experimental.clear_autotune_cache(kernel=my_kernel)    # clear for one kernel
ct_experimental.clear_autotune_cache(key=my_key)          # clear for one key
```

### 23.3 Standard Implementation Template

```python
_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=2)

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048, 4096, 8192]
    for occ in [1, 2, 4]
]

def run(x: torch.Tensor, autotune: bool = False) -> torch.Tensor:
    global _last_autotune_config
    output = torch.empty_like(x)
    n = x.numel()
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (ct.cdiv(n, cfg.tile), 1, 1),
            kernel=my_kernel,
            args_fn=lambda cfg: (x, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile": result.tuned_config.tile,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        ct.launch(stream, (ct.cdiv(n, cfg.tile), 1, 1),
                  my_kernel, (x, output, cfg.tile))
    return output

def get_last_config() -> dict | None:
    return _last_autotune_config
```

---

## 24. run() and get_last_config() Contract

Every `impl_cutile.py` must export:

```python
def run(*inputs, autotune=False, **kwargs) -> torch.Tensor:
    """Execute the operator."""
    ...

def get_last_config() -> dict | None:
    """Return last autotune config dict, or None."""
    return _last_autotune_config
```

- `_last_autotune_config` is module-level, written only in the autotune branch
- `autotune_launch()` has no kernel-level cache (unlike Triton); `_last_autotune_config` persists the result
- If an operator cannot be implemented: `run()` raises `NotImplementedError`, `get_last_config()` returns `None`

---

## 25. Common Kernel Patterns

### Pattern A: Element-wise (mul2, relu, vector_add, dropout, swiglu)

```python
@ct.kernel
def kernel(x, out, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,))
    y_tile = x_tile * 2  # or any element-wise op
    ct.store(out, index=(bid,), tile=y_tile)

# Grid: (cdiv(n_elements, tile), 1, 1)
```

Variants:
- **ReLU**: `ct.where(x >= 0, x, 0.0)`
- **SwiGLU**: `x * (1.0 / (1.0 + ct.exp(-gate)))`
- **Dropout**: `ct.load(mask, ...) * x * scale`

### Pattern B: Row-wise single-pass (softmax, argmax)

Load entire row (padded to power-of-2), one CTA per row:

```python
@ct.kernel
def kernel(x, out, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    tile = ct.load(x, index=(row, 0), shape=(1, TILE),
                   padding_mode=ct.PaddingMode.NEG_INF)
    max_val = ct.max(tile)
    exp_tile = ct.exp(tile - max_val)
    out_tile = exp_tile / ct.sum(exp_tile)
    ct.store(out, index=(row, 0), tile=out_tile)
```

Host: `TILE = 1 << (n_cols - 1).bit_length()`

### Pattern C: Row-wise two-pass (rmsnorm, layernorm, l2_norm)

Two loops over tiles when row doesn't fit in one tile:

```python
@ct.kernel
def kernel(x, w, out, eps, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    # Pass 1: accumulate
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(ct.load(x, index=(row, j), shape=(1, TILE),
                       allow_tma=False, latency=1,
                       padding_mode=ct.PaddingMode.ZERO), np.float32)
        acc = acc + xj * xj

    rstd = ct.rsqrt(ct.sum(acc, axis=1, keepdims=False) / N + eps)

    # Pass 2: normalize
    for j in range(0, num_tiles):
        xj = ct.astype(ct.load(x, ..., padding_mode=ct.PaddingMode.ZERO), np.float32)
        wj = ct.astype(ct.load(w, ..., padding_mode=ct.PaddingMode.ZERO), np.float32)
        ct.store(out, index=(row, j), tile=ct.astype(xj * rstd * wj, x.dtype))
```

### Pattern D: Flash Attention (online softmax + MMA)

```python
@ct.kernel(occupancy=2)
def fmha(Q, K, V, Out, qk_scale: float,
         D: ConstInt, H: ConstInt, M: ConstInt, N: ConstInt, CAUSAL: ConstBool):
    bid_x, bid_y = ct.bid(0), ct.bid(1)
    batch, head = bid_y // H, bid_y % H
    qk_scale *= 1.0 / math.log(2)  # log2 scale for exp2

    m_i = ct.full((M, 1), -np.inf, dtype=np.float32)    # running max
    l_i = ct.full((M, 1), 0.0, dtype=np.float32)        # running sum
    acc = ct.full((M, D), 0.0, dtype=np.float32)         # accumulator

    q = ct.load(Q, ...).reshape((M, D))
    for j in range(0, Tc):
        k = ct.load(K, ..., order=(0,1,3,2)).reshape((D, N))
        qk = ct.mma(q, k, ct.full((M, N), 0., dtype=np.float32))
        # Apply causal mask if needed
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        p = ct.exp2(qk * qk_scale - m_ij, flush_to_zero=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha
        v = ct.load(V, ...).reshape((N, D))
        acc = ct.mma(p.astype(Q.dtype), v, acc)
        m_i = m_ij
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
```

### Pattern E: Dynamic indexing (gather/scatter)

```python
@ct.kernel
def kernel(kv, dest_loc, out, DIM: ConstInt):
    token_id, head_id = ct.bid(0), ct.bid(1)
    dest = ct.gather(dest_loc, token_id)
    offsets = ct.arange(DIM, dtype=np.int32)
    vals = ct.gather(kv, (token_id, head_id, offsets))
    ct.scatter(out, (dest, head_id, offsets), vals)
```

### Pattern F: NotImplementedError

```python
def run(*args, **kwargs):
    raise NotImplementedError("cuTile does not support <reason>")

def get_last_config() -> dict | None:
    return None
```

---

## 26. Critical Constraints & Gotchas

### Tile dimensions must be powers of two
All dimensions in `ct.load/store` `shape` must be powers of 2. Handle non-pow2 problem sizes by:
- Round up TILE_SIZE: `block = 1 << (n - 1).bit_length()`
- Use `padding_mode` for OOB reads
- OOB stores are silently ignored

### ct.store() requires static indices
Use `ct.scatter()` for runtime-computed destination indices. Use `ct.gather()` for runtime-computed source indices.

### Tiles are immutable
```python
# WRONG: x_tile += 1
# RIGHT:
x_tile = x_tile + 1
```

### Always accumulate in fp32
```python
acc = ct.full(..., 0.0, dtype=np.float32)
xj = ct.astype(ct.load(...), np.float32)
result = ct.astype(acc, x.dtype)
```

### Grid tuple
`ct.launch()` accepts 1-, 2-, or 3-tuple. Convention: always use 3-tuple with trailing 1s:
```python
grid = (num_blocks, 1, 1)      # 1D
grid = (grid_x, grid_y, 1)     # 2D
```

### flush_to_zero for exp2
Use `flush_to_zero=True` with `ct.exp2()` in softmax to avoid denormals:
```python
p = ct.exp2(qk, flush_to_zero=True)
```

### Kernel vs host code
Inside `@ct.kernel`: Python `for`/`if` on compile-time values are unrolled/specialized. Array `.shape`, `.dtype` produce compile-time constants. All operations return tiles.

Outside kernel: standard Python/PyTorch. Use `ct.cdiv()` for ceiling division (it works both host and device side).

### Autotune has no persistent cache
`ct_experimental.autotune_launch()` caches in-memory per session (not on disk). `_last_autotune_config` module-level variable is the only way to expose configs to the benchmark engine.

### fp16 argmax precision
fp16 can only exactly represent integers up to 2048. For argmax on large dimensions, cast to fp32 first:
```python
x2d = x2d.float()  # on host, before passing to kernel
```

---

## 27. Quick Reference Tables

### ct.load() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | Array | required | Source tensor |
| `index` | tuple | required | Tile-space coordinates |
| `shape` | tuple[const int] | required | Tile shape (pow2 dims); `()` for scalar |
| `order` | `"C"`/`"F"`/tuple | `"C"` | Axis permutation |
| `padding_mode` | PaddingMode | UNDETERMINED | OOB behavior |
| `latency` | const int | None | DRAM hint 1-10 |
| `allow_tma` | const bool | True | TMA enable |

### ct.store() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `array` | Array | required | Destination tensor |
| `index` | tuple | required | Tile-space coordinates (static only!) |
| `tile` | Tile/scalar | required | Data to write |
| `order` | `"C"`/`"F"`/tuple | `"C"` | Axis permutation |
| `latency` | const int | None | DRAM hint 1-10 |
| `allow_tma` | const bool | True | TMA enable |

### MMA Supported Dtypes

| x, y dtype | acc dtype | Notes |
|------------|-----------|-------|
| float16 | float16, float32 | Most common |
| bfloat16 | float32 | Training |
| float32 | float32 | Uses TF32 cores |
| float64 | float64 | Double precision |
| tfloat32 | float32 | Explicit TF32 |
| float8_e4m3fn | float16, float32 | FP8 inference |
| float8_e5m2 | float16, float32 | FP8 |
| int8/uint8 | int32 | Quantized |


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
