import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK": 1024, "num_warps": 4}


@triton.jit
def pad_kernel(data_ptr, work_ptr, N, M, BLOCK: tl.constexpr):
    """Copy data[0:N] → work[0:N], fill work[N:M] with +inf."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    vals = tl.load(data_ptr + offs, mask=offs < N, other=float("inf"))
    tl.store(work_ptr + offs, vals, mask=offs < M)


@triton.jit
def bitonic_step_kernel(work_ptr, k, j, M, BLOCK: tl.constexpr):
    """
    One compare-exchange pass of bitonic sort.
    Each thread handles a pair (offs, ixj=offs^j); the ixj > offs guard
    ensures each pair is processed by exactly one thread.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    ixj = offs ^ j

    active = (ixj > offs) & (ixj < M) & (offs < M)
    a = tl.load(work_ptr + offs, mask=active, other=0.0)
    b = tl.load(work_ptr + ixj, mask=active, other=0.0)
    ascending = (offs & k) == 0
    swap = tl.where(ascending, a > b, a < b)
    new_a = tl.where(swap, b, a)
    new_b = tl.where(swap, a, b)
    tl.store(work_ptr + offs, new_a, mask=active)
    tl.store(work_ptr + ixj, new_b, mask=active)


_bitonic_step_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK": bs}, num_warps=nw)
        for bs in [512, 1024, 2048, 4096]
        for nw in [2, 4, 8]
    ],
    key=["M"],
)(bitonic_step_kernel)


def run(data: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    Bitonic sort (multi-launch), matching the LeetGPU algorithm:
      1. Pad input to M = next_pow2(N) with +inf.
      2. Host loop over (k, j); each step launches one compare-exchange pass.
      3. Return the first N elements of the sorted work buffer.
    """
    if N <= 1:
        return data.clone()

    M = triton.next_power_of_2(N)
    work = torch.empty((M,), device=data.device, dtype=data.dtype)
    cfg = _DEFAULT_CONFIG

    # Pad phase — always uses the default config (cheap, single launch).
    grid_pad = (triton.cdiv(M, cfg["BLOCK"]),)
    pad_kernel[grid_pad](data, work, N, M, BLOCK=cfg["BLOCK"])

    # Bitonic sort phase — triton.autotune caches per `key=["M"]`, so the first
    # step autotunes and the remaining log²(M)/2 steps are cache hits.
    if autotune:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK"]),)
        k = 2
        while k <= M:
            j = k // 2
            while j > 0:
                _bitonic_step_kernel_autotuned[grid](work, k, j, M)
                j //= 2
            k *= 2
    else:
        grid = (triton.cdiv(M, cfg["BLOCK"]),)
        k = 2
        while k <= M:
            j = k // 2
            while j > 0:
                bitonic_step_kernel[grid](
                    work, k, j, M,
                    BLOCK=cfg["BLOCK"],
                    num_warps=cfg["num_warps"],
                )
                j //= 2
            k *= 2

    return work[:N].contiguous()


def get_last_config() -> dict | None:
    cfg = getattr(_bitonic_step_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK": cfg.kwargs["BLOCK"],
        "num_warps": cfg.num_warps,
    }
