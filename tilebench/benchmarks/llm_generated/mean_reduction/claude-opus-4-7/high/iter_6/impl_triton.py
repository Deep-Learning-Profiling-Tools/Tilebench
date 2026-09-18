import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _mean_kernel(x_ptr, out_ptr, M, N, stride_m, stride_n,
                 BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    row_ptr = x_ptr + row * stride_m
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n_start in tl.range(0, N, BLOCK_N, num_stages=4):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs * stride_n, mask=mask, other=0.0,
                    cache_modifier=".cg",
                    eviction_policy="evict_first").to(tl.float32)
        acc += x
    s = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, s / N)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim == 1
    M, N = x.shape
    output = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_N = 4096
    num_warps = 4
    num_stages = 4

    grid = (M,)
    _mean_kernel[grid](
        x, output, M, N, x.stride(0), x.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N, "num_warps": num_warps,
                      "num_stages": num_stages,
                      "cache_modifier": "cg", "eviction": "evict_first"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
