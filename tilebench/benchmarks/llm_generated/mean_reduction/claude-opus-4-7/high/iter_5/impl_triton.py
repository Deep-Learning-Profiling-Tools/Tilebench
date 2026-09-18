import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _mean_kernel(x_ptr, out_ptr, M, N, stride_m, stride_n,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    inv_n = 1.0 / N

    for n_start in tl.range(0, N, BLOCK_N, num_stages=3):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        mask = m_mask[:, None] & n_mask[None, :]
        ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
        x = tl.load(ptrs, mask=mask, other=0.0,
                    eviction_policy="evict_first").to(tl.float32)
        acc += tl.sum(x, axis=1)

    tl.store(out_ptr + offs_m, acc * inv_n, mask=m_mask)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim == 1
    M, N = x.shape
    output = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_M = 4
    BLOCK_N = 2048
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M),)
    _mean_kernel[grid](
        x, output, M, N, x.stride(0), x.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
                      "num_warps": num_warps, "num_stages": num_stages,
                      "strategy": "multi-row-2d"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
