import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _mean_kernel(x_ptr, out_ptr, N, stride_m, stride_n,
                 BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cid = tl.program_id(1)
    offs = cid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_m + offs * stride_n,
                mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr + row, s / N)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim == 1
    M, N = x.shape
    output = torch.zeros((M,), dtype=torch.float32, device=x.device)

    BLOCK_N = 2048
    num_warps = 4
    num_stages = 2

    grid = (M, triton.cdiv(N, BLOCK_N))
    _mean_kernel[grid](
        x, output, N, x.stride(0), x.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N, "num_warps": num_warps,
                      "num_stages": num_stages, "strategy": "split-k-atomic"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
