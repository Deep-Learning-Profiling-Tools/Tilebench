import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _argmax_kernel(x_ptr, out_ptr, N, stride_m,
                   BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = x_ptr + pid * stride_m

    NEG_INF: tl.constexpr = float('-inf')

    best_val = tl.full([1], NEG_INF, dtype=tl.float32)
    best_idx = tl.full([1], 0, dtype=tl.int64)

    for n_start in tl.range(0, N, BLOCK_N, num_stages=3):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs, mask=mask, other=NEG_INF).to(tl.float32)
        local_max = tl.max(x, axis=0, keep_dims=True)
        local_arg = tl.argmax(x, axis=0, keep_dims=True)
        local_global = local_arg.to(tl.int64) + n_start

        # Strict '>' ensures earliest occurrence wins on ties across tiles.
        new_better = local_max > best_val
        best_val = tl.where(new_better, local_max, best_val)
        best_idx = tl.where(new_better, local_global, best_idx)

    tl.store(out_ptr + pid + tl.arange(0, 1), best_idx)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim in (1, -1)
    assert x.ndim == 2
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty(M, dtype=torch.int64, device=x.device)

    BLOCK_N = 4096
    num_warps = 8
    num_stages = 3

    grid = (M,)
    _argmax_kernel[grid](
        x, output, N, x.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N": BLOCK_N,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
