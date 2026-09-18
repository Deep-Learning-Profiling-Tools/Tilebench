import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _argmax_kernel(x_ptr, out_ptr, M, N, stride_m,
                   BLOCK_N: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    row_offs = pid * ROWS + tl.arange(0, ROWS)
    row_mask = row_offs < M

    NEG_INF: tl.constexpr = float('-inf')
    best_val = tl.full([ROWS], NEG_INF, dtype=tl.float32)
    best_idx = tl.full([ROWS], 0, dtype=tl.int64)

    for n_start in tl.range(0, N, BLOCK_N, num_stages=3):
        offs = n_start + tl.arange(0, BLOCK_N)
        col_mask = offs < N
        mask = row_mask[:, None] & col_mask[None, :]
        ptrs = x_ptr + row_offs[:, None] * stride_m + offs[None, :]
        x = tl.load(ptrs, mask=mask, other=NEG_INF).to(tl.float32)

        local_max = tl.max(x, axis=1)              # [ROWS]
        local_arg = tl.argmax(x, axis=1)           # [ROWS]
        local_global = local_arg.to(tl.int64) + n_start

        # Strict '>' so earliest occurrence wins on ties across tiles.
        new_better = local_max > best_val
        best_val = tl.where(new_better, local_max, best_val)
        best_idx = tl.where(new_better, local_global, best_idx)

    tl.store(out_ptr + row_offs, best_idx, mask=row_mask)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim in (1, -1)
    assert x.ndim == 2
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty(M, dtype=torch.int64, device=x.device)

    BLOCK_N = 4096
    ROWS = 2
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(M, ROWS),)
    _argmax_kernel[grid](
        x, output, M, N, x.stride(0),
        BLOCK_N=BLOCK_N, ROWS=ROWS,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N": BLOCK_N,
        "ROWS": ROWS,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
