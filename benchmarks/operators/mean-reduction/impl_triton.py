import torch
import triton
import triton.language as tl

# One CTA per row (BLOCK_M=1); column tile size drives memory coalescing.
_DEFAULT_CONFIG = {"BLOCK_M": 1, "BLOCK_N": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def _mean_rowwise_kernel(X, Out, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    X:   pointer to input  [M, N] (row-major)
    Out: pointer to output [M]
    Each CTA handles BLOCK_M consecutive rows.
    """
    pid = tl.program_id(0)
    row_ids  = pid * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    row_mask = row_ids < M

    X_row_ptr   = X   + row_ids[:, None] * N            # [BLOCK_M, 1] base ptrs
    Out_row_ptr = Out + row_ids                          # [BLOCK_M]

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols     = off + tl.arange(0, BLOCK_N)[None, :] # [1, BLOCK_N]
        col_mask = cols < N
        mask     = row_mask[:, None] & col_mask

        a = tl.load(X_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc += a

    row_sum = tl.sum(acc, axis=1)                        # [BLOCK_M]
    mean    = row_sum / N                                # [BLOCK_M]
    tl.store(Out_row_ptr, mean, mask=row_mask)


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    """
    Triton row-wise mean reduction.
    Input:  (M, N)  — any floating dtype
    Output: (M,) float32
    """
    assert x.is_cuda

    # Reshape: reduction dim → last, everything else → rows
    if x.ndim == 2 and dim == 1:
        x2d = x.float().contiguous()
    else:
        # Permute so that `dim` is last, treat remainder as M
        dims = list(range(x.ndim))
        dims.remove(dim % x.ndim)
        dims.append(dim % x.ndim)
        x2d = x.float().permute(dims).contiguous().reshape(-1, x.shape[dim])

    M, N = x2d.shape
    out  = torch.empty(M, dtype=torch.float32, device=x.device)

    cfg = _DEFAULT_CONFIG
    grid = (triton.cdiv(M, cfg["BLOCK_M"]),)
    _mean_rowwise_kernel[grid](
        x2d, out, M, N,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_N=cfg["BLOCK_N"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out


def get_last_config() -> dict | None:
    return None
