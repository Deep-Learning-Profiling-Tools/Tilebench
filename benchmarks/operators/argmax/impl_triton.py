import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_N": 128, "num_warps": 4, "num_stages": 1}


@triton.jit
def _argmax_rowwise_kernel(X, Out, N, BLOCK_N: tl.constexpr):
    """
    X:   pointer to input, logically [M, N]
    Out: pointer to output, shape [M], int64
    N:   number of columns (reduction dimension)
    Each program handles one row.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    neg_inf = float("-inf")

    best_val = neg_inf
    best_idx = tl.full((), 0, dtype=tl.int64)

    for start in range(0, N, BLOCK_N):
        cols = start + offs
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=neg_inf).to(tl.float32)

        tile_max = tl.max(x, axis=0)
        tile_arg = tl.argmax(x, axis=0).to(tl.int64)

        better = tile_max > best_val
        best_val = tl.where(better, tile_max, best_val)
        best_idx = tl.where(better, start + tile_arg, best_idx)

    tl.store(Out + row, best_idx)


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    """
    Triton row-wise argmax.
    Input:  (M, N)
    Output: (M,) int64
    """
    assert x.is_cuda, "Input must be on CUDA"

    if dim == 1:
        x2d = x.contiguous()
    else:
        x2d = x.transpose(0, 1).contiguous()

    M, N = x2d.shape
    out = torch.empty(M, dtype=torch.int64, device=x.device)

    BLOCK_N = _DEFAULT_CONFIG["BLOCK_N"]
    num_warps = _DEFAULT_CONFIG["num_warps"]
    num_stages = _DEFAULT_CONFIG["num_stages"]

    _argmax_rowwise_kernel[(M,)](
        x2d, out, N,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def get_last_config() -> dict | None:
    # BLOCK_N is fixed; no autotuner state to report.
    return None
