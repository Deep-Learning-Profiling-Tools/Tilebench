import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_N": 256, "num_warps": 4, "num_stages": 2}


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


_argmax_rowwise_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)
        for bn in [256, 512, 1024, 2048]
        for nw in [4, 8, 16]
        for ns in [2, 3, 4]
    ],
    key=["N"],
)(_argmax_rowwise_kernel)


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    assert x.is_cuda, "Input must be on CUDA"

    if dim == 1:
        x2d = x.contiguous()
    else:
        x2d = x.transpose(0, 1).contiguous()

    M, N = x2d.shape
    out = torch.empty(M, dtype=torch.int64, device=x.device)

    if autotune:
        _argmax_rowwise_kernel_autotuned[(M,)](x2d, out, N)
    else:
        cfg = _DEFAULT_CONFIG
        _argmax_rowwise_kernel[(M,)](
            x2d, out, N,
            BLOCK_N=cfg["BLOCK_N"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return out


def get_last_config() -> dict | None:
    cfg = getattr(_argmax_rowwise_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_N":    cfg.kwargs["BLOCK_N"],
        "num_warps":  cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
