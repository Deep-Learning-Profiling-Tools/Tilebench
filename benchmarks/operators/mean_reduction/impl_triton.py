import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

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

    x_desc = tl.make_tensor_descriptor(
        X,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    out_desc = tl.make_tensor_descriptor(
        Out,
        shape=[M, 1],
        strides=[1, 1],
        block_shape=[BLOCK_M, 1],
    )

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        cols     = off + tl.arange(0, BLOCK_N)[None, :] # [1, BLOCK_N]
        col_mask = cols < N
        mask     = row_mask[:, None] & col_mask

        a = x_desc.load([pid * BLOCK_M, off]).to(tl.float32)
        acc += tl.where(mask, a, 0.0)

    row_sum = tl.sum(acc, axis=1)                        # [BLOCK_M]
    mean    = row_sum / N                                # [BLOCK_M]
    out_desc.store([pid * BLOCK_M, 0], mean[:, None])


_mean_rowwise_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
        for bn in [512, 1024, 2048]
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
    ],
    key=["M", "N"],
)(_mean_rowwise_kernel)


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    ensure_tma_available()
    assert x.is_cuda

    if x.ndim == 2 and dim == 1:
        x2d = x.contiguous()
    else:
        dims = list(range(x.ndim))
        dims.remove(dim % x.ndim)
        dims.append(dim % x.ndim)
        x2d = x.permute(dims).contiguous().reshape(-1, x.shape[dim])

    M, N = x2d.shape
    out = torch.empty(M, dtype=torch.float32, device=x.device)

    if autotune:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        _mean_rowwise_kernel_autotuned[grid](x2d, out, M, N)
    else:
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
    cfg = getattr(_mean_rowwise_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_M": cfg.kwargs["BLOCK_M"],
        "BLOCK_N": cfg.kwargs["BLOCK_N"],
        "num_warps": cfg.num_warps,
    }
