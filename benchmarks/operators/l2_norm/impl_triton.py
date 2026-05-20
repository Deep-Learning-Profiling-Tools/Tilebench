import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

_DEFAULT_CONFIG = {"BLOCK_N": 1024, "num_warps": 4, "num_stages": 3}


@triton.jit
def _l2_norm_fwd_kernel(
    X,
    Y,
    stride_x_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    """One CTA normalises one row with tiled two-pass L2 norm."""
    row = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(
        X + row * stride_x_row,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK_N],
    )
    y_desc = tl.make_tensor_descriptor(
        Y + row * stride_x_row,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK_N],
    )

    # Pass 1: accumulate sum(x²).
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        x = x_desc.load([off]).to(tl.float32)
        x = tl.where(cols < N, x, 0.0)
        acc += x * x
    rstd = 1 / tl.sqrt(tl.sum(acc, axis=0) + eps)

    # Pass 2: normalise (re-load x).
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = x_desc.load([off]).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        y = x * rstd
        y_desc.store([off], y.to(X.dtype.element_ty))


_l2_norm_fwd_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)
        for bn in [512, 1024, 2048]
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
    ],
    key=["N"],
)(_l2_norm_fwd_kernel)


def run(x: torch.Tensor, eps: float = 1e-6, autotune: bool = False, **kwargs) -> torch.Tensor:
    ensure_tma_available()
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    y_2d = torch.empty_like(x_2d)
    N = x_2d.shape[-1]
    M = x_2d.shape[0]

    if autotune:
        with torch.cuda.device(x.device.index):
            _l2_norm_fwd_kernel_autotuned[(M,)](
                x_2d, y_2d,
                x_2d.stride(0),
                N, eps,
            )
    else:
        cfg = _DEFAULT_CONFIG
        with torch.cuda.device(x.device.index):
            _l2_norm_fwd_kernel[(M,)](
                x_2d, y_2d,
                x_2d.stride(0),
                N, eps,
                BLOCK_N=cfg["BLOCK_N"],
                num_warps=cfg["num_warps"],
                num_stages=cfg["num_stages"],
            )
    return y_2d.reshape(orig_shape)


def get_last_config() -> dict | None:
    cfg = getattr(_l2_norm_fwd_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_N": cfg.kwargs["BLOCK_N"], "num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
