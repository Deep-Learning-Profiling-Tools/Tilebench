import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_N": 1024, "num_warps": 4, "num_stages": 3}


@triton.jit
def _l2_norm_fwd_kernel(
    X,
    Y,
    stride_x_row,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    """One CTA normalises one row with tiled two-pass L2 norm."""
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_x_row

    # Pass 1: accumulate sum(x²).
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x * x
    rstd = 1 / tl.sqrt(tl.sum(acc, axis=0) + eps)

    # Pass 2: normalise (re-load x).
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd
        tl.store(Y + cols, y.to(X.dtype.element_ty), mask=mask)


_l2_norm_fwd_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)
        for bn in [512, 1024, 2048]
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
    ],
    key=["N"],
    warmup=1,
    rep=3,
)(_l2_norm_fwd_kernel)


def run(x: torch.Tensor, eps: float = 1e-6, autotune: bool = False, **kwargs) -> torch.Tensor:
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
