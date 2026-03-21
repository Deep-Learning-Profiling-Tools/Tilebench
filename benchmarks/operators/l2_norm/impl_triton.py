import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"num_warps": 8, "num_stages": 2}


@triton.jit
def _l2_norm_fwd_kernel(
    X,
    Y,
    stride_x_row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_x_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    y = x * rstd
    tl.store(Y + cols, y.to(X.dtype.element_ty), mask=mask)


_l2_norm_fwd_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [4, 8, 16]
        for ns in [1, 2, 4]
    ],
    key=["N"],
)(_l2_norm_fwd_kernel)


def run(x: torch.Tensor, eps: float = 1e-6, autotune: bool = False, **kwargs) -> torch.Tensor:
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()
    y_2d = torch.empty_like(x_2d)
    N = x_2d.shape[-1]
    M = x_2d.shape[0]

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("l2_norm: feature dim >= 64KB is not supported.")

    if autotune:
        with torch.cuda.device(x.device.index):
            _l2_norm_fwd_kernel_autotuned[(M,)](
                x_2d, y_2d,
                x_2d.stride(0),
                N, eps,
                BLOCK_N=BLOCK_N,
            )
    else:
        cfg = _DEFAULT_CONFIG
        with torch.cuda.device(x.device.index):
            _l2_norm_fwd_kernel[(M,)](
                x_2d, y_2d,
                x_2d.stride(0),
                N, eps,
                BLOCK_N=BLOCK_N,
                num_warps=cfg["num_warps"],
                num_stages=cfg["num_stages"],
            )
    return y_2d.reshape(orig_shape)


def get_last_config() -> dict | None:
    cfg = getattr(_l2_norm_fwd_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
