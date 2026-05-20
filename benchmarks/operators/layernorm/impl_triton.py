import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

_DEFAULT_CONFIG = {"BLOCK_N_SIZE": 1024, "num_warps": 8, "num_stages": 2}


@triton.jit
def _layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    stride_row,
    N_SIZE,
    eps,
    BLOCK_N_SIZE: tl.constexpr,
):
    """One CTA normalises one row using two-pass LayerNorm.

    Pass 1 – single tiled scan accumulates sum(x) and sum(x²) simultaneously,
             then derives mean and rstd.
    Pass 2 – tiled scan: y = (x - mean) * rstd * weight + bias.

    All accumulators use float32 for numerical stability.
    """
    pid = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(
        x_ptr + pid * stride_row,
        shape=[N_SIZE],
        strides=[1],
        block_shape=[BLOCK_N_SIZE],
    )
    weight_desc = tl.make_tensor_descriptor(
        weight_ptr,
        shape=[N_SIZE],
        strides=[1],
        block_shape=[BLOCK_N_SIZE],
    )
    bias_desc = tl.make_tensor_descriptor(
        bias_ptr,
        shape=[N_SIZE],
        strides=[1],
        block_shape=[BLOCK_N_SIZE],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr + pid * stride_row,
        shape=[N_SIZE],
        strides=[1],
        block_shape=[BLOCK_N_SIZE],
    )
    block_N = tl.arange(0, BLOCK_N_SIZE)

    # --- Pass 1: compute mean and variance ---
    sum_x  = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    sum_x2 = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for n_start in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = n_start + block_N
        mask   = offs_n < N_SIZE
        x = x_desc.load([n_start]).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        sum_x  += x
        sum_x2 += x * x

    mean_val = tl.sum(sum_x,  axis=0) / N_SIZE
    # var = E[x^2] - E[x]^2; OOB elements loaded as 0 contribute 0 to both sums.
    var_val  = tl.sum(sum_x2, axis=0) / N_SIZE - mean_val * mean_val
    rstd     = tl.math.rsqrt(var_val + eps)

    # --- Pass 2: normalize and apply weight / bias ---
    for n_start in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = n_start + block_N
        mask   = offs_n < N_SIZE
        x = x_desc.load([n_start]).to(tl.float32)
        x = tl.where(mask, x, 0.0)
        weight = weight_desc.load([n_start]).to(tl.float32)
        weight = tl.where(mask, weight, 1.0)
        bias = bias_desc.load([n_start]).to(tl.float32)
        bias = tl.where(mask, bias, 0.0)
        y = (x - mean_val) * rstd * weight + bias
        out_desc.store([n_start], y)


_layernorm_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_N_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [512, 1024, 2048]
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
    ],
    key=["N_SIZE"],
)(_layernorm_kernel)


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    autotune: bool = False,
) -> torch.Tensor:
    ensure_tma_available()
    orig_shape = x.shape
    K          = orig_shape[-1]
    batch_M    = x.numel() // K

    x_2d   = x.contiguous().reshape(batch_M, K)
    out    = torch.empty_like(x)
    out_2d = out.reshape(batch_M, K)

    grid = (batch_M,)

    if autotune:
        _layernorm_kernel_autotuned[grid](
            x_2d, weight, bias, out_2d,
            x_2d.stride(0),
            N_SIZE=K, eps=eps,
        )
    else:
        cfg = _DEFAULT_CONFIG
        _layernorm_kernel[grid](
            x_2d, weight, bias, out_2d,
            x_2d.stride(0),
            N_SIZE=K, eps=eps,
            BLOCK_N_SIZE=cfg["BLOCK_N_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return out


def get_last_config() -> dict | None:
    cfg = getattr(_layernorm_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_N_SIZE": cfg.kwargs["BLOCK_N_SIZE"],
        "num_warps":    cfg.num_warps,
        "num_stages":   cfg.num_stages,
    }
