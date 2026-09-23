import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_N_SIZE": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def rmsnorm_kernel(
    x_ptr, rms_w_ptr, out_ptr,
    stride_row,
    N_SIZE,
    eps,
    BLOCK_N_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ptr     = x_ptr   + pid * stride_row
    out_row_ptr = out_ptr + pid * stride_row
    block_N = tl.arange(0, BLOCK_N_SIZE)


    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for n_start in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = n_start + block_N
        mask   = offs_n < N_SIZE
        x = tl.load(row_ptr + offs_n, mask=mask, other=0.0).to(tl.float32)
        var += x * x
    rstd = tl.math.rsqrt(tl.sum(var, axis=0) / N_SIZE + eps)


    for n_start in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = n_start + block_N
        mask   = offs_n < N_SIZE
        x     = tl.load(row_ptr     + offs_n, mask=mask, other=0.0).to(tl.float32)
        rms_w = tl.load(rms_w_ptr   + offs_n, mask=mask, other=1.0).to(tl.float32)
        tl.store(out_row_ptr + offs_n, x * rstd * rms_w, mask=mask)


_rmsnorm_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_N_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [512, 1024, 2048]
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
    ],
    key=["N_SIZE"],
)(rmsnorm_kernel)


def run(
    x: torch.Tensor,
    rms_w: torch.Tensor,
    eps: float = 1e-6,
    autotune: bool = False,
) -> torch.Tensor:
    orig_shape = x.shape
    K          = orig_shape[-1]
    batch_M    = x.numel() // K


    x_2d   = x.reshape(batch_M, K)
    out    = torch.empty_like(x)
    out_2d = out.reshape(batch_M, K)

    grid = (batch_M,)

    if autotune:
        _rmsnorm_kernel_autotuned[grid](
            x_2d, rms_w, out_2d,
            x_2d.stride(0),
            N_SIZE=K, eps=eps,
        )
    else:
        cfg = _DEFAULT_CONFIG
        rmsnorm_kernel[grid](
            x_2d, rms_w, out_2d,
            x_2d.stride(0),
            N_SIZE=K, eps=eps,
            BLOCK_N_SIZE=cfg["BLOCK_N_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return out


def get_last_config() -> dict | None:
    cfg = _rmsnorm_kernel_autotuned.best_config
    if cfg is None:
        return None
    return {
        "BLOCK_N_SIZE": cfg.kwargs["BLOCK_N_SIZE"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
