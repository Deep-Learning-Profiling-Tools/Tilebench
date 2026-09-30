import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_TILE": 64, "num_warps": 4}


@triton.jit
def transpose_kernel(
    x_ptr,
    output_ptr,
    m,
    n,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_TILE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_TILE + tl.arange(0, BLOCK_TILE)
    offs_n = pid_n * BLOCK_TILE + tl.arange(0, BLOCK_TILE)
    mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    values = tl.load(x_ptrs, mask=mask)

    out_ptrs = output_ptr + offs_n[None, :] * stride_om + offs_m[:, None] * stride_on
    tl.store(out_ptrs, values, mask=mask)


_transpose_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_TILE": b}, num_warps=nw)
        for b in [32, 64, 128]
        for nw in [2, 4, 8]
    ],
    key=["m", "n"],
)(transpose_kernel)


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    m, n = x.shape
    output = torch.empty((n, m), device=x.device, dtype=x.dtype)

    if autotune:
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_TILE"]), triton.cdiv(n, meta["BLOCK_TILE"]))
        _transpose_kernel_autotuned[grid](
            x, output, m, n,
            x.stride(0), x.stride(1), output.stride(0), output.stride(1),
        )
    else:
        cfg = _DEFAULT_CONFIG
        tile = cfg["BLOCK_TILE"]
        grid = (triton.cdiv(m, tile), triton.cdiv(n, tile))
        transpose_kernel[grid](
            x, output, m, n,
            x.stride(0), x.stride(1), output.stride(0), output.stride(1),
            BLOCK_TILE=tile,
            num_warps=cfg["num_warps"],
        )
    return output


def get_last_config() -> dict | None:
    cfg = getattr(_transpose_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_TILE": cfg.kwargs["BLOCK_TILE"], "num_warps": cfg.num_warps}
