import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_R": 1,
    "BLOCK_SIZE_C": 1024,
    "num_warps": 4,
    "num_stages": 2,
}


@triton.jit
def _jacobi_stencil_kernel(
    input,
    output,
    rows,
    cols,
    stride_ir,
    stride_ic,
    stride_or,
    stride_oc,
    BLOCK_SIZE_R: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    ptrs_top = (input + (offs_r[:, None] - 1) * stride_ir +
                offs_c[None, :] * stride_ic)
    ptrs_bottom = (input + (offs_r[:, None] + 1) * stride_ir +
                   offs_c[None, :] * stride_ic)
    ptrs_left = (input + offs_r[:, None] * stride_ir +
                 (offs_c[None, :] - 1) * stride_ic)
    ptrs_right = (input + offs_r[:, None] * stride_ir +
                  (offs_c[None, :] + 1) * stride_ic)
    mask_edge = ((offs_r[:, None] - 1 >= 0) & (offs_r[:, None] < rows - 1) &
                 (offs_c[None, :] - 1 >= 0) & (offs_c[None, :] < cols - 1))
    top = tl.load(ptrs_top, mask=mask_edge)
    bottom = tl.load(ptrs_bottom, mask=mask_edge)
    left = tl.load(ptrs_left, mask=mask_edge)
    right = tl.load(ptrs_right, mask=mask_edge)

    ptrs_center = input + offs_r[:, None] * stride_ir + offs_c[None, :] * stride_ic
    mask_center = (offs_r[:, None] < rows) & (offs_c[None, :] < cols)
    center = tl.load(ptrs_center, mask=mask_center)

    avg = 0.25 * (top + bottom + left + right)
    tile_out = tl.where(mask_edge, avg, center)

    ptrs_out = (output + offs_r[:, None] * stride_or +
                offs_c[None, :] * stride_oc)
    tl.store(ptrs_out, tile_out, mask=mask_center)


_jacobi_stencil_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_R": br, "BLOCK_SIZE_C": bc},
            num_warps=nw, num_stages=ns,
        )
        for br in [1, 2, 4]
        for bc in [256, 512, 1024, 2048]
        for nw in [4, 8]
        for ns in [2, 3]
    ],
    key=["rows", "cols"],
)(_jacobi_stencil_kernel)


def run(input: torch.Tensor, rows: int, cols: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty_like(input)

    if autotune:
        grid = lambda meta: (
            triton.cdiv(rows, meta["BLOCK_SIZE_R"]),
            triton.cdiv(cols, meta["BLOCK_SIZE_C"]),
        )
        _jacobi_stencil_kernel_autotuned[grid](
            input, output, rows, cols,
            input.stride(0), input.stride(1),
            output.stride(0), output.stride(1),
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (
            triton.cdiv(rows, cfg["BLOCK_SIZE_R"]),
            triton.cdiv(cols, cfg["BLOCK_SIZE_C"]),
        )
        _jacobi_stencil_kernel[grid](
            input, output, rows, cols,
            input.stride(0), input.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_SIZE_R=cfg["BLOCK_SIZE_R"],
            BLOCK_SIZE_C=cfg["BLOCK_SIZE_C"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_jacobi_stencil_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE_R": cfg.kwargs["BLOCK_SIZE_R"],
        "BLOCK_SIZE_C": cfg.kwargs["BLOCK_SIZE_C"],
        "num_warps":    cfg.num_warps,
        "num_stages":   cfg.num_stages,
    }
