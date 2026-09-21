import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_R": 2, "BLOCK_C": 128, "num_warps": 4}


@triton.jit
def max_pool2d_kernel(
    input_ptr,
    output_ptr,
    H,
    W,
    H_out,
    W_out,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    plane = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_c = tl.program_id(2)

    oh = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    ow = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    in_base = input_ptr + plane * H * W
    acc = tl.full((BLOCK_R, BLOCK_C), -float("inf"),
                  input_ptr.dtype.element_ty)

    for kh in tl.static_range(0, kernel_size):
        for kw in tl.static_range(0, kernel_size):
            ih = oh * stride + kh - padding
            iw = ow * stride + kw - padding
            valid = ((ih[:, None] >= 0) & (ih[:, None] < H)
                     & (iw[None, :] >= 0) & (iw[None, :] < W))
            x = tl.load(in_base + ih[:, None] * W + iw[None, :],
                        mask=valid, other=-float("inf"))
            acc = tl.maximum(acc, x)

    out_mask = (oh[:, None] < H_out) & (ow[None, :] < W_out)
    out_base = output_ptr + plane * H_out * W_out
    tl.store(out_base + oh[:, None] * W_out + ow[None, :], acc,
             mask=out_mask)


_max_pool2d_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": br, "BLOCK_C": bc}, num_warps=nw)
        for br, bc in [(1, 128), (1, 256), (1, 512), (2, 128), (2, 256),
                       (4, 128), (4, 256), (8, 64)]
        for nw in [4, 8]
    ],
    key=["H_out", "W_out", "kernel_size", "stride", "padding"],
)(max_pool2d_kernel)


def run(input, N, C, H, W, kernel_size, stride, padding,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    total_out = N * C * H_out * W_out

    if total_out <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_out, dtype=input.dtype, device=input.device)

    if autotune:
        grid = lambda meta: (
            N * C,
            triton.cdiv(H_out, meta["BLOCK_R"]),
            triton.cdiv(W_out, meta["BLOCK_C"]),
        )
        _max_pool2d_kernel_autotuned[grid](
            input, output,
            H, W, H_out, W_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (
            N * C,
            triton.cdiv(H_out, cfg["BLOCK_R"]),
            triton.cdiv(W_out, cfg["BLOCK_C"]),
        )
        max_pool2d_kernel[grid](
            input, output,
            H, W, H_out, W_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            BLOCK_R=cfg["BLOCK_R"],
            BLOCK_C=cfg["BLOCK_C"],
            num_warps=cfg["num_warps"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_max_pool2d_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_R": cfg.kwargs["BLOCK_R"],
        "BLOCK_C": cfg.kwargs["BLOCK_C"],
        "num_warps": cfg.num_warps,
    }
