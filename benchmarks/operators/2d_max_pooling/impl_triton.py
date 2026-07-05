import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 256, "num_warps": 4}


@triton.jit
def max_pool2d_kernel(
    input_ptr,
    output_ptr,
    C,
    H,
    W,
    H_out,
    W_out,
    total_out,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out

    # Decompose flat output offset into (n, c, oh, ow)
    ow = offsets % W_out
    oh = (offsets // W_out) % H_out
    c = (offsets // (H_out * W_out)) % C
    n = offsets // (C * H_out * W_out)

    acc = tl.full((BLOCK_SIZE,), -float("inf"), tl.float32)

    for kh in tl.static_range(0, kernel_size):
        for kw in tl.static_range(0, kernel_size):
            ih = oh * stride + kh - padding
            iw = ow * stride + kw - padding

            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            input_idx = ((n * C + c) * H + ih) * W + iw

            x = tl.load(input_ptr + input_idx, mask=valid, other=-float("inf")).to(tl.float32)
            acc = tl.maximum(acc, x)

    tl.store(output_ptr + offsets, acc, mask=mask)


_max_pool2d_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048]
        for nw in [4, 8]
    ],
    key=["total_out", "kernel_size", "stride", "padding"],
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
        grid = lambda meta: (triton.cdiv(total_out, meta["BLOCK_SIZE"]),)
        _max_pool2d_kernel_autotuned[grid](
            input, output,
            C, H, W, H_out, W_out, total_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(total_out, cfg["BLOCK_SIZE"]),)
        max_pool2d_kernel[grid](
            input, output,
            C, H, W, H_out, W_out, total_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_max_pool2d_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps": cfg.num_warps,
    }
