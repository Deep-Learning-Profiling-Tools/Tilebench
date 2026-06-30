import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def conv1d_kernel(input_ptr, kernel_ptr, output_ptr, input_size, kernel_size: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    output_size = input_size - kernel_size + 1

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_size

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for j in tl.static_range(0, kernel_size):
        x = tl.load(input_ptr + offsets + j, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(kernel_ptr + j).to(tl.float32)
        acc += x * w

    tl.store(output_ptr + offsets, acc, mask=mask)


_conv1d_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [256, 512, 1024, 2048]
        for nw in [4, 8]
        for ns in [1, 2]
    ],
    key=["input_size", "kernel_size"],
    warmup=1,
    rep=3,
)(conv1d_kernel)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output_size = input_size - kernel_size + 1

    if output_size <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(output_size, dtype=input.dtype, device=input.device)

    if autotune:
        grid = lambda meta: (triton.cdiv(output_size, meta["BLOCK_SIZE"]),)
        _conv1d_kernel_autotuned[grid](
            input, kernel, output, input_size, kernel_size,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(output_size, cfg["BLOCK_SIZE"]),)
        conv1d_kernel[grid](
            input, kernel, output,
            input_size, kernel_size,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_conv1d_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
