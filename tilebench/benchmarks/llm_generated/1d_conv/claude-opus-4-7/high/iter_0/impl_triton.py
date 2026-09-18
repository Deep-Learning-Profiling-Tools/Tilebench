import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv1d_kernel(x_ptr, w_ptr, out_ptr, input_size, output_size,
                   KERNEL_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    out_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offs < output_size

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for j in tl.static_range(0, KERNEL_SIZE):
        x = tl.load(x_ptr + out_offs + j, mask=out_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + j).to(tl.float32)
        acc += x * w

    tl.store(out_ptr + out_offs,
             acc.to(out_ptr.dtype.element_ty),
             mask=out_mask)


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)

    BLOCK_SIZE = 1024
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(output_size, BLOCK_SIZE),)
    _conv1d_kernel[grid](
        input, kernel, output, input_size, output_size,
        KERNEL_SIZE=kernel_size, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
