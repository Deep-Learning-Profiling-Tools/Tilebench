import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _jacobi_kernel(input_ptr, output_ptr, rows, cols,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    r = offs_m[:, None]
    c = offs_n[None, :]
    base = r * cols + c

    valid = (r < rows) & (c < cols)
    interior = valid & (r > 0) & (r < rows - 1) & (c > 0) & (c < cols - 1)
    boundary = valid & ((r == 0) | (r == rows - 1) | (c == 0) | (c == cols - 1))

    acc = tl.load(input_ptr + (base - cols), mask=interior, other=0.0).to(tl.float32)
    acc += tl.load(input_ptr + (base + cols), mask=interior, other=0.0).to(tl.float32)
    acc += tl.load(input_ptr + (base - 1), mask=interior, other=0.0).to(tl.float32)
    acc += tl.load(input_ptr + (base + 1), mask=interior, other=0.0).to(tl.float32)
    avg = acc * 0.25

    center = tl.load(input_ptr + base, mask=boundary, other=0.0).to(tl.float32)
    out = tl.where(interior, avg, center)

    tl.store(output_ptr + base, out, mask=valid)


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)

    BLOCK_M = 4
    BLOCK_N = 512
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(rows, BLOCK_M), triton.cdiv(cols, BLOCK_N))
    _jacobi_kernel[grid](
        input, output, rows, cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
