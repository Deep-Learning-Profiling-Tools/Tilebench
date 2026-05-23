import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _reverse_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    k = tl.arange(0, BLOCK_SIZE)
    # Output positions (ascending, contiguous)
    dst_offs = pid * BLOCK_SIZE + k
    dst_mask = dst_offs < N
    # Source positions: we want output[dst_offs[i]] = input[N-1-dst_offs[i]]
    # If we load source ASCENDING starting at (N - (pid+1)*BLOCK_SIZE) and then
    # flip the loaded tile, we get the correct values stored ascending.
    src_start = N - (pid + 1) * BLOCK_SIZE
    src_offs = src_start + k
    src_mask = src_offs >= 0  # only the LAST tile may have negative offsets
    x = tl.load(in_ptr + src_offs, mask=src_mask, other=0)
    x = tl.flip(x, 0)
    tl.store(out_ptr + dst_offs, x, mask=dst_mask)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)

    BLOCK_SIZE = 8192
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _reverse_kernel[grid](
        input, output, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
