import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _reverse_i32_kernel(
    input_ptr,
    output_ptr,
    n_words,
    BLOCK_SIZE: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_words

    rev_offsets = n_words - 1 - offsets
    safe_rev_offsets = tl.where(mask, rev_offsets, 0)

    vals = tl.load(input_ptr + safe_rev_offsets, eviction_policy="evict_first")

    if MODE == 1:
        # int8 packed path: reverse four bytes within each 32-bit word.
        b0 = (vals & 0x000000FF) << 24
        b1 = (vals & 0x0000FF00) << 8
        b2 = (vals >> 8) & 0x0000FF00
        b3 = (vals >> 24) & 0x000000FF
        vals = b0 | b1 | b2 | b3
    elif MODE == 2:
        # fp16/bf16 packed path: reverse two 16-bit lanes within each 32-bit word.
        lo = (vals & 0x0000FFFF) << 16
        hi = (vals >> 16) & 0x0000FFFF
        vals = lo | hi

    tl.store(output_ptr + offsets, vals, mask=mask)


@triton.jit
def _reverse_element_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    rev_offsets = n_elements - 1 - offsets
    safe_rev_offsets = tl.where(mask, rev_offsets, 0)

    vals = tl.load(input_ptr + safe_rev_offsets, eviction_policy="evict_first")
    tl.store(output_ptr + offsets, vals, mask=mask)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    n_elements = input.numel()

    BLOCK_SIZE = 2048
    num_warps = 4
    num_stages = 2

    elem_size = input.element_size()
    if elem_size == 1:
        pack = 4
        mode = 1
    elif elem_size == 2:
        pack = 2
        mode = 2
    else:
        pack = 1
        mode = 0

    if n_elements % pack == 0:
        input_i32 = input.view(torch.int32)
        output_i32 = output.view(torch.int32)
        n_words = n_elements // pack

        grid = (triton.cdiv(n_words, BLOCK_SIZE),)
        _reverse_i32_kernel[grid](
            input_i32,
            output_i32,
            n_words,
            BLOCK_SIZE=BLOCK_SIZE,
            MODE=mode,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "BLOCK_SIZE": BLOCK_SIZE,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "word_packed": True,
                "PACK": pack,
                "MODE": mode,
            }
        )
    else:
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _reverse_element_kernel[grid](
            input,
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "BLOCK_SIZE": BLOCK_SIZE,
                "num_warps": num_warps,
                "num_stages": num_stages,
                "word_packed": False,
            }
        )

    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
