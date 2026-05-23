```python title="impl_triton.py"
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

    vals = tl.load(
        input_ptr + safe_rev_offsets,
        mask=mask,
        other=0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    if MODE == 1:
        # int8 path: reverse four bytes within each 32-bit word.
        u = vals.to(tl.uint32, bitcast=True)
        y = (u << 24) | ((u & 0x0000FF00) << 8) | ((u >> 8) & 0x0000FF00) | (u >> 24)
        vals = y.to(tl.int32, bitcast=True)
    elif MODE == 2:
        # fp16/bf16 path: reverse two 16-bit lanes within each 32-bit word.
        u = vals.to(tl.uint32, bitcast=True)
        y = (u << 16) | (u >> 16)
        vals = y.to(tl.int32, bitcast=True)

    tl.store(
        output_ptr + offsets,
        vals,
        mask=mask,
        cache_modifier=".cs",
    )


@triton.jit
def _reverse_element_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    rev_offsets = n_elements - 1 - offsets
    safe_rev_offsets = tl.where(mask, rev_offsets, 0)

    vals = tl.load(
        input_ptr + safe_rev_offsets,
        mask=mask,
        other=0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    tl.store(
        output_ptr + offsets,
        vals,
        mask=mask,
        cache_modifier=".cs",
    )


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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _reverse_i32_kernel(input, output, N_WORDS, MODE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)

    offsets = bid * TILE + ct.arange(TILE, dtype=np.int32)
    valid = offsets < N_WORDS

    src_offsets = N_WORDS - 1 - offsets
    safe_src_offsets = ct.where(valid, src_offsets, 0)

    vals = ct.gather(input, safe_src_offsets, check_bounds=False, latency=1)

    if MODE == 1:
        # int8 path: reverse four bytes within each 32-bit word.
        u = ct.bitcast(vals, ct.uint32)
        y = (u << 24) | ((u & 0x0000FF00) << 8) | ((u >> 8) & 0x0000FF00) | (u >> 24)
        vals = ct.bitcast(y, ct.int32)
    elif MODE == 2:
        # fp16/bf16 path: reverse two 16-bit lanes within each 32-bit word.
        u = ct.bitcast(vals, ct.uint32)
        y = (u << 16) | (u >> 16)
        vals = ct.bitcast(y, ct.int32)

    ct.store(output, index=(bid,), tile=vals, latency=1)


@ct.kernel
def _reverse_element_kernel(input, output, N_ELEMENTS, TILE: ConstInt):
    bid = ct.bid(0)

    offsets = bid * TILE + ct.arange(TILE, dtype=np.int32)
    valid = offsets < N_ELEMENTS

    src_offsets = N_ELEMENTS - 1 - offsets
    safe_src_offsets = ct.where(valid, src_offsets, 0)

    vals = ct.gather(input, safe_src_offsets, check_bounds=False, latency=1)
    ct.store(output, index=(bid,), tile=vals, latency=1)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    n_elements = input.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 8

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

        grid = (ct.cdiv(n_words, TILE), 1, 1)
        kernel = _reverse_i32_kernel.with_hints(occupancy=occupancy)
        ct.launch(stream, grid, kernel, (input_i32, output_i32, n_words, mode, TILE))

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "TILE": TILE,
                "occupancy": occupancy,
                "word_packed": True,
                "PACK": pack,
                "MODE": mode,
                "latency": 1,
            }
        )
    else:
        grid = (ct.cdiv(n_elements, TILE), 1, 1)
        kernel = _reverse_element_kernel.with_hints(occupancy=occupancy)
        ct.launch(stream, grid, kernel, (input, output, n_elements, TILE))

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "TILE": TILE,
                "occupancy": occupancy,
                "word_packed": False,
                "latency": 1,
            }
        )

    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
