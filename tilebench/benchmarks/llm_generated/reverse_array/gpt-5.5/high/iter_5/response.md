```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _reverse_i64_kernel(
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

    vals_i = tl.load(input_ptr + safe_rev_offsets, eviction_policy="evict_first")
    vals = vals_i.to(tl.uint64, bitcast=True)

    if MODE == 1:
        # int8 packed path: reverse eight bytes within each 64-bit word.
        m8 = 0x00FF00FF00FF00FF
        m16 = 0x0000FFFF0000FFFF
        vals = ((vals & m8) << 8) | ((vals >> 8) & m8)
        vals = ((vals & m16) << 16) | ((vals >> 16) & m16)
        vals = (vals << 32) | ((vals >> 32) & 0x00000000FFFFFFFF)
    elif MODE == 2:
        # fp16/bf16 packed path: reverse four 16-bit lanes within each 64-bit word.
        m16 = 0x0000FFFF0000FFFF
        vals = ((vals & m16) << 16) | ((vals >> 16) & m16)
        vals = (vals << 32) | ((vals >> 32) & 0x00000000FFFFFFFF)

    tl.store(output_ptr + offsets, vals.to(tl.int64, bitcast=True), mask=mask)


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

    vals_i = tl.load(input_ptr + safe_rev_offsets, eviction_policy="evict_first")

    if MODE == 1:
        vals = vals_i.to(tl.uint32, bitcast=True)
        b0 = (vals & 0x000000FF) << 24
        b1 = (vals & 0x0000FF00) << 8
        b2 = (vals >> 8) & 0x0000FF00
        b3 = (vals >> 24) & 0x000000FF
        vals = b0 | b1 | b2 | b3
        tl.store(output_ptr + offsets, vals.to(tl.int32, bitcast=True), mask=mask)
    elif MODE == 2:
        vals = vals_i.to(tl.uint32, bitcast=True)
        lo = (vals & 0x0000FFFF) << 16
        hi = (vals >> 16) & 0x0000FFFF
        vals = lo | hi
        tl.store(output_ptr + offsets, vals.to(tl.int32, bitcast=True), mask=mask)
    else:
        tl.store(output_ptr + offsets, vals_i, mask=mask)


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

    if elem_size == 1 and n_elements % 8 == 0:
        pack = 8
        mode = 1
        input_i64 = input.view(torch.int64)
        output_i64 = output.view(torch.int64)
        n_words = n_elements // pack

        grid = (triton.cdiv(n_words, BLOCK_SIZE),)
        _reverse_i64_kernel[grid](
            input_i64,
            output_i64,
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
                "WORD_BITS": 64,
            }
        )
    elif elem_size == 2 and n_elements % 4 == 0:
        pack = 4
        mode = 2
        input_i64 = input.view(torch.int64)
        output_i64 = output.view(torch.int64)
        n_words = n_elements // pack

        grid = (triton.cdiv(n_words, BLOCK_SIZE),)
        _reverse_i64_kernel[grid](
            input_i64,
            output_i64,
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
                "WORD_BITS": 64,
            }
        )
    elif elem_size == 1 and n_elements % 4 == 0:
        pack = 4
        mode = 1
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
                "WORD_BITS": 32,
            }
        )
    elif elem_size == 2 and n_elements % 2 == 0:
        pack = 2
        mode = 2
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
                "WORD_BITS": 32,
            }
        )
    elif elem_size == 4:
        pack = 1
        mode = 0
        input_i32 = input.view(torch.int32)
        output_i32 = output.view(torch.int32)
        n_words = n_elements

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
                "WORD_BITS": 32,
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


@ct.kernel(occupancy=8)
def _reverse_i64_kernel(input, output, N_WORDS: ConstInt, MODE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)

    offsets = bid * TILE + ct.arange(TILE, dtype=np.int32)
    valid = offsets < N_WORDS

    src_offsets = N_WORDS - 1 - offsets
    safe_src_offsets = ct.where(valid, src_offsets, 0)

    vals_i = ct.gather(input, safe_src_offsets, check_bounds=False, latency=1)
    vals = ct.bitcast(vals_i, ct.uint64)

    if MODE == 1:
        # int8 packed path: reverse eight bytes within each 64-bit word.
        m8 = 0x00FF00FF00FF00FF
        m16 = 0x0000FFFF0000FFFF
        vals = ct.bitwise_lshift(vals & m8, 8) | (ct.bitwise_rshift(vals, 8) & m8)
        vals = ct.bitwise_lshift(vals & m16, 16) | (ct.bitwise_rshift(vals, 16) & m16)
        vals = ct.bitwise_lshift(vals, 32) | (ct.bitwise_rshift(vals, 32) & 0x00000000FFFFFFFF)
    elif MODE == 2:
        # fp16/bf16 packed path: reverse four 16-bit lanes within each 64-bit word.
        m16 = 0x0000FFFF0000FFFF
        vals = ct.bitwise_lshift(vals & m16, 16) | (ct.bitwise_rshift(vals, 16) & m16)
        vals = ct.bitwise_lshift(vals, 32) | (ct.bitwise_rshift(vals, 32) & 0x00000000FFFFFFFF)

    ct.store(output, index=(bid,), tile=ct.bitcast(vals, ct.int64))


@ct.kernel(occupancy=8)
def _reverse_i32_kernel(input, output, N_WORDS: ConstInt, MODE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)

    offsets = bid * TILE + ct.arange(TILE, dtype=np.int32)
    valid = offsets < N_WORDS

    src_offsets = N_WORDS - 1 - offsets
    safe_src_offsets = ct.where(valid, src_offsets, 0)

    vals_i = ct.gather(input, safe_src_offsets, check_bounds=False, latency=1)

    if MODE == 1:
        vals = ct.bitcast(vals_i, ct.uint32)
        b0 = ct.bitwise_lshift(vals & 0x000000FF, 24)
        b1 = ct.bitwise_lshift(vals & 0x0000FF00, 8)
        b2 = ct.bitwise_rshift(vals, 8) & 0x0000FF00
        b3 = ct.bitwise_rshift(vals, 24) & 0x000000FF
        vals = b0 | b1 | b2 | b3
        ct.store(output, index=(bid,), tile=ct.bitcast(vals, ct.int32))
    elif MODE == 2:
        vals = ct.bitcast(vals_i, ct.uint32)
        vals = ct.bitwise_lshift(vals & 0x0000FFFF, 16) | (ct.bitwise_rshift(vals, 16) & 0x0000FFFF)
        ct.store(output, index=(bid,), tile=ct.bitcast(vals, ct.int32))
    else:
        ct.store(output, index=(bid,), tile=vals_i)


@ct.kernel(occupancy=8)
def _reverse_element_kernel(input, output, N_ELEMENTS: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)

    offsets = bid * TILE + ct.arange(TILE, dtype=np.int32)
    valid = offsets < N_ELEMENTS

    src_offsets = N_ELEMENTS - 1 - offsets
    safe_src_offsets = ct.where(valid, src_offsets, 0)

    vals = ct.gather(input, safe_src_offsets, check_bounds=False, latency=1)
    ct.store(output, index=(bid,), tile=vals)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    n_elements = input.numel()
    stream = torch.cuda.current_stream()

    TILE64 = 2048
    TILE32 = 4096
    occupancy = 8

    elem_size = input.element_size()

    if elem_size == 1 and n_elements % 8 == 0:
        pack = 8
        mode = 1
        input_i64 = input.view(torch.int64)
        output_i64 = output.view(torch.int64)
        n_words = n_elements // pack

        grid = (ct.cdiv(n_words, TILE64), 1, 1)
        ct.launch(stream, grid, _reverse_i64_kernel, (input_i64, output_i64, n_words, mode, TILE64))

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "TILE": TILE64,
                "occupancy": occupancy,
                "word_packed": True,
                "PACK": pack,
                "MODE": mode,
                "WORD_BITS": 64,
            }
        )
    elif elem_size == 2 and n_elements % 4 == 0:
        pack = 4
        mode = 2
        input_i64 = input.view(torch.int64)
        output_i64 = output.view(torch.int64)
        n_words = n_elements // pack

        grid = (ct.cdiv(n_words, TILE64), 1, 1)
        ct.launch(stream, grid, _reverse_i64_kernel, (input_i64, output_i64, n_words, mode, TILE64))

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "TILE": TILE64,
                "occupancy": occupancy,
                "word_packed": True,
                "PACK": pack,
                "MODE": mode,
                "WORD_BITS": 64,
            }
        )
    elif elem_size == 1 and n_elements % 4 == 0:
        pack = 4
        mode = 1
        input_i32 = input.view(torch.int32)
        output_i32 = output.view(torch.int32)
        n_words = n_elements // pack

        grid = (ct.cdiv(n_words, TILE32), 1, 1)
        ct.launch(stream, grid, _reverse_i32_kernel, (input_i32, output_i32, n_words, mode, TILE32))

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "TILE": TILE32,
                "occupancy": occupancy,
                "word_packed": True,
                "PACK": pack,
                "MODE": mode,
                "WORD_BITS": 32,
            }
        )
    elif elem_size == 2 and n_elements % 2 == 0:
        pack = 2
        mode = 2
        input_i32 = input.view(torch.int32)
        output_i32 = output.view(torch.int32)
        n_words = n_elements // pack

        grid = (ct.cdiv(n_words, TILE32), 1, 1)
        ct.launch(stream, grid, _reverse_i32_kernel, (input_i32, output_i32, n_words, mode, TILE32))

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "TILE": TILE32,
                "occupancy": occupancy,
                "word_packed": True,
                "PACK": pack,
                "MODE": mode,
                "WORD_BITS": 32,
            }
        )
    elif elem_size == 4:
        pack = 1
        mode = 0
        input_i32 = input.view(torch.int32)
        output_i32 = output.view(torch.int32)
        n_words = n_elements

        grid = (ct.cdiv(n_words, TILE32), 1, 1)
        ct.launch(stream, grid, _reverse_i32_kernel, (input_i32, output_i32, n_words, mode, TILE32))

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "TILE": TILE32,
                "occupancy": occupancy,
                "word_packed": True,
                "PACK": pack,
                "MODE": mode,
                "WORD_BITS": 32,
            }
        )
    else:
        grid = (ct.cdiv(n_elements, TILE32), 1, 1)
        ct.launch(stream, grid, _reverse_element_kernel, (input, output, n_elements, TILE32))

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "TILE": TILE32,
                "occupancy": occupancy,
                "word_packed": False,
            }
        )

    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
