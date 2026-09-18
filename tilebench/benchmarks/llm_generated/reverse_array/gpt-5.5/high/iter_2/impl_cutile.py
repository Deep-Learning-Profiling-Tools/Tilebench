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
