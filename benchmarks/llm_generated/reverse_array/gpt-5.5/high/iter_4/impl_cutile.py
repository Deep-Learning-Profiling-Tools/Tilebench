import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _reverse_i32_kernel(input, output, N_WORDS: ConstInt, MODE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)

    offsets = bid * TILE + ct.arange(TILE, dtype=np.int32)
    valid = offsets < N_WORDS

    src_offsets = N_WORDS - 1 - offsets
    safe_src_offsets = ct.where(valid, src_offsets, 0)

    vals = ct.gather(input, safe_src_offsets, check_bounds=False, latency=1)

    if MODE == 1:
        # int8 packed path: staged 32-bit byte swap.
        m8 = 0x00FF00FF
        vals = ct.bitwise_lshift(vals & m8, 8) | (ct.bitwise_rshift(vals, 8) & m8)
        vals = ct.bitwise_lshift(vals, 16) | (ct.bitwise_rshift(vals, 16) & 0x0000FFFF)
    elif MODE == 2:
        # fp16/bf16 packed path: swap two 16-bit lanes in each 32-bit word.
        vals = ct.bitwise_lshift(vals, 16) | (ct.bitwise_rshift(vals, 16) & 0x0000FFFF)

    ct.store(output, index=(bid,), tile=vals)


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
        ct.launch(stream, grid, _reverse_i32_kernel, (input_i32, output_i32, n_words, mode, TILE))

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "TILE": TILE,
                "occupancy": occupancy,
                "word_packed": True,
                "PACK": pack,
                "MODE": mode,
                "SWAP_PATTERN": 1,
            }
        )
    else:
        grid = (ct.cdiv(n_elements, TILE), 1, 1)
        ct.launch(stream, grid, _reverse_element_kernel, (input, output, n_elements, TILE))

        _LAST_CFG.clear()
        _LAST_CFG.update(
            {
                "TILE": TILE,
                "occupancy": occupancy,
                "word_packed": False,
                "SWAP_PATTERN": 1,
            }
        )

    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
