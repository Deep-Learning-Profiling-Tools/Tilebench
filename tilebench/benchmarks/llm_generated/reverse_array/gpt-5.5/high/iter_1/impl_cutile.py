import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _reverse_kernel(input, output, N_ELEMENTS: ConstInt, TILE: ConstInt):
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

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _reverse_kernel, (input, output, n_elements, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
