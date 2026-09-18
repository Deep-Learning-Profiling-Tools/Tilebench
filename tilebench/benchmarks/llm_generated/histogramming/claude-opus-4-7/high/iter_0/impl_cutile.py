import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _hist_kernel(input, output, N, num_bins, TILE: ConstInt):
    bid = ct.bid(0)
    offs = bid * TILE + ct.arange(TILE, dtype=np.int32)
    x = ct.load(input, index=(bid,), shape=(TILE,),
                padding_mode=ct.PaddingMode.UNDETERMINED)
    mask = offs < N
    # Remap OOB tail elements to an out-of-range bin index so the
    # atomic_add is silently dropped by check_bounds=True.
    x = ct.where(mask, x, num_bins)
    ones = ct.ones((TILE,), dtype=np.int32)
    ct.atomic_add(output, (x,), ones,
                  memory_order=ct.MemoryOrder.RELAXED,
                  memory_scope=ct.MemoryScope.DEVICE)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.zeros(num_bins, dtype=torch.int32, device=input.device)
    stream = torch.cuda.current_stream()

    TILE = 1024
    occupancy = 4

    grid = (ct.cdiv(N, TILE), 1, 1)
    kernel = _hist_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (input, output, N, num_bins, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
