import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _relu_kernel(x, free_size):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    partitions = 128
    programs = nl.num_programs()
    pid = nl.program_id(0)
    tiles_per_core = x.shape[0] // (programs * partitions * free_size)

    tiled_shape = (programs, tiles_per_core, partitions, free_size)
    src = x.reshape(tiled_shape)
    dst = out.reshape(tiled_shape)

    for i in range(tiles_per_core):
        tile = nl.ndarray(
            (partitions, free_size), dtype=x.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=tile, src=src[pid, i])
        nisa.tensor_scalar(
            dst=tile,
            data=tile,
            op0=nl.maximum,
            operand0=0.0,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(dst=dst[pid, i], src=tile)

    return out


def run(x: torch.Tensor):
    global _LAST_CONFIG

    n = x.numel()
    partitions = 128
    programs = 2
    target_tiles_per_core = 4

    free_size = 1024
    while (
        free_size < 4096
        and n >= programs * target_tiles_per_core * partitions * (2 * free_size)
    ):
        free_size *= 2

    # Every configured length is divisible by this complete two-core tile group.
    assert n % (programs * partitions * free_size) == 0

    result = _relu_kernel[2](x, free_size)
    _LAST_CONFIG = {
        "n": n,
        "lnc": programs,
        "partition_size": partitions,
        "free_size": free_size,
        "tile_elements": partitions * free_size,
        "tiles_per_core": n // (programs * partitions * free_size),
        "engine": "vector",
        "operation": "maximum_zero",
    }
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
