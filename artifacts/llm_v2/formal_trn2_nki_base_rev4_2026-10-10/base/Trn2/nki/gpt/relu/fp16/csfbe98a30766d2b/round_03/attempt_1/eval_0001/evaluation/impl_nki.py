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
    elements_per_core = x.shape[0] // programs
    tile_elements = partitions * free_size
    tiles_per_core = (elements_per_core + tile_elements - 1) // tile_elements

    src = x.reshape((programs, elements_per_core))
    dst = out.reshape((programs, elements_per_core))

    for i in range(tiles_per_core):
        start = i * tile_elements
        count = min(tile_elements, elements_per_core - start)
        width = count // partitions

        tile = nl.ndarray(
            (partitions, width), dtype=x.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=tile,
            src=src[pid, start:start + count].reshape((partitions, width)),
        )
        nisa.tensor_scalar(
            dst=tile,
            data=tile,
            op0=nl.maximum,
            operand0=0.0,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(
            dst=dst[pid, start:start + count].reshape((partitions, width)),
            src=tile,
        )

    return out


def run(x: torch.Tensor):
    global _LAST_CONFIG

    n = x.numel()
    partitions = 128
    programs = 2
    target_tiles_per_core = 4

    free_size = 1024
    while (
        free_size < 8192
        and n >= programs * target_tiles_per_core * partitions * (2 * free_size)
    ):
        free_size *= 2

    assert n % (programs * partitions) == 0
    elements_per_core = n // programs
    tile_elements = partitions * free_size
    tail_elements = elements_per_core % tile_elements

    result = _relu_kernel[2](x, free_size)
    _LAST_CONFIG = {
        "n": n,
        "lnc": programs,
        "partition_size": partitions,
        "free_size": free_size,
        "tile_elements": tile_elements,
        "tiles_per_core": (
            elements_per_core + tile_elements - 1
        ) // tile_elements,
        "tail_free_size": tail_elements // partitions,
        "target_tiles_per_core": target_tiles_per_core,
        "engine": "vector",
        "operation": "maximum_zero",
    }
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
