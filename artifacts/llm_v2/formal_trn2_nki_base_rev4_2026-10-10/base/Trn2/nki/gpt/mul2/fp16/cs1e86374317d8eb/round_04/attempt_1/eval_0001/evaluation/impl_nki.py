import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _mul2_kernel(x, tile_free):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    partitions = 128
    tile_elements = partitions * tile_free
    core_elements = x.shape[0] // nl.num_programs()
    core_start = nl.program_id(0) * core_elements
    tile_count = (core_elements + tile_elements - 1) // tile_elements

    previous_start = core_start
    previous_elements = min(tile_elements, core_elements)
    previous_free = previous_elements // partitions
    previous = nl.ndarray(
        (partitions, previous_free), dtype=x.dtype, buffer=nl.sbuf
    )
    nisa.dma_copy(
        dst=previous,
        src=x[
            previous_start:previous_start + previous_elements
        ].reshape((partitions, previous_free)),
    )

    for tile in range(1, tile_count):
        current_start = core_start + tile * tile_elements
        current_elements = min(
            tile_elements, core_elements - tile * tile_elements
        )
        current_free = current_elements // partitions
        current = nl.ndarray(
            (partitions, current_free), dtype=x.dtype, buffer=nl.sbuf
        )
        nisa.dma_copy(
            dst=current,
            src=x[
                current_start:current_start + current_elements
            ].reshape((partitions, current_free)),
        )

        nisa.tensor_tensor(
            dst=previous,
            data1=previous,
            data2=previous,
            op=nl.add,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(
            dst=out[
                previous_start:previous_start + previous_elements
            ].reshape((partitions, previous_free)),
            src=previous,
        )

        previous = current
        previous_start = current_start
        previous_elements = current_elements
        previous_free = current_free

    nisa.tensor_tensor(
        dst=previous,
        data1=previous,
        data2=previous,
        op=nl.add,
        engine=nisa.engine.vector,
    )
    nisa.dma_copy(
        dst=out[
            previous_start:previous_start + previous_elements
        ].reshape((partitions, previous_free)),
        src=previous,
    )
    return out


def run(x: torch.Tensor) -> torch.Tensor:
    global _LAST_CONFIG

    n = x.numel()
    tile_free = min(8192, n // 512)
    tile_elements = 128 * tile_free
    core_elements = n // 2

    result = _mul2_kernel[2](x, tile_free)
    _LAST_CONFIG = {
        "lnc": 2,
        "partition_size": 128,
        "tile_free": tile_free,
        "tile_elements": tile_elements,
        "tiles_per_core": (
            core_elements + tile_elements - 1
        ) // tile_elements,
        "tail_elements_per_core": core_elements % tile_elements,
        "lookahead_tiles": 1,
        "engine": "vector",
        "arithmetic": "add",
        "in_place_sbuf": True,
        "buffer_reuse": False,
    }
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
