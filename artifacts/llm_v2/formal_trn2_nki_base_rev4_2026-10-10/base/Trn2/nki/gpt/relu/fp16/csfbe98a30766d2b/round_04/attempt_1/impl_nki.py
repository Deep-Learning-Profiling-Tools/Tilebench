import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _relu_kernel(x, free_size, prefetch_distance):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    partitions = 128
    programs = nl.num_programs()
    pid = nl.program_id(0)
    tiles_per_core = x.shape[0] // (programs * partitions * free_size)

    tiled_shape = (programs, tiles_per_core, partitions, free_size)
    src = x.reshape(tiled_shape)
    dst = out.reshape(tiled_shape)

    # Disjoint on-chip storage keeps later loads independent of earlier stores.
    tiles = nl.ndarray(
        (partitions, tiles_per_core, free_size),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )

    for i in range(min(prefetch_distance, tiles_per_core)):
        nisa.dma_copy(dst=tiles[:, i, :], src=src[pid, i])

    for i in range(tiles_per_core):
        next_tile = i + prefetch_distance
        if next_tile < tiles_per_core:
            nisa.dma_copy(
                dst=tiles[:, next_tile, :],
                src=src[pid, next_tile],
            )

        nisa.tensor_scalar(
            dst=tiles[:, i, :],
            data=tiles[:, i, :],
            op0=nl.maximum,
            operand0=0.0,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(dst=dst[pid, i], src=tiles[:, i, :])

    return out


def run(x: torch.Tensor):
    global _LAST_CONFIG

    n = x.numel()
    partitions = 128
    programs = 2
    target_tiles_per_core = 4
    prefetch_distance = 2

    free_size = 1024
    while (
        free_size < 4096
        and n >= programs * target_tiles_per_core * partitions * (2 * free_size)
    ):
        free_size *= 2

    tile_elements = partitions * free_size
    assert n % (programs * tile_elements) == 0
    tiles_per_core = n // (programs * tile_elements)

    result = _relu_kernel[2](x, free_size, prefetch_distance)
    _LAST_CONFIG = {
        "n": n,
        "lnc": programs,
        "partition_size": partitions,
        "free_size": free_size,
        "tile_elements": tile_elements,
        "tiles_per_core": tiles_per_core,
        "prefetch_distance": prefetch_distance,
        "sbuf_bytes_per_core": (n // programs) * 2,
        "storage": "disjoint_on_chip_tiles",
        "engine": "vector",
        "operation": "maximum_zero",
    }
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
