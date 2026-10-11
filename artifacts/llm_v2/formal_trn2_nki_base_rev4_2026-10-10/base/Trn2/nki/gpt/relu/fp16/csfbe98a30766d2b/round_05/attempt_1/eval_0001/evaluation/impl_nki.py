import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _relu_kernel(x, free_size, prefetch_distance, output_slots):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    partitions = 128
    programs = nl.num_programs()
    pid = nl.program_id(0)
    tiles_per_core = x.shape[0] // (programs * partitions * free_size)

    # Each core processes alternating, individually contiguous HBM tiles.
    tiled_shape = (tiles_per_core, programs, partitions, free_size)
    src = x.reshape(tiled_shape)
    dst = out.reshape(tiled_shape)

    inputs = nl.ndarray(
        (partitions, tiles_per_core, free_size),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )
    outputs = nl.ndarray(
        (partitions, output_slots, free_size),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )

    for i in range(min(prefetch_distance, tiles_per_core)):
        nisa.dma_copy(dst=inputs[:, i, :], src=src[i, pid])

    for i in range(tiles_per_core):
        next_tile = i + prefetch_distance
        if next_tile < tiles_per_core:
            nisa.dma_copy(
                dst=inputs[:, next_tile, :],
                src=src[next_tile, pid],
            )

        slot = i % output_slots
        # Separate destinations let input DMA and output DMA overlap without
        # writing back into the loaded input tiles.
        nisa.tensor_scalar(
            dst=outputs[:, slot, :],
            data=inputs[:, i, :],
            op0=nl.maximum,
            operand0=0.0,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(dst=dst[i, pid], src=outputs[:, slot, :])

    return out


def run(x: torch.Tensor):
    global _LAST_CONFIG

    n = x.numel()
    partitions = 128
    programs = 2
    target_tiles_per_core = 4
    prefetch_distance = 2
    output_slots = 4

    free_size = 1024
    while (
        free_size < 4096
        and n >= programs * target_tiles_per_core * partitions * (2 * free_size)
    ):
        free_size *= 2

    tile_elements = partitions * free_size
    assert n % (programs * tile_elements) == 0
    tiles_per_core = n // (programs * tile_elements)

    result = _relu_kernel[2](
        x, free_size, prefetch_distance, output_slots
    )
    _LAST_CONFIG = {
        "n": n,
        "lnc": programs,
        "partition_size": partitions,
        "free_size": free_size,
        "tile_elements": tile_elements,
        "tiles_per_core": tiles_per_core,
        "prefetch_distance": prefetch_distance,
        "output_slots": output_slots,
        "sbuf_bytes_per_core": (
            tiles_per_core + output_slots
        ) * tile_elements * 2,
        "core_mapping": "interleaved_contiguous_tiles",
        "storage": "disjoint_inputs_and_output_ring",
        "engine": "vector",
        "operation": "maximum_zero",
    }
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
