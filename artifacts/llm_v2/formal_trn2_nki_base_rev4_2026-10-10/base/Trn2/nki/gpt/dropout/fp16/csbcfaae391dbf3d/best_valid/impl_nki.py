import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _masked_rescale(x, x_keep, scale, tile_free):
    partitions = 128
    tile_elements = partitions * tile_free
    tile_count = x.shape[0] // tile_elements
    programs = nl.num_programs()
    pid = nl.program_id(0)
    tiles_per_program = tile_count // programs
    first_tile = pid * tiles_per_program

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    x_tiles = x.reshape((tile_count, partitions, tile_free))
    keep_tiles = x_keep.reshape((tile_count, partitions, tile_free))
    out_tiles = out.reshape((tile_count, partitions, tile_free))

    values = nl.ndarray(
        (partitions, tile_free), dtype=x.dtype, buffer=nl.sbuf
    )
    keep = nl.ndarray(
        (partitions, tile_free), dtype=x_keep.dtype, buffer=nl.sbuf
    )
    nisa.dma_copy(dst=values, src=x_tiles[first_tile])
    nisa.dma_copy(dst=keep, src=keep_tiles[first_tile])

    for i in range(tiles_per_program):
        index = first_tile + i
        masked = nl.ndarray(
            (partitions, tile_free), dtype=nl.float32, buffer=nl.sbuf
        )

        nisa.tensor_tensor(
            dst=masked,
            data1=values,
            data2=keep,
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
        nisa.tensor_scalar(
            dst=values,
            data=masked,
            op0=nl.multiply,
            operand0=scale,
            engine=nisa.engine.scalar,
        )

        # Issue the next input transfers before the current output transfer,
        # allowing DMA to overlap the current tile's arithmetic.
        if i + 1 < tiles_per_program:
            next_values = nl.ndarray(
                (partitions, tile_free), dtype=x.dtype, buffer=nl.sbuf
            )
            next_keep = nl.ndarray(
                (partitions, tile_free), dtype=x_keep.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=next_values, src=x_tiles[index + 1])
            nisa.dma_copy(dst=next_keep, src=keep_tiles[index + 1])

        nisa.dma_copy(dst=out_tiles[index], src=values)

        if i + 1 < tiles_per_program:
            values = next_values
            keep = next_keep

    return out


def run(x, x_keep, p):
    global _LAST_CONFIG

    tile_free = 2048
    tile_elements = 128 * tile_free
    n = x.shape[0]

    _LAST_CONFIG = {
        "lnc": 2,
        "tile_partitions": 128,
        "tile_free": tile_free,
        "tile_elements": tile_elements,
        "tiles_per_core": n // (2 * tile_elements),
        "prefetch_tiles": 1,
        "compute_dtype": "float32",
        "mask_engine": "vector",
        "rescale_engine": "scalar",
    }

    return _masked_rescale[2](
        x, x_keep, 1.0 / (1.0 - p), tile_free
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
