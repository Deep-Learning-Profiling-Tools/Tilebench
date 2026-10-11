import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _masked_rescale(x, x_keep, scale, tile_free, compute_free):
    partitions = 128
    tile_elements = partitions * tile_free
    tile_count = x.shape[0] // tile_elements
    tiles_per_program = tile_count // nl.num_programs()
    first_tile = nl.program_id(0) * tiles_per_program

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

        # Keep the following tile's transfers independent of this tile's
        # arithmetic and output transfer.
        if i + 1 < tiles_per_program:
            next_values = nl.ndarray(
                (partitions, tile_free), dtype=x.dtype, buffer=nl.sbuf
            )
            next_keep = nl.ndarray(
                (partitions, tile_free), dtype=x_keep.dtype, buffer=nl.sbuf
            )
            nisa.dma_copy(dst=next_values, src=x_tiles[index + 1])
            nisa.dma_copy(dst=next_keep, src=keep_tiles[index + 1])

        # Smaller compute slices let VectorE masking and ScalarE rescaling
        # overlap, while DMA operates on larger contiguous tiles.
        for j in range(tile_free // compute_free):
            f0 = j * compute_free
            f1 = f0 + compute_free
            masked = nl.ndarray(
                (partitions, compute_free),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            nisa.tensor_tensor(
                dst=masked,
                data1=values[:, f0:f1],
                data2=keep[:, f0:f1],
                op=nl.multiply,
                engine=nisa.engine.vector,
            )
            nisa.tensor_scalar(
                dst=values[:, f0:f1],
                data=masked,
                op0=nl.multiply,
                operand0=scale,
                engine=nisa.engine.scalar,
            )

        nisa.dma_copy(dst=out_tiles[index], src=values)

        if i + 1 < tiles_per_program:
            values = next_values
            keep = next_keep

    return out


def run(x, x_keep, p):
    global _LAST_CONFIG

    n = x.shape[0]
    tile_free = 4096 if n >= 4 * 1048576 else 2048
    compute_free = 1024
    tile_elements = 128 * tile_free

    _LAST_CONFIG = {
        "lnc": 2,
        "tile_partitions": 128,
        "tile_free": tile_free,
        "tile_elements": tile_elements,
        "compute_free": compute_free,
        "tiles_per_core": n // (2 * tile_elements),
        "prefetch_tiles": 1,
        "compute_dtype": "float32",
        "mask_engine": "vector",
        "rescale_engine": "scalar",
    }

    return _masked_rescale[2](
        x, x_keep, 1.0 / (1.0 - p), tile_free, compute_free
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
