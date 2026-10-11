import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _masked_rescale(x, x_keep, scale, tile_free):
    partitions = 128
    programs = nl.num_programs()
    pid = nl.program_id(0)

    elements_per_program = x.shape[0] // programs
    tile_elements = partitions * tile_free
    tile_count = (
        elements_per_program + tile_elements - 1
    ) // tile_elements

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    for i in range(tile_count):
        local_offset = i * tile_elements
        offset = pid * elements_per_program + local_offset
        width = min(
            tile_free,
            (elements_per_program - local_offset) // partitions,
        )

        values = nl.ndarray(
            (partitions, width), dtype=x.dtype, buffer=nl.sbuf
        )
        keep = nl.ndarray(
            (partitions, width), dtype=x_keep.dtype, buffer=nl.sbuf
        )
        masked = nl.ndarray(
            (partitions, width), dtype=nl.float32, buffer=nl.sbuf
        )

        nisa.dma_copy(
            dst=values,
            src=x.ap(
                pattern=[[width, partitions], [1, width]],
                offset=offset,
            ),
        )
        nisa.dma_copy(
            dst=keep,
            src=x_keep.ap(
                pattern=[[width, partitions], [1, width]],
                offset=offset,
            ),
        )

        # Mask and rescale in fp32, narrowing only the final result.
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

        nisa.dma_copy(
            dst=out.ap(
                pattern=[[width, partitions], [1, width]],
                offset=offset,
            ),
            src=values,
        )

    return out


def run(x, x_keep, p):
    global _LAST_CONFIG

    partitions = 128
    tile_free = 8192
    tile_elements = partitions * tile_free
    elements_per_core = x.shape[0] // 2
    tiles_per_core = (
        elements_per_core + tile_elements - 1
    ) // tile_elements
    last_tile_free = (
        elements_per_core - (tiles_per_core - 1) * tile_elements
    ) // partitions

    _LAST_CONFIG = {
        "lnc": 2,
        "tile_partitions": partitions,
        "tile_free": tile_free,
        "tile_elements": tile_elements,
        "tiles_per_core": tiles_per_core,
        "last_tile_free": last_tile_free,
        "compute_dtype": "float32",
        "mask_engine": "vector",
        "rescale_engine": "scalar",
    }

    return _masked_rescale[2](
        x, x_keep, 1.0 / (1.0 - p), tile_free
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
