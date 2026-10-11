import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _vector_add(x, y, partitions, free_size):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    programs = nl.num_programs()
    pid = nl.program_id(0)
    elements_per_program = x.shape[0] // programs
    program_start = pid * elements_per_program
    tile_elements = partitions * free_size
    tiles_per_program = (
        elements_per_program + tile_elements - 1
    ) // tile_elements

    # Larger DMA-native tiles amortize instruction and descriptor overhead.
    # Each core owns a contiguous half, including its final partial tile.
    for tile in range(tiles_per_program):
        local_start = tile * tile_elements
        elements = min(
            tile_elements, elements_per_program - local_start
        )
        columns = elements // partitions
        start = program_start + local_start
        end = start + elements

        xs = x[start:end].reshape((partitions, columns))
        ys = y[start:end].reshape((partitions, columns))
        os = out[start:end].reshape((partitions, columns))

        nisa.dma_compute(
            dst=os,
            srcs=[xs, ys],
            reduce_op=nl.add,
        )

    return out


def run(x, y):
    global _last_config

    n = int(x.shape[0])
    programs = 2
    partitions = 128
    max_free_size = 16384

    assert n % (programs * partitions) == 0

    elements_per_program = n // programs
    free_size = min(
        max_free_size, elements_per_program // partitions
    )
    tile_elements = partitions * free_size
    tiles_per_program = (
        elements_per_program + tile_elements - 1
    ) // tile_elements
    last_tile_elements = (
        elements_per_program
        - (tiles_per_program - 1) * tile_elements
    )

    result = _vector_add[2](x, y, partitions, free_size)

    _last_config = {
        "method": "dma_compute",
        "lnc": programs,
        "assignment": "contiguous_halves",
        "partitions": partitions,
        "max_free_size": max_free_size,
        "free_size": free_size,
        "tile_elements": tile_elements,
        "tiles_per_program": tiles_per_program,
        "tile_count": programs * tiles_per_program,
        "last_tile_free_size": last_tile_elements // partitions,
        "n": n,
    }
    return result


def get_last_config() -> dict:
    return dict(_last_config)
