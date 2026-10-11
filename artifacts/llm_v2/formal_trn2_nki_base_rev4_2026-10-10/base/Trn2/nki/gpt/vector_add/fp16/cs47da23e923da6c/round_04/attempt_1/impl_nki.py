import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _vector_add(x, y, partitions, free_size):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    tile_elements = partitions * free_size
    tile_count = x.shape[0] // tile_elements
    tiles_per_program = tile_count // nl.num_programs()
    first_tile = nl.program_id(0) * tiles_per_program

    xv = x.reshape((tile_count, partitions, free_size))
    yv = y.reshape((tile_count, partitions, free_size))
    ov = out.reshape((tile_count, partitions, free_size))

    a0 = nl.ndarray((partitions, free_size), dtype=x.dtype, buffer=nl.sbuf)
    b0 = nl.ndarray((partitions, free_size), dtype=y.dtype, buffer=nl.sbuf)
    a1 = nl.ndarray((partitions, free_size), dtype=x.dtype, buffer=nl.sbuf)
    b1 = nl.ndarray((partitions, free_size), dtype=y.dtype, buffer=nl.sbuf)

    nisa.dma_copy(dst=a0, src=xv[first_tile])
    nisa.dma_copy(dst=b0, src=yv[first_tile])

    # Read ahead into the alternate buffers while VectorE processes
    # the current tile. Only the final result is written to HBM.
    for i in range(tiles_per_program):
        tile = first_tile + i
        if i % 2 == 0:
            if i + 1 < tiles_per_program:
                nisa.dma_copy(dst=a1, src=xv[tile + 1])
                nisa.dma_copy(dst=b1, src=yv[tile + 1])
            nisa.tensor_tensor(
                dst=a0,
                data1=a0,
                data2=b0,
                op=nl.add,
                engine=nisa.engine.vector,
            )
            nisa.dma_copy(dst=ov[tile], src=a0)
        else:
            if i + 1 < tiles_per_program:
                nisa.dma_copy(dst=a0, src=xv[tile + 1])
                nisa.dma_copy(dst=b0, src=yv[tile + 1])
            nisa.tensor_tensor(
                dst=a1,
                data1=a1,
                data2=b1,
                op=nl.add,
                engine=nisa.engine.vector,
            )
            nisa.dma_copy(dst=ov[tile], src=a1)

    return out


def run(x, y):
    global _last_config

    n = int(x.shape[0])
    partitions = 128
    free_size = 4096
    programs = 2
    tile_elements = partitions * free_size

    # Every configured length consists of complete tiles on both cores.
    assert n % (programs * tile_elements) == 0

    result = _vector_add[2](x, y, partitions, free_size)
    _last_config = {
        "method": "dma_load_vector_add_dma_store",
        "lnc": programs,
        "assignment": "contiguous_halves",
        "partitions": partitions,
        "free_size": free_size,
        "tile_elements": tile_elements,
        "tiles_per_program": n // (programs * tile_elements),
        "buffers": 2,
        "prefetch_distance": 1,
        "n": n,
    }
    return result


def get_last_config() -> dict:
    return dict(_last_config)
