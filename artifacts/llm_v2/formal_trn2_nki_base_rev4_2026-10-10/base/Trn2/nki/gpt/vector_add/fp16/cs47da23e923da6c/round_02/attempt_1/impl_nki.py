import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


def _add_tile(x, y, out, offset, partitions, free_size):
    pattern = [[free_size, partitions], [1, free_size]]

    xt = nl.ndarray(
        (partitions, free_size), dtype=x.dtype, buffer=nl.sbuf
    )
    yt = nl.ndarray(
        (partitions, free_size), dtype=y.dtype, buffer=nl.sbuf
    )

    nisa.dma_copy(dst=xt, src=x.ap(pattern=pattern, offset=offset))
    nisa.dma_copy(dst=yt, src=y.ap(pattern=pattern, offset=offset))
    nisa.tensor_tensor(
        dst=xt,
        data1=xt,
        data2=yt,
        op=nl.add,
        engine=nisa.engine.vector,
    )
    nisa.dma_copy(
        dst=out.ap(pattern=pattern, offset=offset),
        src=xt,
    )


@nki.jit
def _vector_add(x, y, partitions, free_size):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    programs = nl.num_programs()
    pid = nl.program_id(0)
    elements_per_program = x.shape[0] // programs
    base = pid * elements_per_program

    tile_elements = partitions * free_size
    full_tiles = elements_per_program // tile_elements
    remainder = elements_per_program % tile_elements

    for tile in range(full_tiles):
        _add_tile(
            x, y, out,
            base + tile * tile_elements,
            partitions, free_size,
        )

    if remainder > 0:
        tail_rows = min(partitions, remainder)
        tail_free = remainder // tail_rows
        _add_tile(
            x, y, out,
            base + full_tiles * tile_elements,
            tail_rows, tail_free,
        )

    return out


def run(x, y):
    global _last_config

    n = int(x.shape[0])
    partitions = 128
    free_size = 8192
    tile_elements = partitions * free_size
    elements_per_program = n // 2

    result = _vector_add[2](x, y, partitions, free_size)

    _last_config = {
        "method": "sbuf_vector_add",
        "lnc": 2,
        "assignment": "contiguous_halves",
        "partitions": partitions,
        "free_size": free_size,
        "tile_elements": tile_elements,
        "full_tiles_per_program": elements_per_program // tile_elements,
        "tail_elements_per_program": elements_per_program % tile_elements,
        "n": n,
    }
    return result


def get_last_config() -> dict:
    return dict(_last_config)
