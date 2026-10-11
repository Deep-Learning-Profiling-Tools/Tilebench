import nki
import nki.language as nl
import nki.isa as nisa


_last_config = {}


@nki.jit
def _vector_add(x, y, partitions, free_size):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    elements_per_program = partitions * free_size
    offset = nl.program_id(0) * elements_per_program
    pattern = [[free_size, partitions], [1, free_size]]

    # The DMA engines stream each contiguous half directly between HBM
    # buffers, adding in fp32 and rounding once to the output dtype.
    nisa.dma_compute(
        dst=out.ap(pattern=pattern, offset=offset),
        srcs=[
            x.ap(pattern=pattern, offset=offset),
            y.ap(pattern=pattern, offset=offset),
        ],
        reduce_op=nl.add,
    )
    return out


def run(x, y):
    global _last_config

    n = int(x.shape[0])
    partitions = 128
    programs = 2

    # All configured lengths divide exactly into two 128-row HBM views.
    free_size = n // (programs * partitions)
    assert n == programs * partitions * free_size

    result = _vector_add[2](x, y, partitions, free_size)

    _last_config = {
        "method": "dma_compute",
        "lnc": programs,
        "assignment": "contiguous_halves",
        "partitions": partitions,
        "free_size": free_size,
        "tile_elements": partitions * free_size,
        "tiles_per_program": 1,
        "n": n,
    }
    return result


def get_last_config() -> dict:
    return dict(_last_config)
