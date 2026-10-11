import nki
import nki.language as nl
import nki.isa as nisa


_PARTITIONS = 128
_FREE_SIZE = 4096
_LAST_CONFIG = {}


@nki.jit
def _swiglu_kernel(x, y, partitions, free_size):
    m, n = x.shape
    elements = m * n
    rows = elements // free_size

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    xf = x.reshape((rows, free_size))
    yf = y.reshape((rows, free_size))
    of = out.reshape((rows, free_size))

    programs = nl.num_programs()
    pid = nl.program_id(0)
    tiles = rows // partitions
    tiles_per_program = tiles // programs

    # All configured shapes are exact multiples of the two-core tile.
    assert elements % (partitions * free_size) == 0
    assert tiles % programs == 0

    # Explicit ping-pong storage keeps the next DMA tile live while the
    # current tile is processed, preventing reuse of its on-chip storage.
    xt = nl.ndarray(
        (partitions, 2, free_size), dtype=nl.float32, buffer=nl.sbuf
    )
    yt = nl.ndarray(
        (partitions, 2, free_size), dtype=nl.float32, buffer=nl.sbuf
    )
    result = nl.ndarray(
        (partitions, 2, free_size), dtype=x.dtype, buffer=nl.sbuf
    )

    first_row = pid * tiles_per_program * partitions
    nisa.dma_copy(
        dst=xt[:, 0, :],
        src=xf[first_row:first_row + partitions, :],
    )
    nisa.dma_copy(
        dst=yt[:, 0, :],
        src=yf[first_row:first_row + partitions, :],
    )

    for block in range(tiles_per_program):
        slot = block % 2
        r0 = first_row + block * partitions

        # Issue independent loads ahead of the current compute. DMA can
        # progress concurrently with the Scalar and Vector engines.
        if block + 1 < tiles_per_program:
            next_slot = (block + 1) % 2
            next_row = r0 + partitions
            nisa.dma_copy(
                dst=xt[:, next_slot, :],
                src=xf[next_row:next_row + partitions, :],
            )
            nisa.dma_copy(
                dst=yt[:, next_slot, :],
                src=yf[next_row:next_row + partitions, :],
            )

        # Both inputs are already fp32. SiLU remains fp32, and only the
        # final multiplication result is downcast to the output dtype.
        nisa.activation(
            dst=xt[:, slot, :],
            op=nl.silu,
            data=xt[:, slot, :],
        )
        nisa.tensor_tensor(
            dst=result[:, slot, :],
            data1=xt[:, slot, :],
            data2=yt[:, slot, :],
            op=nl.multiply,
        )
        nisa.dma_copy(
            dst=of[r0:r0 + partitions, :],
            src=result[:, slot, :],
        )

    return out


def run(x, y):
    global _LAST_CONFIG
    assert x.shape == y.shape
    assert x.dtype == y.dtype

    m, n = x.shape
    _LAST_CONFIG = {
        "partitions": _PARTITIONS,
        "free_size": _FREE_SIZE,
        "buffers": 2,
        "prefetch_distance": 1,
        "lnc": 2,
        "tiles_per_program": (m * n) // (_PARTITIONS * _FREE_SIZE * 2),
        "mapping": "flattened_contiguous_ping_pong",
        "activation": "native_silu",
        "compute_dtype": "float32",
        "output_dtype": "float16",
    }
    return _swiglu_kernel[2](x, y, _PARTITIONS, _FREE_SIZE)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
