import nki
import nki.language as nl
import nki.isa as nisa


_PARTITIONS = 128
_COMPUTE_FREE = 2048
_LARGE_TILE_THRESHOLD = 16 * 1024 * 1024
_LAST_CONFIG = {}


@nki.jit
def _swiglu_kernel(x, y, partitions, free_size, compute_free):
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

    # All configured shapes divide these tiles exactly.
    assert elements % (partitions * free_size * programs) == 0
    assert free_size % compute_free == 0

    for block in range(tiles_per_program):
        r0 = (pid * tiles_per_program + block) * partitions

        xt = nl.ndarray(
            (partitions, free_size), dtype=x.dtype, buffer=nl.sbuf
        )
        yt = nl.ndarray(
            (partitions, free_size), dtype=y.dtype, buffer=nl.sbuf
        )
        gate = nl.ndarray(
            (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
        )
        result = nl.ndarray(
            (partitions, free_size), dtype=x.dtype, buffer=nl.sbuf
        )

        # Large contiguous transfers amortize DMA instruction overhead.
        nisa.dma_copy(dst=xt, src=xf[r0:r0 + partitions, :])
        nisa.dma_copy(dst=yt, src=yf[r0:r0 + partitions, :])

        # Independent subtiles let ScalarE and VectorE overlap while
        # retaining a single large output transfer.
        for chunk in range(free_size // compute_free):
            c0 = chunk * compute_free
            c1 = c0 + compute_free

            # ScalarE promotes the input before arithmetic; the gate
            # remains fp32, with no intermediate narrowing conversion.
            nisa.activation(
                dst=gate[:, c0:c1],
                op=nl.silu,
                data=xt[:, c0:c1],
            )

            # VectorE promotes y before multiplying in fp32. Writing
            # result performs the sole downcast.
            nisa.tensor_tensor(
                dst=result[:, c0:c1],
                data1=gate[:, c0:c1],
                data2=yt[:, c0:c1],
                op=nl.multiply,
            )

        nisa.dma_copy(
            dst=of[r0:r0 + partitions, :],
            src=result,
        )

    return out


def run(x, y):
    global _LAST_CONFIG
    assert x.shape == y.shape
    assert x.dtype == y.dtype

    m, n = x.shape
    elements = m * n
    free_size = 8192 if elements >= _LARGE_TILE_THRESHOLD else 4096

    _LAST_CONFIG = {
        "partitions": _PARTITIONS,
        "free_size": free_size,
        "compute_free": _COMPUTE_FREE,
        "lnc": 2,
        "tiles_per_program": elements // (_PARTITIONS * free_size * 2),
        "mapping": "flattened_contiguous",
        "activation": "native_silu",
        "input_storage_dtype": "float16",
        "input_promotion": "compute_engine_fp32",
        "compute_dtype": "float32",
        "gate_dtype": "float32",
        "output_dtype": "float16",
    }

    return _swiglu_kernel[2](
        x, y, _PARTITIONS, free_size, _COMPUTE_FREE
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
