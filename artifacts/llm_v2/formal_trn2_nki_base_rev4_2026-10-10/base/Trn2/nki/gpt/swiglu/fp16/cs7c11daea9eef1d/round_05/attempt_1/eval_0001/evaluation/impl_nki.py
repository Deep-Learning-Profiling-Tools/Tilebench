import nki
import nki.language as nl
import nki.isa as nisa


_PARTITIONS = 128
_COMPUTE_FREE = 2048
_MIN_DMA_FREE = 4096
_MAX_DMA_FREE = 16384
_LAST_CONFIG = {}


@nki.jit
def _swiglu_kernel(x, y, partitions, free_size, compute_free):
    m, n = x.shape
    elements = m * n

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    # These tiles divide every shape in the configured input domain.
    assert elements % (2 * partitions * free_size) == 0
    assert free_size % compute_free == 0

    rows = elements // free_size
    xf = x.reshape((rows, free_size))
    yf = y.reshape((rows, free_size))
    of = out.reshape((rows, free_size))

    programs = nl.num_programs()
    pid = nl.program_id(0)
    tiles_per_program = rows // (partitions * programs)

    for block in range(tiles_per_program):
        r0 = (pid * tiles_per_program + block) * partitions

        xt = nl.ndarray(
            (partitions, free_size), dtype=x.dtype, buffer=nl.sbuf
        )
        yt = nl.ndarray(
            (partitions, free_size), dtype=y.dtype, buffer=nl.sbuf
        )

        nisa.dma_copy(dst=xt, src=xf[r0:r0 + partitions, :])
        nisa.dma_copy(dst=yt, src=yf[r0:r0 + partitions, :])

        for chunk in range(free_size // compute_free):
            c0 = chunk * compute_free
            c1 = c0 + compute_free

            # Chunk-local fp32 intermediates allow storage reuse without
            # imposing dependencies between independent compute chunks.
            gate = nl.ndarray(
                (partitions, compute_free),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )

            # ScalarE promotes x before computing SiLU in fp32.
            nisa.activation(
                dst=gate,
                op=nl.silu,
                data=xt[:, c0:c1],
            )

            # VectorE promotes y and performs the product in fp32.
            # Reuse the private SBUF y tile for the sole downcast;
            # the HBM input y remains read-only.
            nisa.tensor_tensor(
                dst=yt[:, c0:c1],
                data1=gate,
                data2=yt[:, c0:c1],
                op=nl.multiply,
            )

        nisa.dma_copy(
            dst=of[r0:r0 + partitions, :],
            src=yt,
        )

    return out


def run(x, y):
    global _LAST_CONFIG
    assert x.shape == y.shape
    assert x.dtype == y.dtype

    m, n = x.shape
    elements = m * n

    # Grow contiguous DMA transfers while retaining at least eight
    # tiles per physical core when selecting a larger tile.
    free_size = _MIN_DMA_FREE
    while (
        free_size < _MAX_DMA_FREE
        and elements >= 32 * _PARTITIONS * free_size
    ):
        free_size *= 2

    _LAST_CONFIG = {
        "partitions": _PARTITIONS,
        "free_size": free_size,
        "compute_free": _COMPUTE_FREE,
        "lnc": 2,
        "tiles_per_program": elements // (2 * _PARTITIONS * free_size),
        "mapping": "flattened_contiguous",
        "activation": "native_silu",
        "input_storage_dtype": "float16",
        "input_promotion": "compute_engine_fp32",
        "compute_dtype": "float32",
        "gate_dtype": "float32",
        "gate_allocation": "compute_chunk_local",
        "output_dtype": "float16",
        "output_staging": "reuse_private_y_tile",
    }

    return _swiglu_kernel[2](
        x, y, _PARTITIONS, free_size, _COMPUTE_FREE
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
