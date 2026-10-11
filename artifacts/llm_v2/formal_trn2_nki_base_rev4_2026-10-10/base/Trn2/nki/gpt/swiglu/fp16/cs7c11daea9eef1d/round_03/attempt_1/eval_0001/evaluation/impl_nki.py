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

    # Every configured shape is exactly divisible by the two-core tile.
    assert elements % (partitions * free_size) == 0
    assert tiles % programs == 0

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

        # Preserve compact DMA/SBUF storage. ScalarE and VectorE widen
        # their input operands to fp32 before performing arithmetic.
        nisa.dma_copy(dst=xt, src=xf[r0:r0 + partitions, :])
        nisa.dma_copy(dst=yt, src=yf[r0:r0 + partitions, :])

        # SiLU is evaluated in fp32 and retained without a downcast.
        nisa.activation(dst=gate, op=nl.silu, data=xt)

        # VectorE widens y to fp32, multiplies in fp32, and performs the
        # sole narrowing conversion when writing the final result.
        nisa.tensor_tensor(
            dst=result,
            data1=gate,
            data2=yt,
            op=nl.multiply,
        )
        nisa.dma_copy(dst=of[r0:r0 + partitions, :], src=result)

    return out


def run(x, y):
    global _LAST_CONFIG
    assert x.shape == y.shape
    assert x.dtype == y.dtype

    m, n = x.shape
    _LAST_CONFIG = {
        "partitions": _PARTITIONS,
        "free_size": _FREE_SIZE,
        "lnc": 2,
        "tiles_per_program": (m * n) // (_PARTITIONS * _FREE_SIZE * 2),
        "mapping": "flattened_contiguous",
        "activation": "native_silu",
        "input_storage_dtype": "float16",
        "input_promotion": "compute_engine_fp32",
        "compute_dtype": "float32",
        "gate_dtype": "float32",
        "output_dtype": "float16",
    }
    return _swiglu_kernel[2](x, y, _PARTITIONS, _FREE_SIZE)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
