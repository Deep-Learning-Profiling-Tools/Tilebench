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

    # Every configured shape divides exactly into these tiles.
    assert elements % (partitions * free_size) == 0
    assert tiles % programs == 0

    for block in range(tiles_per_program):
        r0 = (pid * tiles_per_program + block) * partitions

        xt = nl.ndarray(
            (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
        )
        yt = nl.ndarray(
            (partitions, free_size), dtype=nl.float32, buffer=nl.sbuf
        )
        result = nl.ndarray(
            (partitions, free_size), dtype=x.dtype, buffer=nl.sbuf
        )

        # Convert both operands to fp32 before arithmetic.
        nisa.dma_copy(dst=xt, src=xf[r0:r0 + partitions, :])
        nisa.dma_copy(dst=yt, src=yf[r0:r0 + partitions, :])

        # Native fp32 SiLU implements x * sigmoid(x).
        nisa.activation(dst=xt, op=nl.silu, data=xt)
        # The fp32 product is downcast once into the output tile.
        nisa.tensor_tensor(
            dst=result, data1=xt, data2=yt, op=nl.multiply
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
        "compute_dtype": "float32",
        "output_dtype": "float16",
    }
    return _swiglu_kernel[2](x, y, _PARTITIONS, _FREE_SIZE)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
