import torch
import nki
import nki.compiler
import nki.language as nl
import nki.isa as nisa


_PARTITIONS = 128
_MAX_FREE = 8192
_LAST_CONFIG = {}


def _map_tile(x, y, offset, rows, cols):
    value = nl.ndarray((rows, cols), dtype=x.dtype, buffer=nl.sbuf)
    result = nl.ndarray((rows, cols), dtype=x.dtype, buffer=nl.sbuf)

    nisa.dma_copy(
        dst=value,
        src=x.ap(pattern=[[cols, rows], [1, cols]], offset=offset),
    )
    nisa.tensor_scalar(
        dst=result,
        data=value,
        op0=nl.multiply,
        operand0=0.01,
        engine=nisa.engine.vector,
    )
    nisa.tensor_tensor(
        dst=result,
        data1=value,
        data2=result,
        op=nl.maximum,
        engine=nisa.engine.vector,
    )
    nisa.dma_copy(
        dst=y.ap(pattern=[[cols, rows], [1, cols]], offset=offset),
        src=result,
    )


@nki.jit(
    compile_config=nki.compiler.CompileConfig(prefetch_auto=True)
)
def _leaky_relu(x, N, tile_free):
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    programs = nl.num_programs()
    pid = nl.program_id(0)
    per_program = N // programs
    base = pid * per_program
    block = 128 * tile_free
    full_blocks = per_program // block

    # Independent allocations let the compiler prefetch subsequent tiles.
    # Loaded tiles remain read-only; only result tiles are stored.
    for i in range(full_blocks):
        _map_tile(x, y, base + i * block, 128, tile_free)

    remaining = per_program - full_blocks * block
    tail_base = base + full_blocks * block

    tail_free = (remaining // 256) * 2
    if tail_free > 0:
        _map_tile(x, y, tail_base, 128, tail_free)

    final_elements = remaining - 128 * tail_free
    if final_elements > 0:
        _map_tile(
            x,
            y,
            tail_base + 128 * tail_free,
            1,
            final_elements,
        )

    return y


def run(input: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG

    N = int(N)
    if tuple(input.shape) != (N,) or N % 2 != 0:
        raise ValueError("Expected an even-length input with shape (N,).")

    per_core = N // 2
    tile_free = 2048
    while tile_free < _MAX_FREE:
        next_free = tile_free * 2
        if per_core < 4 * _PARTITIONS * next_free:
            break
        tile_free = next_free

    block = _PARTITIONS * tile_free
    remaining = per_core % block
    tail_free = (remaining // 256) * 2

    _LAST_CONFIG = {
        "lnc": 2,
        "tile_partitions": _PARTITIONS,
        "tile_free": tile_free,
        "block_size": block,
        "elements_per_core": per_core,
        "full_blocks_per_core": per_core // block,
        "tail_partitions": _PARTITIONS if tail_free else 0,
        "tail_free": tail_free,
        "final_elements": remaining - _PARTITIONS * tail_free,
        "prefetch_auto": True,
        "separate_output_buffers": True,
        "engine": "vector",
        "formulation": "maximum(x, 0.01*x)",
        "negative_slope": 0.01,
    }

    return _leaky_relu[2](input, N, tile_free)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
