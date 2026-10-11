import torch
import nki
import nki.compiler
import nki.language as nl
import nki.isa as nisa


_PARTITIONS = 125
_MIN_FREE = 2500
_MAX_FREE = 10000
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


@nki.jit(compile_config=nki.compiler.CompileConfig(prefetch_auto=True))
def _leaky_relu(x, N, partitions, tile_free):
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    per_program = N // nl.num_programs()
    base = nl.program_id(0) * per_program
    block = partitions * tile_free
    full_blocks = per_program // block

    # Independent input/output tiles permit overlapping DMA and computation.
    for i in range(full_blocks):
        _map_tile(x, y, base + i * block, partitions, tile_free)

    remaining = per_program - full_blocks * block
    tail_base = base + full_blocks * block
    tail_free = (remaining // (2 * partitions)) * 2

    if tail_free > 0:
        _map_tile(x, y, tail_base, partitions, tail_free)

    final_elements = remaining - partitions * tail_free
    if final_elements > 0:
        _map_tile(
            x,
            y,
            tail_base + partitions * tail_free,
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
    tile_free = _MIN_FREE
    while tile_free < _MAX_FREE:
        next_free = tile_free * 2
        if per_core < 4 * _PARTITIONS * next_free:
            break
        tile_free = next_free

    block = _PARTITIONS * tile_free
    remaining = per_core % block
    tail_free = (remaining // (2 * _PARTITIONS)) * 2

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

    return _leaky_relu[2](input, N, _PARTITIONS, tile_free)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
