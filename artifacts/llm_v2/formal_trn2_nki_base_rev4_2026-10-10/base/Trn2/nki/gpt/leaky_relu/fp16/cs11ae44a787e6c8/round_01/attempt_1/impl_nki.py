import torch
import nki
import nki.language as nl
import nki.isa as nisa


_TILE_PARTITIONS = 128
_TILE_FREE = 4096
_LAST_CONFIG = {}


def _map_tile(x, y, offset, rows, cols):
    value = nl.ndarray((rows, cols), dtype=x.dtype, buffer=nl.sbuf)
    scaled = nl.ndarray((rows, cols), dtype=x.dtype, buffer=nl.sbuf)

    nisa.dma_copy(
        dst=value,
        src=x.ap(pattern=[[cols, rows], [1, cols]], offset=offset),
    )
    nisa.tensor_scalar(
        dst=scaled,
        data=value,
        op0=nl.multiply,
        operand0=0.01,
        engine=nisa.engine.vector,
    )
    nisa.tensor_tensor(
        dst=value,
        data1=value,
        data2=scaled,
        op=nl.maximum,
        engine=nisa.engine.vector,
    )
    nisa.dma_copy(
        dst=y.ap(pattern=[[cols, rows], [1, cols]], offset=offset),
        src=value,
    )


@nki.jit
def _leaky_relu(x, N, tile_free):
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    programs = nl.num_programs()
    pid = nl.program_id(0)
    elements_per_program = N // programs
    base = pid * elements_per_program

    block = 128 * tile_free
    full_blocks = elements_per_program // block

    for i in range(full_blocks):
        _map_tile(x, y, base + i * block, 128, tile_free)

    remaining = elements_per_program - full_blocks * block
    tail_base = base + full_blocks * block
    tail_rows = remaining // tile_free
    tail_cols = remaining % tile_free

    if tail_rows > 0:
        _map_tile(x, y, tail_base, tail_rows, tile_free)

    if tail_cols > 0:
        _map_tile(
            x,
            y,
            tail_base + tail_rows * tile_free,
            1,
            tail_cols,
        )

    return y


def run(input: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG

    N = int(N)
    if tuple(input.shape) != (N,) or N % 2 != 0:
        raise ValueError("Expected an even-length input with shape (N,).")

    per_core = N // 2
    block = _TILE_PARTITIONS * _TILE_FREE
    remaining = per_core % block

    _LAST_CONFIG = {
        "lnc": 2,
        "tile_partitions": _TILE_PARTITIONS,
        "tile_free": _TILE_FREE,
        "block_size": block,
        "elements_per_core": per_core,
        "full_blocks_per_core": per_core // block,
        "tail_rows": remaining // _TILE_FREE,
        "tail_cols": remaining % _TILE_FREE,
        "engine": "vector",
        "formulation": "maximum(x, 0.01*x)",
        "negative_slope": 0.01,
    }

    return _leaky_relu[2](input, N, _TILE_FREE)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
