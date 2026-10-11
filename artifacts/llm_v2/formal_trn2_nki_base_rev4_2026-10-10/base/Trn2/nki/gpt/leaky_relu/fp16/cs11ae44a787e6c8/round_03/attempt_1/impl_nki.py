import torch
import nki
import nki.language as nl
import nki.isa as nisa


_PARTITIONS = 128
_MAX_FREE = 8192
_LAST_CONFIG = {}


def _tile_geometry(index, full_blocks, tile_free, tail_free, final_elements):
    block = 128 * tile_free
    if index < full_blocks:
        return index * block, 128, tile_free
    elif tail_free > 0 and index == full_blocks:
        return full_blocks * block, 128, tail_free
    else:
        return full_blocks * block + 128 * tail_free, 1, final_elements


@nki.jit
def _leaky_relu(x, N, tile_free):
    y = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    programs = nl.num_programs()
    pid = nl.program_id(0)
    per_program = N // programs
    base = pid * per_program

    block = 128 * tile_free
    full_blocks = per_program // block
    remaining = per_program - full_blocks * block
    tail_free = (remaining // 256) * 2
    final_elements = remaining - 128 * tail_free

    tile_count = full_blocks
    if tail_free > 0:
        tile_count += 1
    if final_elements > 0:
        tile_count += 1

    values = nl.ndarray((128, 2, tile_free), dtype=x.dtype, buffer=nl.sbuf)
    results = nl.ndarray((128, 2, tile_free), dtype=x.dtype, buffer=nl.sbuf)

    # Two input slots and separate output slots allow the next read to
    # precede the current write without overwriting a pending result.
    for i in range(min(2, tile_count)):
        offset, rows, cols = _tile_geometry(
            i, full_blocks, tile_free, tail_free, final_elements
        )
        nisa.dma_copy(
            dst=values[0:rows, i:i + 1, 0:cols],
            src=x.ap(
                pattern=[[cols, rows], [1, cols]],
                offset=base + offset,
            ),
        )

    for i in range(tile_count):
        slot = i % 2
        offset, rows, cols = _tile_geometry(
            i, full_blocks, tile_free, tail_free, final_elements
        )

        nisa.tensor_scalar(
            dst=results[0:rows, slot:slot + 1, 0:cols],
            data=values[0:rows, slot:slot + 1, 0:cols],
            op0=nl.multiply,
            operand0=0.01,
            engine=nisa.engine.vector,
        )
        nisa.tensor_tensor(
            dst=results[0:rows, slot:slot + 1, 0:cols],
            data1=values[0:rows, slot:slot + 1, 0:cols],
            data2=results[0:rows, slot:slot + 1, 0:cols],
            op=nl.maximum,
            engine=nisa.engine.vector,
        )

        if i + 2 < tile_count:
            next_offset, next_rows, next_cols = _tile_geometry(
                i + 2, full_blocks, tile_free, tail_free, final_elements
            )
            nisa.dma_copy(
                dst=values[0:next_rows, slot:slot + 1, 0:next_cols],
                src=x.ap(
                    pattern=[[next_cols, next_rows], [1, next_cols]],
                    offset=base + next_offset,
                ),
            )

        nisa.dma_copy(
            dst=y.ap(
                pattern=[[cols, rows], [1, cols]],
                offset=base + offset,
            ),
            src=results[0:rows, slot:slot + 1, 0:cols],
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
    final_elements = remaining - _PARTITIONS * tail_free

    _LAST_CONFIG = {
        "lnc": 2,
        "tile_partitions": _PARTITIONS,
        "tile_free": tile_free,
        "block_size": block,
        "elements_per_core": per_core,
        "full_blocks_per_core": per_core // block,
        "tail_partitions": _PARTITIONS if tail_free else 0,
        "tail_free": tail_free,
        "final_elements": final_elements,
        "pipeline_slots": 2,
        "separate_output_buffers": True,
        "engine": "vector",
        "formulation": "maximum(x, 0.01*x)",
        "negative_slope": 0.01,
    }

    return _leaky_relu[2](input, N, tile_free)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
