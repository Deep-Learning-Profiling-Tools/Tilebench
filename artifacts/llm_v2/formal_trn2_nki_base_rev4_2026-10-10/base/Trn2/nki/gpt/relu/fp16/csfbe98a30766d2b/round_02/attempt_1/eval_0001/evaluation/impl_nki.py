import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _relu_kernel(x, free_size, pipeline_depth):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    partitions = 128
    programs = nl.num_programs()
    pid = nl.program_id(0)
    tiles_per_core = x.shape[0] // (programs * partitions * free_size)
    stages = min(pipeline_depth, tiles_per_core)

    tiled_shape = (programs, tiles_per_core, partitions, free_size)
    src = x.reshape(tiled_shape)
    dst = out.reshape(tiled_shape)

    inputs = nl.ndarray(
        (partitions, stages, free_size), dtype=x.dtype, buffer=nl.sbuf
    )
    outputs = nl.ndarray(
        (partitions, stages, free_size), dtype=x.dtype, buffer=nl.sbuf
    )

    for i in range(stages):
        nisa.dma_copy(dst=inputs[:, i, :], src=src[pid, i])

    for i in range(tiles_per_core):
        slot = i % stages
        nisa.tensor_scalar(
            dst=outputs[:, slot, :],
            data=inputs[:, slot, :],
            op0=nl.maximum,
            operand0=0.0,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(dst=dst[pid, i], src=outputs[:, slot, :])

        if i + stages < tiles_per_core:
            nisa.dma_copy(
                dst=inputs[:, slot, :],
                src=src[pid, i + stages],
            )

    return out


def run(x: torch.Tensor):
    global _LAST_CONFIG

    n = x.numel()
    partitions = 128
    programs = 2
    pipeline_depth = 4

    free_size = 1024
    while (
        free_size < 4096
        and n >= programs * pipeline_depth * partitions * (2 * free_size)
    ):
        free_size *= 2

    assert n % (programs * partitions * free_size) == 0
    tiles_per_core = n // (programs * partitions * free_size)

    result = _relu_kernel[2](x, free_size, pipeline_depth)
    _LAST_CONFIG = {
        "n": n,
        "lnc": programs,
        "partition_size": partitions,
        "free_size": free_size,
        "tile_elements": partitions * free_size,
        "tiles_per_core": tiles_per_core,
        "pipeline_depth": min(pipeline_depth, tiles_per_core),
        "separate_input_output_buffers": True,
        "engine": "vector",
        "operation": "maximum_zero",
    }
    return result


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
