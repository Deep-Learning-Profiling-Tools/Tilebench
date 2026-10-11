import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _dequantize_rowwise(x, state_x, groups_per_tile):
    rows, cols = x.shape
    cores = nl.num_programs()
    pid = nl.program_id(0)
    partitions = 128
    groups_per_partition = rows // (cores * partitions)

    out = nl.ndarray(x.shape, dtype=nl.float16, buffer=nl.shared_hbm)

    x_core = x.reshape(
        (cores, partitions, groups_per_partition, cols)
    )[pid]
    out_core = out.reshape(
        (cores, partitions, groups_per_partition, cols)
    )[pid]
    state_core = state_x.reshape(
        (cores, partitions, groups_per_partition)
    )[pid]

    # Load every scale assigned to this core in one transfer.
    scales = nl.ndarray(
        (partitions, groups_per_partition),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(dst=scales, src=state_core)

    blocks = (
        groups_per_partition + groups_per_tile - 1
    ) // groups_per_tile

    first_groups = min(groups_per_tile, groups_per_partition)
    current = nl.ndarray(
        (partitions, first_groups, cols),
        dtype=nl.float16,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=current,
        src=x_core[:, 0:first_groups, :],
    )

    for block in range(blocks):
        g0 = block * groups_per_tile
        group_count = min(
            groups_per_tile, groups_per_partition - g0
        )

        result = nl.ndarray(
            (partitions, group_count, cols),
            dtype=nl.float16,
            buffer=nl.sbuf,
        )

        for group in range(group_count):
            scale_index = g0 + group

            # Int8 values are represented exactly by the fp16 staging
            # tile. VectorE converts them to fp32, performs both
            # multiplies, and rounds only the final product to fp16.
            nisa.tensor_scalar(
                dst=result[:, group, :],
                data=current[:, group, :],
                op0=nl.multiply,
                operand0=scales[:, scale_index:scale_index + 1],
                op1=nl.multiply,
                operand1=1.0 / 127.0,
                engine=nisa.engine.vector,
            )

        # Issue the next input transfer ahead of the current output
        # transfer, permitting input DMA to overlap vector computation.
        if block + 1 < blocks:
            next_g0 = g0 + groups_per_tile
            next_groups = min(
                groups_per_tile,
                groups_per_partition - next_g0,
            )
            next_values = nl.ndarray(
                (partitions, next_groups, cols),
                dtype=nl.float16,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=next_values,
                src=x_core[:, next_g0:next_g0 + next_groups, :],
            )

        nisa.dma_copy(
            dst=out_core[:, g0:g0 + group_count, :],
            src=result,
        )

        if block + 1 < blocks:
            current = next_values

    return out


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    global _LAST_CONFIG
    rows, cols = x.shape

    groups_per_partition = rows // 256
    groups_per_tile = min(
        groups_per_partition, max(1, 4096 // cols)
    )

    _LAST_CONFIG = {
        "kernel": "dequantize_rowwise",
        "lnc": 2,
        "partition_rows": 128,
        "groups_per_partition": int(groups_per_partition),
        "groups_per_tile": int(groups_per_tile),
        "tile_rows": int(128 * groups_per_tile),
        "tile_cols": int(cols),
        "rows": int(rows),
        "cols": int(cols),
        "scale_loads_per_core": 1,
        "prefetch_tiles": 1,
        "input_staging_dtype": "float16",
        "arithmetic_dtype": "float32",
        "output_dtype": "float16",
    }

    return _dequantize_rowwise[2](x, state_x, groups_per_tile)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
