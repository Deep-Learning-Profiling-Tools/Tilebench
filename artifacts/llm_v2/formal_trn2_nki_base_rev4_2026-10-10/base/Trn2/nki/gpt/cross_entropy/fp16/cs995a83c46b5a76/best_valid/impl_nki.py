import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _cross_entropy(logits, targets, row_tile, statistics_rows):
    batch, classes = logits.shape
    pid = nl.program_id(0)
    programs = nl.num_programs()
    rows_per_program = batch // programs
    rows_per_lane = rows_per_program // row_tile
    start = pid * rows_per_program

    out = nl.ndarray((batch,), dtype=logits.dtype, buffer=nl.shared_hbm)
    flat_logits = logits.reshape((batch * classes, 1))

    x = nl.ndarray(
        (row_tile, rows_per_lane, classes),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    indices = nl.ndarray(
        (row_tile, rows_per_lane), dtype=nl.int32, buffer=nl.sbuf
    )
    offsets = nl.ndarray(
        (row_tile, rows_per_lane), dtype=nl.int32, buffer=nl.sbuf
    )

    nisa.dma_copy(
        dst=x,
        src=logits[start:start + rows_per_program, :].reshape(
            (row_tile, rows_per_lane, classes)
        ),
    )
    nisa.dma_copy(
        dst=indices,
        src=targets[start:start + rows_per_program].reshape(
            (row_tile, rows_per_lane)
        ),
    )
    nisa.iota(
        dst=offsets,
        pattern=[[classes, rows_per_lane]],
        offset=start * classes,
        channel_multiplier=rows_per_lane * classes,
    )
    nisa.tensor_tensor(
        dst=offsets, data1=offsets, data2=indices, op=nl.add
    )

    negative_max = nl.ndarray(
        (row_tile, rows_per_lane), dtype=nl.float32, buffer=nl.sbuf
    )
    target_logit = nl.ndarray(
        (row_tile, rows_per_lane), dtype=nl.float32, buffer=nl.sbuf
    )
    exp_sum = nl.ndarray(
        (row_tile, rows_per_lane), dtype=nl.float32, buffer=nl.sbuf
    )

    # Distinct destinations avoid cross-row scratch-buffer dependencies.
    exponentials = nl.ndarray(
        (row_tile, rows_per_lane, classes),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )

    # Small groups let VectorE maxima overlap subsequent ScalarE work.
    for group in range(
        (rows_per_lane + statistics_rows - 1) // statistics_rows
    ):
        first = group * statistics_rows
        count = min(statistics_rows, rows_per_lane - first)

        nisa.tensor_reduce(
            dst=negative_max[:, first:first + count],
            op=nl.maximum,
            data=x[:, first:first + count, :],
            axis=2,
            negate=True,
        )

        for local_row in range(count):
            row = first + local_row

            target_source = flat_logits.ap(
                pattern=[[1, row_tile], [1, 1]],
                offset=0,
                vector_offset=offsets[:, row:row + 1],
                indirect_dim=0,
            )
            nisa.dma_copy(
                dst=target_logit[:, row:row + 1],
                src=target_source,
            )

            nisa.activation(
                dst=exponentials[:, row, :],
                op=nl.exp,
                data=x[:, row, :],
                bias=negative_max[:, row:row + 1],
                reduce_op=nl.add,
                reduce_res=exp_sum[:, row:row + 1],
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )

    log_sum = nl.ndarray(
        (row_tile, rows_per_lane), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.activation(dst=log_sum, op=nl.log, data=exp_sum)

    nisa.tensor_tensor(
        dst=target_logit,
        data1=target_logit,
        data2=negative_max,
        op=nl.add,
    )
    loss = nl.ndarray(
        (row_tile, rows_per_lane), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.tensor_tensor(
        dst=loss,
        data1=log_sum,
        data2=target_logit,
        op=nl.subtract,
    )

    nisa.dma_copy(
        dst=out[start:start + rows_per_program].reshape(
            (row_tile, rows_per_lane)
        ),
        src=loss,
    )
    return out


def run(logits: torch.Tensor, targets: torch.Tensor):
    global _LAST_CONFIG
    batch, classes = logits.shape
    row_tile = 128
    statistics_rows = 4
    rows_per_program = batch // 2

    _LAST_CONFIG = {
        "row_tile": row_tile,
        "class_tile": classes,
        "lnc": 2,
        "rows_per_program": rows_per_program,
        "rows_per_lane": rows_per_program // row_tile,
        "statistics_rows": statistics_rows,
        "input_layout": "contiguous_bulk_rows_per_partition",
        "max_reduction": "pipelined_grouped_axis_2",
        "target_gather": "flattened_indirect_dma_bulk_offsets",
        "exp_sum": "fused_activation_reduce",
        "exponential_scratch": "distinct_per_row",
        "accumulation_dtype": "float32",
        "output_store": "contiguous_bulk",
    }
    return _cross_entropy[2](
        logits, targets, row_tile, statistics_rows
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
