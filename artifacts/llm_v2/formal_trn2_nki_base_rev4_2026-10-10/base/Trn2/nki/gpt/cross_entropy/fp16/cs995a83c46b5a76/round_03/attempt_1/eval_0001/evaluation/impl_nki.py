import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _cross_entropy(logits, targets, row_tile):
    batch, classes = logits.shape
    programs = nl.num_programs()
    pid = nl.program_id(0)
    rows_per_program = batch // programs
    rows_per_lane = rows_per_program // row_tile
    start = pid * rows_per_program

    out = nl.ndarray((batch,), dtype=logits.dtype, buffer=nl.shared_hbm)

    # Each partition holds consecutive complete rows. This makes the
    # input, target, and output transfers contiguous bulk transfers.
    x = nl.ndarray(
        (row_tile, rows_per_lane, classes),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    target_indices = nl.ndarray(
        (row_tile, rows_per_lane),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=x,
        src=logits[start:start + rows_per_program, :].reshape(
            (row_tile, rows_per_lane, classes)
        ),
    )
    nisa.dma_copy(
        dst=target_indices,
        src=targets[start:start + rows_per_program].reshape(
            (row_tile, rows_per_lane)
        ),
    )

    columns = nl.ndarray(
        (row_tile, classes), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.iota(
        dst=columns,
        pattern=[[1, classes]],
        offset=0,
        channel_multiplier=0,
    )

    matches = nl.ndarray(
        (row_tile, classes), dtype=nl.uint8, buffer=nl.sbuf
    )
    zeros = nl.ndarray(
        (row_tile, classes), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.memset(dst=zeros, value=0.0)

    negative_max = nl.ndarray(
        (row_tile, rows_per_lane), dtype=nl.float32, buffer=nl.sbuf
    )
    exp_sum = nl.ndarray(
        (row_tile, rows_per_lane), dtype=nl.float32, buffer=nl.sbuf
    )
    target_logit = nl.ndarray(
        (row_tile, rows_per_lane), dtype=nl.float32, buffer=nl.sbuf
    )
    exponentials = nl.ndarray(
        (row_tile, classes), dtype=nl.float32, buffer=nl.sbuf
    )

    for row in range(rows_per_lane):
        nisa.tensor_reduce(
            dst=negative_max[:, row:row + 1],
            op=nl.maximum,
            data=x[:, row, :],
            axis=1,
            negate=True,
        )
        nisa.activation(
            dst=exponentials,
            op=nl.exp,
            data=x[:, row, :],
            bias=negative_max[:, row:row + 1],
            reduce_op=nl.add,
            reduce_res=exp_sum[:, row:row + 1],
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        # Extract the target from the resident row using a one-hot select.
        # Selection, rather than multiplication, also handles -inf logits.
        nisa.tensor_scalar(
            dst=matches,
            data=columns,
            op0=nl.equal,
            operand0=target_indices[:, row:row + 1],
        )
        selected = nl.where(matches, x[:, row, :], zeros)
        nisa.tensor_reduce(
            dst=target_logit[:, row:row + 1],
            op=nl.add,
            data=selected,
            axis=1,
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

    # The completed fp32 loss is rounded only at the output store.
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
    rows_per_program = batch // 2
    _LAST_CONFIG = {
        "row_tile": row_tile,
        "class_tile": classes,
        "lnc": 2,
        "rows_per_program": rows_per_program,
        "rows_per_lane": rows_per_program // row_tile,
        "input_layout": "contiguous_bulk_rows_per_partition",
        "target_gather": "on_chip_one_hot_selection",
        "exp_sum": "fused_activation_reduce",
        "accumulation_dtype": "float32",
        "output_store": "contiguous_bulk",
    }
    return _cross_entropy[2](logits, targets, row_tile)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
