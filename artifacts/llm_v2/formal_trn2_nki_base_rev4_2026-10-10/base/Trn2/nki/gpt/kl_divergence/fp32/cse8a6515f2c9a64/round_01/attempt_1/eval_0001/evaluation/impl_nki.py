import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _kl_divergence(log_p, target, column_tile):
    rows, cols = log_p.shape
    row_tile = 128

    loss = nl.ndarray((rows,), dtype=nl.float32, buffer=nl.shared_hbm)
    loss_rows = loss.reshape((rows, 1))

    programs = nl.num_programs()
    pid = nl.program_id(0)
    rows_per_program = rows // programs
    first_row = pid * rows_per_program

    for r0 in range(first_row, first_row + rows_per_program, row_tile):
        accumulator = nl.ndarray(
            (row_tile, 1), dtype=nl.float32, buffer=nl.sbuf
        )

        for c0 in range(0, cols, column_tile):
            width = min(column_tile, cols - c0)

            y = nl.ndarray(
                (row_tile, width), dtype=nl.float32, buffer=nl.sbuf
            )
            p = nl.ndarray(
                (row_tile, width), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=y, src=target[r0:r0 + row_tile, c0:c0 + width]
            )
            nisa.dma_copy(
                dst=p, src=log_p[r0:r0 + row_tile, c0:c0 + width]
            )

            # Guard the logarithm before multiplication. Non-positive
            # targets use log(1), and their multiplication weight is zero.
            positive = nl.greater(y, 0.0)
            log_y = nl.where(positive, y, 1.0)
            nisa.activation(dst=log_y, op=nl.log, data=log_y)
            nisa.tensor_scalar(
                dst=y, data=y, op0=nl.maximum, operand0=0.0
            )

            nisa.tensor_tensor(
                dst=p, data1=log_y, data2=p, op=nl.subtract
            )
            nisa.tensor_tensor(
                dst=p, data1=y, data2=p, op=nl.multiply
            )

            if c0 == 0:
                nisa.tensor_reduce(
                    dst=accumulator, op=nl.add, data=p, axis=1
                )
            else:
                partial = nl.ndarray(
                    (row_tile, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_reduce(
                    dst=partial, op=nl.add, data=p, axis=1
                )
                nisa.tensor_tensor(
                    dst=accumulator,
                    data1=accumulator,
                    data2=partial,
                    op=nl.add,
                )

        nisa.dma_copy(
            dst=loss_rows[r0:r0 + row_tile, :], src=accumulator
        )

    return loss


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    global _LAST_CONFIG

    rows, cols = log_y_pred.shape
    column_tile = min(cols, 4096)
    _LAST_CONFIG = {
        "lnc": 2,
        "row_tile": 128,
        "column_tile": column_tile,
        "column_chunks": (cols + column_tile - 1) // column_tile,
        "row_tiles_per_core": rows // 256,
        "accumulation_dtype": "float32",
        "reduction": "chunked_row_sum",
        "nonpositive_guard": True,
    }

    return _kl_divergence[2](log_y_pred, y_true, column_tile)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
