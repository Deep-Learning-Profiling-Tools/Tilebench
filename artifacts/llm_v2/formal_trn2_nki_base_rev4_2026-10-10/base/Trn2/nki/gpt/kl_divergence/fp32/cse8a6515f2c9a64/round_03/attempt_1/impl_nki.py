import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _prepare_tile(log_p, target, row, column, width):
    raw_y = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.sbuf)
    weight = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.sbuf)
    term = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.sbuf)
    log_y = nl.ndarray((128, width), dtype=nl.float32, buffer=nl.sbuf)

    nisa.dma_copy(
        dst=raw_y, src=target[row:row + 128, column:column + width]
    )
    nisa.dma_copy(
        dst=term, src=log_p[row:row + 128, column:column + width]
    )

    # Independent reads let the predicate and weight clamp use different
    # engines concurrently. Non-positive targets receive zero weight.
    nisa.tensor_scalar(
        dst=log_y, data=raw_y, op0=nl.less_equal, operand0=0.0
    )
    nisa.activation(dst=weight, op=nl.relu, data=raw_y)

    return weight, term, log_y


def _start_log(weight, log_y):
    # The argument is Y for positive Y, and exactly 1 otherwise.
    nisa.tensor_tensor(
        dst=log_y, data1=weight, data2=log_y, op=nl.add
    )
    nisa.activation(dst=log_y, op=nl.log, data=log_y)


def _form_term(weight, term, log_y):
    nisa.tensor_tensor(
        dst=term, data1=log_y, data2=term, op=nl.subtract
    )
    nisa.tensor_tensor(
        dst=term, data1=weight, data2=term, op=nl.multiply
    )


def _reduce_and_store(term, accumulator, loss_rows, row, column, width, cols):
    if column == 0:
        nisa.tensor_reduce(
            dst=accumulator, op=nl.add, data=term, axis=1
        )
    else:
        partial = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=partial, op=nl.add, data=term, axis=1)
        nisa.tensor_tensor(
            dst=accumulator,
            data1=accumulator,
            data2=partial,
            op=nl.add,
        )

    if column + width == cols:
        nisa.dma_copy(
            dst=loss_rows[row:row + 128, :], src=accumulator
        )


@nki.jit
def _kl_divergence(log_p, target, column_chunks):
    rows, cols = log_p.shape
    loss = nl.ndarray((rows,), dtype=nl.float32, buffer=nl.shared_hbm)
    loss_rows = loss.reshape((rows, 1))

    programs = nl.num_programs()
    pid = nl.program_id(0)
    rows_per_program = rows // programs
    first_row = pid * rows_per_program
    tile_count = (rows_per_program // 128) * column_chunks

    # Distribute 1024-column units evenly across the chunks.
    column_units = cols // 1024
    base_units = column_units // column_chunks
    extra_units = column_units % column_chunks

    previous_row = first_row
    previous_column = 0
    previous_width = base_units * 1024
    if extra_units > 0:
        previous_width = previous_width + 1024

    accumulator = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)

    previous_weight, previous_term, previous_log_y = _prepare_tile(
        log_p, target, previous_row, previous_column, previous_width
    )
    _start_log(previous_weight, previous_log_y)

    for tile_index in range(1, tile_count):
        chunk_index = tile_index % column_chunks
        current_row = first_row + (tile_index // column_chunks) * 128
        current_column = (
            chunk_index * base_units + min(chunk_index, extra_units)
        ) * 1024
        current_width = base_units * 1024
        if chunk_index < extra_units:
            current_width = current_width + 1024

        current_weight, current_term, current_log_y = _prepare_tile(
            log_p, target, current_row, current_column, current_width
        )

        # Finish the preceding map while the current weight clamp runs.
        _form_term(previous_weight, previous_term, previous_log_y)

        # Start the current logarithm before reducing the preceding tile.
        # Its ScalarE work overlaps the reduction and the next preparation.
        _start_log(current_weight, current_log_y)
        _reduce_and_store(
            previous_term,
            accumulator,
            loss_rows,
            previous_row,
            previous_column,
            previous_width,
            cols,
        )

        previous_weight = current_weight
        previous_term = current_term
        previous_log_y = current_log_y
        previous_row = current_row
        previous_column = current_column
        previous_width = current_width

    _form_term(previous_weight, previous_term, previous_log_y)
    _reduce_and_store(
        previous_term,
        accumulator,
        loss_rows,
        previous_row,
        previous_column,
        previous_width,
        cols,
    )

    return loss


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    global _LAST_CONFIG

    rows, cols = log_y_pred.shape
    column_chunks = (cols + 4095) // 4096
    column_units = cols // 1024
    base_units = column_units // column_chunks
    extra_units = column_units % column_chunks
    maximum_width = (base_units + int(extra_units > 0)) * 1024

    _LAST_CONFIG = {
        "lnc": 2,
        "row_tile": 128,
        "column_tile": maximum_width,
        "column_chunks": column_chunks,
        "column_chunk_base": base_units * 1024,
        "extra_column_chunks": extra_units,
        "row_tiles_per_core": rows // 256,
        "pipeline_depth": 2,
        "pipeline": "staggered_map_and_reduction",
        "guard_engine": "scalar_relu",
        "nonpositive_guard": True,
        "accumulation_dtype": "float32",
        "reduction": "chunked_row_sum",
    }

    return _kl_divergence[2](log_y_pred, y_true, column_chunks)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
