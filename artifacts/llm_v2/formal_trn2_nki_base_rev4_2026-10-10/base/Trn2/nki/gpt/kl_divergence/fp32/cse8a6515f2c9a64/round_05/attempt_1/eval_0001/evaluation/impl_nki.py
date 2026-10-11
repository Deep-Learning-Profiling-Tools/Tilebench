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

    # These independent reads use separate engines, leaving ScalarE
    # available for logarithms of preceding tiles.
    nisa.tensor_scalar(
        dst=log_y,
        data=raw_y,
        op0=nl.less_equal,
        operand0=0.0,
        engine=nisa.engine.gpsimd,
    )
    nisa.tensor_scalar(
        dst=weight,
        data=raw_y,
        op0=nl.maximum,
        operand0=0.0,
        engine=nisa.engine.vector,
    )
    return weight, term, log_y


def _prepare_index(log_p, target, first_row, index, chunks, base, extra):
    chunk_index = index % chunks
    row = first_row + (index // chunks) * 128
    column = (chunk_index * base + min(chunk_index, extra)) * 1024
    width = base * 1024
    if chunk_index < extra:
        width = width + 1024

    weight, term, log_y = _prepare_tile(
        log_p, target, row, column, width
    )
    return weight, term, log_y, row, column, width


def _start_log(weight, log_y):
    # Positive targets retain their exact fp32 value. Non-positive targets
    # have zero weight and receive log(1), avoiding log(0) and log(negative).
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

    column_units = cols // 1024
    base_units = column_units // column_chunks
    extra_units = column_units % column_chunks

    accumulator = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)

    oldest_weight, oldest_term, oldest_log_y, oldest_row, oldest_column, oldest_width = _prepare_index(
        log_p, target, first_row, 0, column_chunks, base_units, extra_units
    )
    _start_log(oldest_weight, oldest_log_y)

    next_weight, next_term, next_log_y, next_row, next_column, next_width = _prepare_index(
        log_p, target, first_row, 1, column_chunks, base_units, extra_units
    )
    _start_log(next_weight, next_log_y)

    # Keep logarithms two tiles ahead of term formation and reduction.
    for tile_index in range(2, tile_count):
        current_weight, current_term, current_log_y, current_row, current_column, current_width = _prepare_index(
            log_p,
            target,
            first_row,
            tile_index,
            column_chunks,
            base_units,
            extra_units,
        )
        _start_log(current_weight, current_log_y)

        _form_term(oldest_weight, oldest_term, oldest_log_y)
        _reduce_and_store(
            oldest_term,
            accumulator,
            loss_rows,
            oldest_row,
            oldest_column,
            oldest_width,
            cols,
        )

        oldest_weight = next_weight
        oldest_term = next_term
        oldest_log_y = next_log_y
        oldest_row = next_row
        oldest_column = next_column
        oldest_width = next_width

        next_weight = current_weight
        next_term = current_term
        next_log_y = current_log_y
        next_row = current_row
        next_column = current_column
        next_width = current_width

    _form_term(oldest_weight, oldest_term, oldest_log_y)
    _reduce_and_store(
        oldest_term,
        accumulator,
        loss_rows,
        oldest_row,
        oldest_column,
        oldest_width,
        cols,
    )

    _form_term(next_weight, next_term, next_log_y)
    _reduce_and_store(
        next_term,
        accumulator,
        loss_rows,
        next_row,
        next_column,
        next_width,
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
        "pipeline_depth": 3,
        "pipeline": "two_logs_ahead",
        "predicate_engine": "gpsimd",
        "weight_clamp_engine": "vector",
        "log_engine": "scalar",
        "nonpositive_guard": True,
        "accumulation_dtype": "float32",
        "reduction": "chunked_row_sum",
    }

    return _kl_divergence[2](log_y_pred, y_true, column_chunks)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
