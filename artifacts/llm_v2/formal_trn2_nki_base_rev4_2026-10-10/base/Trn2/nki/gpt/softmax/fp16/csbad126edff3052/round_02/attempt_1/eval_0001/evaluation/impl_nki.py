import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _softmax_pipelined(x, row_tile):
    rows, cols = x.shape
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    rows_per_program = rows // programs
    tile_count = rows_per_program // row_tile
    row_base = pid * rows_per_program

    # Three independently reusable slots. Each fp16 slot first stages
    # input and subsequently stages the final, rounded output.
    io = nl.ndarray(
        (row_tile, 3, cols), dtype=x.dtype, buffer=nl.sbuf
    )
    values = nl.ndarray(
        (row_tile, 3, cols), dtype=nl.float32, buffer=nl.sbuf
    )
    negative_max = nl.ndarray(
        (row_tile, 3, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    denominator = nl.ndarray(
        (row_tile, 3, 1), dtype=nl.float32, buffer=nl.sbuf
    )
    inverse = nl.ndarray(
        (row_tile, 3, 1), dtype=nl.float32, buffer=nl.sbuf
    )

    # Prepare the first resident row tile.
    nisa.dma_copy(
        dst=io[:, 0, :],
        src=x[row_base:row_base + row_tile, :],
    )
    nisa.tensor_copy(dst=values[:, 0, :], src=io[:, 0, :])
    nisa.tensor_reduce(
        dst=negative_max[:, 0, :],
        op=nl.maximum,
        data=values[:, 0, :],
        axis=1,
        negate=True,
    )

    for tile in range(tile_count):
        slot = tile % 3
        next_slot = (tile + 1) % 3
        r0 = row_base + tile * row_tile

        if tile + 1 < tile_count:
            next_r0 = row_base + (tile + 1) * row_tile
            nisa.dma_copy(
                dst=io[:, next_slot, :],
                src=x[next_r0:next_r0 + row_tile, :],
            )

        # The whole row is resident, so its final maximum is already
        # known. Retain fp32 exponentials in place for normalization.
        nisa.activation(
            dst=values[:, slot, :],
            op=nl.exp,
            data=values[:, slot, :],
            bias=negative_max[:, slot, :],
            reduce_op=nl.add,
            reduce_res=denominator[:, slot, :],
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        # Prepare the following tile on VectorE while ScalarE evaluates
        # the current tile's exponentials and denominator.
        if tile + 1 < tile_count:
            nisa.tensor_copy(
                dst=values[:, next_slot, :],
                src=io[:, next_slot, :],
            )
            nisa.tensor_reduce(
                dst=negative_max[:, next_slot, :],
                op=nl.maximum,
                data=values[:, next_slot, :],
                axis=1,
                negate=True,
            )

        nisa.reciprocal(
            dst=inverse[:, slot, :],
            data=denominator[:, slot, :],
        )

        # Arithmetic is fp32. Only the final normalized results are
        # downcast; the following DMA stores their bits unchanged.
        nisa.tensor_scalar(
            dst=io[:, slot, :],
            data=values[:, slot, :],
            op0=nl.multiply,
            operand0=inverse[:, slot, :],
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(
            dst=out[r0:r0 + row_tile, :],
            src=io[:, slot, :],
        )

    return out


def run(x):
    global _LAST_CONFIG
    rows, cols = x.shape
    _LAST_CONFIG = {
        "algorithm": "resident_row_softmax",
        "lnc": 2,
        "row_tile": 128,
        "column_tile": int(cols),
        "staging_slots": 3,
        "rows_per_core": int(rows) // 2,
        "statistics_dtype": "float32",
        "fused_exp_sum": True,
        "inplace_exponentials": True,
        "prefetch_next_tile": True,
        "normalization_output_dtype": str(x.dtype),
    }
    return _softmax_pipelined[2](x, 128)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
