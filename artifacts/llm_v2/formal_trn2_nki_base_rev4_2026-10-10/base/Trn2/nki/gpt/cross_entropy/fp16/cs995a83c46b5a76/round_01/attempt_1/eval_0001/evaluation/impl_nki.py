import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _cross_entropy(logits, targets, row_tile):
    batch, classes = logits.shape
    out = nl.ndarray((batch,), dtype=logits.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    rows_per_program = batch // programs

    for block in range((rows_per_program + row_tile - 1) // row_tile):
        local_start = block * row_tile
        rows = min(row_tile, rows_per_program - local_start)
        start = pid * rows_per_program + local_start

        x = nl.ndarray((rows, classes), dtype=nl.float32, buffer=nl.sbuf)
        indices = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        target_logit = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)

        nisa.dma_copy(dst=x, src=logits[start:start + rows, :])
        nisa.dma_copy(dst=indices, src=targets[start:start + rows])

        # Keep the row stride affine; indirectly address one class per row.
        target_source = logits.ap(
            pattern=[[classes, rows], [1, 1]],
            offset=start * classes,
            vector_offset=indices,
            indirect_dim=1,
        )
        nisa.dma_copy(dst=target_logit, src=target_source)

        negative_max = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=negative_max,
            op=nl.maximum,
            data=x,
            axis=1,
            negate=True,
        )

        exponentials = nl.ndarray(
            (rows, classes), dtype=nl.float32, buffer=nl.sbuf
        )
        exp_sum = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=exponentials,
            op=nl.exp,
            data=x,
            bias=negative_max,
            reduce_op=nl.add,
            reduce_res=exp_sum,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        log_sum = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=log_sum, op=nl.log, data=exp_sum)

        shifted_target = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=shifted_target,
            data1=target_logit,
            data2=negative_max,
            op=nl.add,
        )

        loss = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=loss,
            data1=log_sum,
            data2=shifted_target,
            op=nl.subtract,
        )

        # The only rounding of the loss to fp16 occurs here.
        nisa.dma_copy(dst=out[start:start + rows], src=loss)

    return out


def run(logits: torch.Tensor, targets: torch.Tensor):
    global _LAST_CONFIG
    batch, classes = logits.shape
    row_tile = 128
    _LAST_CONFIG = {
        "row_tile": row_tile,
        "class_tile": classes,
        "lnc": 2,
        "rows_per_program": batch // 2,
        "blocks_per_program": (batch // 2 + row_tile - 1) // row_tile,
        "accumulation_dtype": "float32",
        "target_gather": "indirect_dma",
        "exp_sum": "fused_activation_reduce",
    }
    return _cross_entropy[2](logits, targets, row_tile)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
