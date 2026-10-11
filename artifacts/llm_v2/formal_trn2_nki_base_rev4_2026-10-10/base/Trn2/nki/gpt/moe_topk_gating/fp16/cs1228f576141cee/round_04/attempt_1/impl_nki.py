import torch
import nki
import nki.language as nl
import nki.isa as nisa

_NEG_INF = float("-inf")
_LAST_CONFIG = {}


@nki.jit
def _moe_topk_gating(logits, k, expert_width, selected_width, groups):
    M, E = logits.shape
    P = 128
    pid = nl.program_id(0)
    rows_per_core = M // nl.num_programs()
    batch_rows = P * groups

    weights = nl.ndarray(
        (M, k), dtype=logits.dtype, buffer=nl.shared_hbm
    )
    indices = nl.ndarray(
        (M, k), dtype=nl.int32, buffer=nl.shared_hbm
    )

    columns = nl.ndarray(
        (P, expert_width), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.iota(
        dst=columns,
        pattern=[[1, expert_width]],
        offset=0,
        channel_multiplier=0,
    )

    excluded = nl.ndarray(
        (P, expert_width), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.memset(dst=excluded, value=_NEG_INF)

    for block in range(rows_per_core // batch_rows):
        r0 = pid * rows_per_core + block * batch_rows

        # Partition p contains row p from each of the groups.
        loaded = nl.ndarray(
            (P, groups, expert_width),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        if expert_width > E:
            nisa.memset(dst=loaded, value=_NEG_INF)
        nisa.dma_copy(
            dst=loaded[:, :, 0:E],
            src=logits.ap(
                pattern=[[E, P], [P * E, groups], [1, E]],
                offset=r0 * E,
            ),
        )

        selected = nl.ndarray(
            (P, groups, selected_width),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        selected_indices = nl.ndarray(
            (P, groups, selected_width),
            dtype=nl.int32,
            buffer=nl.sbuf,
        )
        stored_weights = nl.ndarray(
            (P, groups, selected_width),
            dtype=logits.dtype,
            buffer=nl.sbuf,
        )

        if selected_width > k:
            nisa.memset(dst=selected, value=_NEG_INF)
            nisa.memset(dst=selected_indices, value=0)

        for group in range(groups):
            remaining = loaded[:, group, :]

            for slot in range(k):
                maximum = nl.ndarray(
                    (P, 1), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_reduce(
                    dst=maximum,
                    op=nl.maximum,
                    data=remaining,
                    axis=1,
                )
                nisa.tensor_copy(
                    dst=selected[:, group, slot:slot + 1],
                    src=maximum,
                )

                matches = nl.ndarray(
                    (P, expert_width),
                    dtype=nl.uint8,
                    buffer=nl.sbuf,
                )
                nisa.tensor_scalar(
                    dst=matches,
                    data=remaining,
                    op0=nl.equal,
                    operand0=maximum,
                    engine=nisa.engine.vector,
                )

                matching_columns = nl.ndarray(
                    (P, expert_width),
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                nisa.tensor_tensor(
                    dst=matching_columns,
                    data1=columns,
                    data2=matches,
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )

                maximum_index = nl.ndarray(
                    (P, 1), dtype=nl.int32, buffer=nl.sbuf
                )
                nisa.tensor_reduce(
                    dst=maximum_index,
                    op=nl.maximum,
                    data=matching_columns,
                    axis=1,
                )
                nisa.tensor_copy(
                    dst=selected_indices[:, group, slot:slot + 1],
                    src=maximum_index,
                )

                # Exact selection of -inf avoids arithmetic with infinities.
                remaining = nl.where(matches, excluded, remaining)

            negative_maximum = nl.ndarray(
                (P, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_reduce(
                dst=negative_maximum,
                op=nl.maximum,
                data=selected[:, group, :],
                axis=1,
                negate=True,
            )

            exponentials = nl.ndarray(
                (P, selected_width),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            denominator = nl.ndarray(
                (P, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.activation(
                dst=exponentials,
                op=nl.exp,
                data=selected[:, group, :],
                bias=negative_maximum,
                reduce_op=nl.add,
                reduce_res=denominator,
                reduce_cmd=nisa.reduce_cmd.reset_reduce,
            )

            inverse_denominator = nl.ndarray(
                (P, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.reciprocal(
                dst=inverse_denominator,
                data=denominator,
            )
            nisa.tensor_scalar(
                dst=stored_weights[:, group, :],
                data=exponentials,
                op0=nl.multiply,
                operand0=inverse_denominator,
                engine=nisa.engine.vector,
            )

        # Amortize both output transfers across the grouped rows.
        nisa.dma_copy(
            dst=weights.ap(
                pattern=[[k, P], [P * k, groups], [1, k]],
                offset=r0 * k,
            ),
            src=stored_weights[:, :, 0:k],
        )
        nisa.dma_copy(
            dst=indices.ap(
                pattern=[[k, P], [P * k, groups], [1, k]],
                offset=r0 * k,
            ),
            src=selected_indices[:, :, 0:k],
        )

    return weights, indices


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    global _LAST_CONFIG

    assert logits.shape == (M, E)
    assert M % 1024 == 0
    assert 0 < k <= E

    expert_width = 1 << (E - 1).bit_length()
    selected_width = 1 << (k - 1).bit_length()
    groups = 4

    _LAST_CONFIG = {
        "lnc": 2,
        "rows_per_tile": 128,
        "groups_per_batch": groups,
        "rows_per_batch": 128 * groups,
        "rows_per_core": M // 2,
        "expert_width": expert_width,
        "selected_width": selected_width,
        "selection_rounds": k,
        "selection": "sequential_max_argmax_exclusion",
        "exclusion": "where_negative_infinity",
        "selection_dtype": "float32",
        "softmax_dtype": "float32",
        "softmax": "max_subtract_fused_exp_sum_reciprocal",
        "loops": "unrolled",
    }

    return _moe_topk_gating[2](
        logits, k, expert_width, selected_width, groups
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
