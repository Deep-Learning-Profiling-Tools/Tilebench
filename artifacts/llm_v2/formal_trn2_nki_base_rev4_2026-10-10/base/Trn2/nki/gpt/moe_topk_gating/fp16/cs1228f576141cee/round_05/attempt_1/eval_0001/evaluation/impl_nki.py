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

    negative_infinity = nl.ndarray(
        (P, expert_width), dtype=nl.float32, buffer=nl.sbuf
    )
    nisa.memset(dst=negative_infinity, value=_NEG_INF)
    infinity_bits = negative_infinity.view(nl.uint32)

    zero_bits = nl.ndarray(
        (P, expert_width), dtype=nl.uint32, buffer=nl.sbuf
    )
    nisa.memset(dst=zero_bits, value=0)

    for block in range((rows_per_core + batch_rows - 1) // batch_rows):
        local_start = block * batch_rows
        active_groups = min(groups, (rows_per_core - local_start) // P)
        r0 = pid * rows_per_core + local_start

        # Each partition holds one row from each group.
        loaded = nl.ndarray(
            (P, active_groups, expert_width),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        if expert_width > E:
            nisa.memset(dst=loaded, value=_NEG_INF)
        nisa.dma_copy(
            dst=loaded[:, :, 0:E],
            src=logits.ap(
                pattern=[[E, P], [P * E, active_groups], [1, E]],
                offset=r0 * E,
            ),
        )

        # Slot-major free dimensions make each softmax instruction operate
        # on all groups together, without broadcasting across free axes.
        selected = nl.ndarray(
            (P, selected_width, active_groups),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        selected_indices = nl.ndarray(
            (P, selected_width, active_groups),
            dtype=nl.int32,
            buffer=nl.sbuf,
        )
        if selected_width > k:
            nisa.memset(dst=selected, value=_NEG_INF)
            nisa.memset(dst=selected_indices, value=0)

        for group in range(active_groups):
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
                    dst=selected[:, slot, group:group + 1],
                    src=maximum,
                )

                matches = nl.ndarray(
                    (P, expert_width), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_scalar(
                    dst=matches,
                    data=remaining,
                    op0=nl.equal,
                    operand0=maximum,
                    engine=nisa.engine.vector,
                )

                matching_columns = nl.ndarray(
                    (P, expert_width), dtype=nl.float32, buffer=nl.sbuf
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
                    dst=selected_indices[:, slot, group:group + 1],
                    src=maximum_index,
                )

                # Replace exactly the selected float32 bit pattern with
                # -inf; unmatched entries retain their original bits.
                exclusion_mask = nl.ndarray(
                    (P, expert_width), dtype=nl.uint32, buffer=nl.sbuf
                )
                nisa.tensor_copy(dst=exclusion_mask, src=matches)
                nisa.tensor_tensor(
                    dst=exclusion_mask,
                    data1=zero_bits,
                    data2=exclusion_mask,
                    op=nl.subtract,
                    engine=nisa.engine.gpsimd,
                )

                replacement_delta = nl.ndarray(
                    (P, expert_width), dtype=nl.uint32, buffer=nl.sbuf
                )
                remaining_bits = remaining.view(nl.uint32)
                nisa.tensor_tensor(
                    dst=replacement_delta,
                    data1=remaining_bits,
                    data2=infinity_bits,
                    op=nl.bitwise_xor,
                    engine=nisa.engine.gpsimd,
                )
                nisa.tensor_tensor(
                    dst=replacement_delta,
                    data1=replacement_delta,
                    data2=exclusion_mask,
                    op=nl.bitwise_and,
                    engine=nisa.engine.gpsimd,
                )
                nisa.tensor_tensor(
                    dst=remaining_bits,
                    data1=remaining_bits,
                    data2=replacement_delta,
                    op=nl.bitwise_xor,
                    engine=nisa.engine.gpsimd,
                )

        # Two-element max and sum reduction trees, batched over groups.
        selected_maximum = nl.ndarray(
            (P, active_groups), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=selected_maximum,
            data1=selected[:, 0, :],
            data2=selected[:, 1, :],
            op=nl.maximum,
            engine=nisa.engine.vector,
        )

        shifted = nl.ndarray(
            (P, selected_width, active_groups),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        exponentials = nl.ndarray(
            (P, selected_width, active_groups),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        for slot in range(k):
            nisa.tensor_tensor(
                dst=shifted[:, slot, :],
                data1=selected[:, slot, :],
                data2=selected_maximum,
                op=nl.subtract,
                engine=nisa.engine.vector,
            )
            nisa.activation(
                dst=exponentials[:, slot, :],
                op=nl.exp,
                data=shifted[:, slot, :],
            )

        denominator = nl.ndarray(
            (P, active_groups), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=denominator,
            data1=exponentials[:, 0, :],
            data2=exponentials[:, 1, :],
            op=nl.add,
            engine=nisa.engine.vector,
        )

        inverse_denominator = nl.ndarray(
            (P, active_groups), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.reciprocal(
            dst=inverse_denominator,
            data=denominator,
        )

        stored_weights = nl.ndarray(
            (P, selected_width, active_groups),
            dtype=logits.dtype,
            buffer=nl.sbuf,
        )
        for slot in range(k):
            nisa.tensor_tensor(
                dst=stored_weights[:, slot, :],
                data1=exponentials[:, slot, :],
                data2=inverse_denominator,
                op=nl.multiply,
                engine=nisa.engine.vector,
            )

        nisa.dma_copy(
            dst=weights.ap(
                pattern=[[k, P], [1, k], [P * k, active_groups]],
                offset=r0 * k,
            ),
            src=stored_weights[:, 0:k, :],
        )
        nisa.dma_copy(
            dst=indices.ap(
                pattern=[[k, P], [1, k], [P * k, active_groups]],
                offset=r0 * k,
            ),
            src=selected_indices[:, 0:k, :],
        )

    return weights, indices


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    global _LAST_CONFIG

    assert logits.shape == (M, E)
    assert M % 1024 == 0
    assert E == 128 and k == 2

    expert_width = 1 << (E - 1).bit_length()
    selected_width = 1 << (k - 1).bit_length()
    rows_per_core = M // 2
    groups = min(16, rows_per_core // 128)

    _LAST_CONFIG = {
        "lnc": 2,
        "rows_per_tile": 128,
        "groups_per_batch": groups,
        "rows_per_batch": 128 * groups,
        "rows_per_core": rows_per_core,
        "expert_width": expert_width,
        "selected_width": selected_width,
        "selection_rounds": k,
        "selection": "sequential_max_argmax_exclusion",
        "exclusion": "exact_uint32_bit_select",
        "selection_dtype": "float32",
        "softmax_dtype": "float32",
        "softmax": "batched_two_element_max_subtract_exp_sum_reciprocal",
        "softmax_layout": "partition_slot_group",
        "loops": "unrolled",
    }

    return _moe_topk_gating[2](
        logits, k, expert_width, selected_width, groups
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
