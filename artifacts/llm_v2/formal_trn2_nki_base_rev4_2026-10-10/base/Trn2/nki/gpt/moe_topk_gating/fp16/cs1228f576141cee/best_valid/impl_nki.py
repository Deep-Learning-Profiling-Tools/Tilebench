import torch
import nki
import nki.language as nl
import nki.isa as nisa

_NEG_INF = float("-inf")
_LAST_CONFIG = {}


@nki.jit
def _moe_topk_gating(logits, k, expert_width, selected_width):
    M, E = logits.shape
    P = 128
    pid = nl.program_id(0)
    rows_per_core = M // nl.num_programs()

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

    for block in range(rows_per_core // P):
        r0 = pid * rows_per_core + block * P

        remaining = nl.ndarray(
            (P, expert_width), dtype=nl.float32, buffer=nl.sbuf
        )
        if expert_width > E:
            nisa.memset(dst=remaining, value=_NEG_INF)
        nisa.dma_copy(
            dst=remaining[:, 0:E],
            src=logits[r0:r0 + P, :],
        )

        selected = nl.ndarray(
            (P, selected_width), dtype=nl.float32, buffer=nl.sbuf
        )
        selected_indices = nl.ndarray(
            (P, selected_width), dtype=nl.int32, buffer=nl.sbuf
        )
        if selected_width > k:
            nisa.memset(dst=selected, value=_NEG_INF)
            nisa.memset(dst=selected_indices, value=0)

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
                dst=selected[:, slot:slot + 1],
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
                dst=selected_indices[:, slot:slot + 1],
                src=maximum_index,
            )

            # Exact bit selection: unmatched entries remain unchanged;
            # the selected entry is overwritten with IEEE float32 -inf.
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

        selected_maximum = nl.ndarray(
            (P, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_reduce(
            dst=selected_maximum,
            op=nl.maximum,
            data=selected,
            axis=1,
        )

        shifted = nl.ndarray(
            (P, selected_width), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=shifted,
            data=selected,
            op0=nl.subtract,
            operand0=selected_maximum,
            engine=nisa.engine.vector,
        )

        exponentials = nl.ndarray(
            (P, selected_width), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(
            dst=exponentials,
            op=nl.exp,
            data=shifted,
        )

        denominator = nl.ndarray(
            (P, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_reduce(
            dst=denominator,
            op=nl.add,
            data=exponentials,
            axis=1,
        )

        inverse_denominator = nl.ndarray(
            (P, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.reciprocal(
            dst=inverse_denominator,
            data=denominator,
        )

        stored_weights = nl.ndarray(
            (P, selected_width), dtype=logits.dtype, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=stored_weights,
            data=exponentials,
            op0=nl.multiply,
            operand0=inverse_denominator,
            engine=nisa.engine.vector,
        )

        nisa.dma_copy(
            dst=weights[r0:r0 + P, :],
            src=stored_weights[:, 0:k],
        )
        nisa.dma_copy(
            dst=indices[r0:r0 + P, :],
            src=selected_indices[:, 0:k],
        )

    return weights, indices


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    global _LAST_CONFIG

    assert logits.shape == (M, E)
    assert M % 256 == 0
    assert 0 < k <= E

    expert_width = 1 << (E - 1).bit_length()
    selected_width = 1 << (k - 1).bit_length()

    _LAST_CONFIG = {
        "lnc": 2,
        "rows_per_tile": 128,
        "rows_per_core": M // 2,
        "expert_width": expert_width,
        "selected_width": selected_width,
        "selection_rounds": k,
        "selection": "sequential_max_argmax_exclusion",
        "exclusion": "exact_uint32_bit_select",
        "selection_dtype": "float32",
        "softmax_dtype": "float32",
        "softmax": "explicit_max_subtract_exp_sum_reciprocal",
        "loops": "unrolled",
    }

    return _moe_topk_gating[2](
        logits, k, expert_width, selected_width
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
