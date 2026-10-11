import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _combine(mid_o, mid_o_lse, b_seqlen,
             split_length, split_shift, head_group):
    batch, heads, splits, dim = mid_o.shape
    out = nl.ndarray(
        (batch, heads, dim), dtype=nl.float32, buffer=nl.shared_hbm
    )
    out_view = out.reshape((batch, heads, 1, dim))

    pid = nl.program_id(0)
    programs = nl.num_programs()
    split_tile = 128

    for b in range(pid, batch, programs):
        seq_i = nl.ndarray((heads, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=seq_i,
            src=b_seqlen.ap(
                pattern=[[0, heads], [1, 1]], offset=b
            )
        )

        if split_shift >= 0:
            nisa.tensor_scalar(
                dst=seq_i, data=seq_i,
                op0=nl.add, operand0=split_length - 1
            )
            nisa.tensor_scalar(
                dst=seq_i, data=seq_i,
                op0=nl.right_shift, operand0=split_shift
            )
            nb = nl.ndarray(
                (heads, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=nb, src=seq_i)
        else:
            seq_f = nl.ndarray(
                (heads, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=seq_f, src=seq_i)
            nb = nl.floor(
                nl.multiply(
                    nl.add(seq_f, float(split_length - 1)),
                    1.0 / float(split_length)
                )
            )

        lse = nl.ndarray(
            (heads, splits), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.dma_copy(dst=lse, src=mid_o_lse[b, :, :])

        indices = nl.ndarray(
            (heads, splits), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.iota(
            dst=indices, pattern=[[1, splits]],
            offset=0, channel_multiplier=0
        )
        valid = nl.ndarray(
            (heads, splits), dtype=nl.uint8, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=valid, data=indices, op0=nl.less, operand0=nb
        )

        # Exact bitwise selection between lse and negative infinity.
        # In particular, this avoids multiplying infinity by a zero mask.
        valid_i = nl.ndarray(
            (heads, splits), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=valid_i, src=valid)

        bit_mask = nl.ndarray(
            (heads, splits), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=bit_mask, data=valid_i,
            op0=nl.multiply, operand0=-1,
            engine=nisa.engine.gpsimd
        )

        masked_bits = nl.ndarray(
            (heads, splits), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=masked_bits,
            data1=lse.view(nl.int32), data2=bit_mask,
            op=nl.bitwise_and, engine=nisa.engine.gpsimd
        )

        invalid_bits = nl.ndarray(
            (heads, splits), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=invalid_bits, data=bit_mask,
            op0=nl.bitwise_xor, operand0=-1,
            engine=nisa.engine.gpsimd
        )
        # Signed int32 representation of IEEE fp32 negative infinity.
        nisa.tensor_scalar(
            dst=invalid_bits, data=invalid_bits,
            op0=nl.bitwise_and, operand0=-8388608,
            engine=nisa.engine.gpsimd
        )
        nisa.tensor_tensor(
            dst=masked_bits,
            data1=masked_bits, data2=invalid_bits,
            op=nl.bitwise_or, engine=nisa.engine.gpsimd
        )
        masked_lse = masked_bits.view(nl.float32)

        negative_max = nl.ndarray(
            (heads, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_reduce(
            dst=negative_max, op=nl.maximum,
            data=masked_lse, axis=1
        )
        nisa.tensor_scalar(
            dst=negative_max, data=negative_max,
            op0=nl.multiply, operand0=-1.0
        )

        weights = nl.ndarray(
            (heads, splits), dtype=nl.float32, buffer=nl.sbuf
        )
        denominator = nl.ndarray(
            (heads, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.activation(
            dst=weights, op=nl.exp, data=masked_lse,
            bias=negative_max,
            reduce_op=nl.add, reduce_res=denominator,
            reduce_cmd=nisa.reduce_cmd.reset_reduce
        )

        inverse_denominator = nl.ndarray(
            (heads, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.reciprocal(
            dst=inverse_denominator, data=denominator
        )

        for h0 in range(0, heads, head_group):
            group = min(head_group, heads - h0)
            width = group * dim
            numerator = nl.ndarray(
                (group, width), dtype=nl.float32, buffer=nl.psum
            )

            for tile in range((splits + split_tile - 1) // split_tile):
                s0 = tile * split_tile
                count = min(split_tile, splits - s0)

                weight_rows = nl.ndarray(
                    (group, count), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=weight_rows,
                    src=weights[h0:h0 + group, s0:s0 + count]
                )

                weight_transpose = nl.ndarray(
                    (count, group), dtype=nl.float32, buffer=nl.psum
                )
                nisa.nc_transpose(
                    dst=weight_transpose, data=weight_rows,
                    engine=nisa.engine.tensor
                )
                stationary = nl.ndarray(
                    (count, group), dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.tensor_copy(
                    dst=stationary, src=weight_transpose
                )

                values = nl.ndarray(
                    (count, group, dim),
                    dtype=nl.float32, buffer=nl.sbuf
                )
                nisa.dma_copy(
                    dst=values,
                    src=mid_o[
                        b, h0:h0 + group, s0:s0 + count, :
                    ].permute((1, 0, 2))
                )
                nisa.nc_matmul(
                    dst=numerator,
                    stationary=stationary,
                    moving=values.reshape((count, width)),
                    accumulate=(tile > 0)
                )

            result = nl.ndarray(
                (group, width), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.tensor_copy(dst=result, src=numerator)

            inverse_rows = nl.ndarray(
                (group, 1), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_copy(
                dst=inverse_rows,
                src=inverse_denominator[h0:h0 + group, :]
            )
            nisa.tensor_scalar(
                dst=result, data=result,
                op0=nl.multiply, operand0=inverse_rows
            )

            # Each diagonal block is its own head's weighted numerator.
            for h in range(group):
                nisa.dma_copy(
                    dst=out_view[b, h0 + h, :, :],
                    src=result[h:h + 1, h * dim:(h + 1) * dim]
                )

    return out


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    global _LAST_CONFIG

    batch, heads, splits, dim = mid_o.shape
    head_group = min(heads, 4, max(1, 512 // dim))

    if isinstance(block_seq, torch.Tensor):
        if block_seq.device.type != "cpu":
            raise ValueError("block_seq must be a CPU scalar or Python int")
        split_length = int(block_seq.item())
    else:
        split_length = int(block_seq)

    if split_length <= 0:
        raise ValueError("block_seq must be positive")

    split_shift = -1
    if (split_length & (split_length - 1)) == 0:
        split_shift = split_length.bit_length() - 1

    _LAST_CONFIG = {
        "algorithm": "fp32_max_then_weighted_sum",
        "numerator": "packed_head_fp32_matmul",
        "mask": "bitwise_negative_infinity_select",
        "lnc": 2,
        "split_tile": 128,
        "heads_per_matmul": head_group,
        "batch": batch,
        "heads": heads,
        "num_blocks": splits,
        "head_dim": dim,
        "block_seq": split_length,
    }

    return _combine[2](
        mid_o, mid_o_lse, b_seqlen,
        split_length, split_shift, head_group
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
