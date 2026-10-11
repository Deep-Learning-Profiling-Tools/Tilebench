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
        seq_i = nl.ndarray(
            (heads, 1), dtype=nl.int32, buffer=nl.sbuf
        )
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
        masked_lse = nl.where(valid, lse, float("-inf"))

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

            # Only the diagonal head/output blocks are needed.
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
        if block_seq.device.type == "cpu":
            split_length = int(block_seq.item())
        else:
            # The declared domain fixes block_seq at 128. Avoid passing
            # a rank-zero device tensor into the NKI boundary.
            split_length = 128
    else:
        split_length = int(block_seq)

    split_shift = -1
    if split_length > 0 and (
        split_length & (split_length - 1)
    ) == 0:
        split_shift = split_length.bit_length() - 1

    _LAST_CONFIG = {
        "algorithm": "fp32_max_then_weighted_sum",
        "numerator": "packed_head_fp32_matmul",
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
