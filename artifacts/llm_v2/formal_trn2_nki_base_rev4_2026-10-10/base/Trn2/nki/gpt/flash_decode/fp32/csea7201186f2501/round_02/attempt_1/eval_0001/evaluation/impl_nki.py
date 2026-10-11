import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _decode_impl(mid_o, mid_o_lse, b_seqlen, block_tensor,
                 block_seq, block_shift, head_group):
    batch, heads, splits, dim = mid_o.shape
    out = nl.ndarray((batch, heads, dim), dtype=nl.float32,
                     buffer=nl.shared_hbm)
    out_view = out.reshape((batch, heads, 1, dim))

    pid = nl.program_id(0)
    programs = nl.num_programs()
    split_tile = 128

    for b in range(pid, batch, programs):
        seq_source = b_seqlen.ap(
            pattern=[[0, heads], [1, 1]], offset=b
        )

        # Derive the valid split count from the current device input.
        if block_tensor is None and block_shift >= 0:
            seq_i = nl.ndarray((heads, 1), dtype=nl.int32,
                               buffer=nl.sbuf)
            nisa.dma_copy(dst=seq_i, src=seq_source)
            nisa.tensor_scalar(
                dst=seq_i, data=seq_i,
                op0=nl.add, operand0=block_seq - 1
            )
            nisa.tensor_scalar(
                dst=seq_i, data=seq_i,
                op0=nl.right_shift, operand0=block_shift
            )
            nb = nl.ndarray((heads, 1), dtype=nl.float32,
                            buffer=nl.sbuf)
            nisa.tensor_copy(dst=nb, src=seq_i)
        else:
            seq_f = nl.ndarray((heads, 1), dtype=nl.float32,
                               buffer=nl.sbuf)
            nisa.dma_copy(dst=seq_f, src=seq_source)
            if block_tensor is None:
                count_f = nl.multiply(
                    nl.add(seq_f, float(block_seq - 1)),
                    1.0 / float(block_seq)
                )
            else:
                block_f = nl.ndarray((1, 1), dtype=nl.float32,
                                     buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=block_f, src=block_tensor.reshape((1, 1))
                )
                block_rows = nl.broadcast_to(block_f, (heads, 1))
                count_f = nl.multiply(
                    nl.add(nl.add(seq_f, block_rows), -1.0),
                    nl.reciprocal(block_rows)
                )
            nb = nl.floor(count_f)

        lse = nl.ndarray((heads, splits), dtype=nl.float32,
                         buffer=nl.sbuf)
        nisa.dma_copy(dst=lse, src=mid_o_lse[b, :, :])

        indices = nl.ndarray((heads, splits), dtype=nl.float32,
                             buffer=nl.sbuf)
        nisa.iota(dst=indices, pattern=[[1, splits]], offset=0,
                  channel_multiplier=0)
        valid = nl.ndarray((heads, splits), dtype=nl.uint8,
                           buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=valid, data=indices, op0=nl.less, operand0=nb
        )
        masked_lse = nl.where(valid, lse, float("-inf"))

        maximum = nl.ndarray((heads, 1), dtype=nl.float32,
                             buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=maximum, op=nl.maximum, data=masked_lse, axis=1
        )
        nisa.tensor_scalar(
            dst=maximum, data=maximum,
            op0=nl.multiply, operand0=-1.0
        )

        weights = nl.ndarray((heads, splits), dtype=nl.float32,
                             buffer=nl.sbuf)
        denominator = nl.ndarray((heads, 1), dtype=nl.float32,
                                 buffer=nl.sbuf)
        nisa.activation(
            dst=weights, op=nl.exp, data=masked_lse, bias=maximum,
            reduce_op=nl.add, reduce_res=denominator,
            reduce_cmd=nisa.reduce_cmd.reset_reduce
        )
        inverse_denominator = nl.ndarray(
            (heads, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.reciprocal(dst=inverse_denominator, data=denominator)

        # Pack several heads into each moving operand. Only diagonal
        # head/output blocks are retained from the fp32 matrix product.
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
                nisa.tensor_copy(dst=stationary, src=weight_transpose)

                values = nl.ndarray(
                    (count, group, dim), dtype=nl.float32,
                    buffer=nl.sbuf
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

            for h in range(group):
                nisa.dma_copy(
                    dst=out_view[b, h0 + h, :, :],
                    src=result[h:h + 1, h * dim:(h + 1) * dim]
                )

    return out


@nki.jit
def _decode_constant(mid_o, mid_o_lse, b_seqlen,
                     block_seq, block_shift, head_group):
    return _decode_impl(
        mid_o, mid_o_lse, b_seqlen, None,
        block_seq, block_shift, head_group
    )


@nki.jit
def _decode_device_scalar(mid_o, mid_o_lse, b_seqlen,
                          block_tensor, head_group):
    return _decode_impl(
        mid_o, mid_o_lse, b_seqlen, block_tensor,
        128, -1, head_group
    )


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    global _LAST_CONFIG

    batch, heads, splits, dim = mid_o.shape
    head_group = min(heads, 4, max(1, 512 // dim))

    device_scalar = False
    if isinstance(block_seq, torch.Tensor):
        if block_seq.device.type == "cpu":
            split_length = int(block_seq.item())
        else:
            device_scalar = True
            split_length = 128
    else:
        split_length = int(block_seq)

    shift = -1
    if not device_scalar:
        if split_length > 0 and (
            split_length & (split_length - 1)
        ) == 0:
            shift = split_length.bit_length() - 1

    _LAST_CONFIG = {
        "algorithm": "fp32_max_then_weighted_sum",
        "numerator": "packed_head_matmul",
        "lnc": 2,
        "split_tile": 128,
        "heads_per_matmul": head_group,
        "batch": batch,
        "heads": heads,
        "num_blocks": splits,
        "head_dim": dim,
        "block_seq": "device" if device_scalar else split_length,
    }

    if device_scalar:
        return _decode_device_scalar[2](
            mid_o, mid_o_lse, b_seqlen, block_seq, head_group
        )
    return _decode_constant[2](
        mid_o, mid_o_lse, b_seqlen,
        split_length, shift, head_group
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
