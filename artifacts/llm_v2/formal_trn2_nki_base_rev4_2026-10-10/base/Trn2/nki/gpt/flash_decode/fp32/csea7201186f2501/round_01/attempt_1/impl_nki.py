import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _flash_decode(mid_o, mid_o_lse, b_seqlen,
                  block_seq, count_shift, split_tile, column_tile):
    B, H, S, D = mid_o.shape
    out = nl.ndarray((B, H, D), dtype=nl.float32, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()

    for b in range(pid, B, programs):
        # ceil(seqlen / block_seq), evaluated on the device.
        lengths = nl.ndarray((H, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=lengths,
            src=b_seqlen.ap(pattern=[[0, H], [0, 1]], offset=b),
        )

        rounding = nl.ndarray((H, 1), dtype=nl.int32, buffer=nl.sbuf)
        shifts = nl.ndarray((H, 1), dtype=nl.int32, buffer=nl.sbuf)
        counts = nl.ndarray((H, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dst=rounding, value=block_seq - 1)
        nisa.memset(dst=shifts, value=count_shift)
        nisa.tensor_tensor(
            dst=counts, data1=lengths, data2=rounding, op=nl.add
        )
        nisa.tensor_tensor(
            dst=counts, data1=counts, data2=shifts, op=nl.right_shift
        )

        count_float = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=count_float, src=counts)

        count_full = nl.ndarray((H, S), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=count_full, value=1.0)
        nisa.tensor_scalar(
            dst=count_full,
            data=count_full,
            op0=nl.multiply,
            operand0=count_float,
        )

        indices = nl.ndarray((H, S), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=indices, pattern=[[1, S]], offset=0)
        valid = nl.ndarray((H, S), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=valid, data1=indices, data2=count_full, op=nl.less
        )

        # Hold every head's LSE axis on chip for the stable max pass.
        lse = nl.ndarray((H, S), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lse, src=mid_o_lse[b, :, :])
        invalid_lse = nl.full(
            (H, S), float("-inf"), dtype=nl.float32, buffer=nl.sbuf
        )
        masked_lse = nl.where(valid, lse, invalid_lse)

        maximum = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        negative_maximum = nl.ndarray(
            (H, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_reduce(
            dst=maximum, data=masked_lse, op=nl.maximum, axis=1
        )
        nisa.tensor_scalar(
            dst=negative_maximum,
            data=maximum,
            op0=nl.multiply,
            operand0=-1.0,
        )

        weights = nl.ndarray((H, S), dtype=nl.float32, buffer=nl.sbuf)
        denominator = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=weights,
            op=nl.exp,
            data=masked_lse,
            bias=negative_maximum,
            reduce_op=nl.add,
            reduce_res=denominator,
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
        )

        # Each diagonal head block is the required weighted numerator.
        # Other matrix blocks are unused. All contractions remain fp32
        # and entirely within this program.
        accumulator = nl.ndarray(
            (H, H * D), dtype=nl.float32, buffer=nl.psum
        )
        for s0 in range(0, S, split_tile):
            K = min(split_tile, S - s0)
            weight_tile = nl.ndarray(
                (K, H), dtype=nl.float32, buffer=nl.sbuf
            )
            nisa.dma_transpose(
                dst=weight_tile, src=weights[:, s0:s0 + K]
            )

            value_tile = nl.ndarray(
                (K, H * D), dtype=nl.float32, buffer=nl.sbuf
            )
            for h in range(H):
                nisa.dma_copy(
                    dst=value_tile[:, h * D:(h + 1) * D],
                    src=mid_o[b, h, s0:s0 + K, :],
                )

            for c0 in range(0, H * D, column_tile):
                N = min(column_tile, H * D - c0)
                nisa.nc_matmul(
                    dst=accumulator[:, c0:c0 + N],
                    stationary=weight_tile,
                    moving=value_tile[:, c0:c0 + N],
                    accumulate=(s0 != 0),
                )

        matrix_numerator = nl.ndarray(
            (H, H * D), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=matrix_numerator, src=accumulator)

        numerator = nl.ndarray((H, D), dtype=nl.float32, buffer=nl.sbuf)
        for h in range(H):
            nisa.dma_copy(
                dst=numerator[h:h + 1, :],
                src=matrix_numerator[h:h + 1, h * D:(h + 1) * D],
            )

        # One final division, implemented by an fp32 reciprocal and multiply.
        inverse_denominator = nl.ndarray(
            (H, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.reciprocal(dst=inverse_denominator, data=denominator)
        nisa.tensor_scalar(
            dst=numerator,
            data=numerator,
            op0=nl.multiply,
            operand0=inverse_denominator,
        )
        nisa.dma_copy(dst=out[b, :, :], src=numerator)

    return out


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    global _LAST_CONFIG

    if isinstance(block_seq, torch.Tensor):
        seq_block = int(block_seq.item())
    else:
        seq_block = int(block_seq)

    # The declared domain uses block_seq=128.
    assert seq_block > 0 and (seq_block & (seq_block - 1)) == 0
    count_shift = seq_block.bit_length() - 1

    batch, heads, splits, head_dim = mid_o.shape
    split_tile = 128
    column_tile = 512

    _LAST_CONFIG = {
        "algorithm": "held_lse_max_then_fp32_gemm",
        "lnc": 2,
        "batch": int(batch),
        "heads": int(heads),
        "num_blocks": int(splits),
        "head_dim": int(head_dim),
        "block_seq": seq_block,
        "count_shift": count_shift,
        "split_tile": split_tile,
        "column_tile": column_tile,
        "head_group": int(heads),
    }

    return _flash_decode[2](
        mid_o, mid_o_lse, b_seqlen,
        seq_block, count_shift, split_tile, column_tile,
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
