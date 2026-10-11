import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _encode_replicate16(data, low_mask, unit, group_shift, lanes, chunk_size):
    groups = data[:, 0, :]
    encoded = data[:, 1, :]

    nisa.tensor_tensor(
        dst=encoded[0:1, :],
        data1=groups[0:1, :],
        data2=low_mask.broadcast(1, chunk_size),
        op=nl.bitwise_and,
        engine=nisa.engine.gpsimd,
    )
    nisa.tensor_tensor(
        dst=encoded[0:1, :],
        data1=encoded[0:1, :],
        data2=unit.broadcast(1, chunk_size),
        op=nl.left_shift,
        engine=nisa.engine.gpsimd,
    )
    nisa.tensor_tensor(
        dst=encoded[0:1, :],
        data1=unit.broadcast(1, chunk_size),
        data2=encoded[0:1, :],
        op=nl.left_shift,
        engine=nisa.engine.gpsimd,
    )
    nisa.tensor_tensor(
        dst=groups[0:1, :],
        data1=groups[0:1, :],
        data2=group_shift.broadcast(1, chunk_size),
        op=nl.right_shift,
        engine=nisa.engine.gpsimd,
    )

    copied = 1
    while copied < lanes:
        count = min(copied, lanes - copied)
        nisa.dma_copy(
            dst=data[copied:copied + count, :, :],
            src=data[0:count, :, :],
        )
        copied = copied + count


def _accumulate_swar16(
    data, workspace, accumulator, labels,
    pair_shifts, quad_shifts, oct_shifts, final_shifts,
    pair_mask, quad_mask, oct_mask, final_mask,
    bin_groups, lanes, groups_per_chunk,
):
    G = groups_per_chunk
    chunk_size = 3 * G
    quarter_groups = G // 5
    final_groups = quarter_groups // 16
    tile_count = (bin_groups + lanes - 1) // lanes

    for tile in range(tile_count):
        rows = min(lanes, bin_groups - tile * lanes)
        matched = workspace[0:rows, :]

        nisa.tensor_tensor(
            dst=matched,
            data1=data[0:rows, 0, :],
            data2=labels[0:rows, tile, :].broadcast(1, chunk_size),
            op=nl.equal,
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_tensor(
            dst=matched,
            data1=matched,
            data2=data[0:rows, 1, :],
            op=nl.multiply,
            engine=nisa.engine.gpsimd,
        )

        # Sixteen two-bit fields: three one-hot words sum without carries.
        words = matched.reshape((rows, 3, G))
        nisa.tensor_tensor(
            dst=words[:, 0:1, :],
            data1=words[:, 0:1, :],
            data2=words[:, 1:2, :],
            op=nl.add,
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_tensor(
            dst=words[:, 0:1, :],
            data1=words[:, 0:1, :],
            data2=words[:, 2:3, :],
            op=nl.add,
            engine=nisa.engine.gpsimd,
        )
        packed = words[:, 0:1, :].reshape((rows, G))

        # Separate alternating fields into two words with four-bit spacing.
        pairs = matched[:, G:3 * G].reshape((rows, G, 2))
        nisa.tensor_tensor(
            dst=pairs,
            data1=packed.expand_dim(2).broadcast(2, 2),
            data2=pair_shifts[0:rows, :, :].broadcast(1, G),
            op=nl.right_shift,
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_tensor(
            dst=pairs,
            data1=pairs,
            data2=pair_mask[0:rows, :, :]
            .broadcast(1, G).broadcast(2, 2),
            op=nl.bitwise_and,
            engine=nisa.engine.gpsimd,
        )

        # Five groups give at most 3 * 5 = 15 in each four-bit field.
        pair_blocks = pairs.reshape((rows, 5, quarter_groups, 2))
        nisa.tensor_tensor(
            dst=pair_blocks[:, 0:1, :, :],
            data1=pair_blocks[:, 0:1, :, :],
            data2=pair_blocks[:, 4:5, :, :],
            op=nl.add,
            engine=nisa.engine.gpsimd,
        )
        width = 4
        while width > 1:
            half = width // 2
            nisa.tensor_tensor(
                dst=pair_blocks[:, 0:half, :, :],
                data1=pair_blocks[:, 0:half, :, :],
                data2=pair_blocks[:, half:width, :, :],
                op=nl.add,
                engine=nisa.engine.gpsimd,
            )
            width = half
        packed_pairs = pair_blocks[:, 0:1, :, :].reshape(
            (rows, quarter_groups, 2)
        )

        # Widen four-bit fields to byte fields, preserving bin order.
        quads = matched[:, 0:4 * quarter_groups].reshape(
            (rows, quarter_groups, 2, 2)
        )
        nisa.tensor_tensor(
            dst=quads,
            data1=packed_pairs.expand_dim(2).broadcast(2, 2),
            data2=quad_shifts[0:rows, :, :, :]
            .broadcast(1, quarter_groups).broadcast(3, 2),
            op=nl.right_shift,
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_tensor(
            dst=quads,
            data1=quads,
            data2=quad_mask[0:rows, :, :, :]
            .broadcast(1, quarter_groups).broadcast(2, 2).broadcast(3, 2),
            op=nl.bitwise_and,
            engine=nisa.engine.gpsimd,
        )

        # Sixteen groups give at most 15 * 16 = 240 per byte.
        quad_blocks = quads.reshape((rows, 16, final_groups, 4))
        width = 16
        while width > 1:
            half = width // 2
            nisa.tensor_tensor(
                dst=quad_blocks[:, 0:half, :, :],
                data1=quad_blocks[:, 0:half, :, :],
                data2=quad_blocks[:, half:width, :, :],
                op=nl.add,
                engine=nisa.engine.gpsimd,
            )
            width = half
        packed_quads = quad_blocks[:, 0:1, :, :].reshape(
            (rows, final_groups, 4)
        )

        # Widen byte fields to halfwords before the final integer tree.
        octets = matched[:, G:G + 8 * final_groups].reshape(
            (rows, final_groups, 2, 4)
        )
        nisa.tensor_tensor(
            dst=octets,
            data1=packed_quads.expand_dim(2).broadcast(2, 2),
            data2=oct_shifts[0:rows, :, :, :]
            .broadcast(1, final_groups).broadcast(3, 4),
            op=nl.right_shift,
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_tensor(
            dst=octets,
            data1=octets,
            data2=oct_mask[0:rows, :, :, :]
            .broadcast(1, final_groups).broadcast(2, 2).broadcast(3, 4),
            op=nl.bitwise_and,
            engine=nisa.engine.gpsimd,
        )

        octet_words = octets.reshape((rows, final_groups, 8))
        width = final_groups
        while width > 1:
            half = width // 2
            nisa.tensor_tensor(
                dst=octet_words[:, 0:half, :],
                data1=octet_words[:, 0:half, :],
                data2=octet_words[:, half:width, :],
                op=nl.add,
                engine=nisa.engine.gpsimd,
            )
            width = half

        # At most 15360 counts per halfword, so no inter-field carries.
        counts = matched[:, 0:16].reshape((rows, 2, 8))
        nisa.tensor_tensor(
            dst=counts,
            data1=octet_words[:, 0:1, :].broadcast(1, 2),
            data2=final_shifts[0:rows, :, :].broadcast(2, 8),
            op=nl.right_shift,
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_tensor(
            dst=counts,
            data1=counts,
            data2=final_mask[0:rows, :, :]
            .broadcast(1, 2).broadcast(2, 8),
            op=nl.bitwise_and,
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_tensor(
            dst=accumulator[0:rows, tile, :],
            data1=accumulator[0:rows, tile, :],
            data2=counts.reshape((rows, 16)),
            op=nl.add,
            engine=nisa.engine.gpsimd,
        )


@nki.jit
def _private_histograms_swar16(input, N, num_bins, groups_per_chunk, lanes):
    pid = nl.program_id(0)
    bin_groups = num_bins // 16
    tile_count = (bin_groups + lanes - 1) // lanes
    chunk_size = 3 * groups_per_chunk
    share_size = N // 2
    base = pid * share_size

    partials = nl.ndarray(
        (2, num_bins), dtype=nl.int32, buffer=nl.shared_hbm
    )
    partial_view = partials.reshape((2, bin_groups, 16))
    input_row = input.reshape((1, N))

    accumulator = nl.ndarray(
        (lanes, tile_count, 16), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.memset(dst=accumulator, value=0)

    # Each physical core zeroes and exclusively owns one global scratch row.
    for tile in range(tile_count):
        rows = min(lanes, bin_groups - tile * lanes)
        start = tile * lanes
        nisa.dma_copy(
            dst=partial_view[pid, start:start + rows, :],
            src=accumulator[0:rows, tile, :],
        )
        nisa.dma_copy(
            dst=accumulator[0:rows, tile, :],
            src=partial_view[pid, start:start + rows, :],
        )

    labels = nl.ndarray(
        (lanes, tile_count, 1), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.iota(
        dst=labels,
        pattern=[[lanes, tile_count], [0, 1]],
        offset=0,
        channel_multiplier=1,
    )

    low_mask = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    unit = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    group_shift = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=low_mask, value=15)
    nisa.memset(dst=unit, value=1)
    nisa.memset(dst=group_shift, value=4)

    pair_shifts = nl.ndarray((lanes, 1, 2), dtype=nl.int32, buffer=nl.sbuf)
    quad_shifts = nl.ndarray(
        (lanes, 1, 2, 1), dtype=nl.int32, buffer=nl.sbuf
    )
    oct_shifts = nl.ndarray(
        (lanes, 1, 2, 1), dtype=nl.int32, buffer=nl.sbuf
    )
    final_shifts = nl.ndarray((lanes, 2, 1), dtype=nl.int32, buffer=nl.sbuf)

    nisa.iota(
        dst=pair_shifts, pattern=[[0, 1], [2, 2]],
        offset=0, channel_multiplier=0,
    )
    nisa.iota(
        dst=quad_shifts, pattern=[[0, 1], [4, 2], [0, 1]],
        offset=0, channel_multiplier=0,
    )
    nisa.iota(
        dst=oct_shifts, pattern=[[0, 1], [8, 2], [0, 1]],
        offset=0, channel_multiplier=0,
    )
    nisa.iota(
        dst=final_shifts, pattern=[[16, 2], [0, 1]],
        offset=0, channel_multiplier=0,
    )

    pair_mask = nl.ndarray((lanes, 1, 1), dtype=nl.int32, buffer=nl.sbuf)
    quad_mask = nl.ndarray(
        (lanes, 1, 1, 1), dtype=nl.int32, buffer=nl.sbuf
    )
    oct_mask = nl.ndarray(
        (lanes, 1, 1, 1), dtype=nl.int32, buffer=nl.sbuf
    )
    final_mask = nl.ndarray((lanes, 1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=pair_mask, value=0x33333333)
    nisa.memset(dst=quad_mask, value=0x0F0F0F0F)
    nisa.memset(dst=oct_mask, value=0x00FF00FF)
    nisa.memset(dst=final_mask, value=65535)

    data = nl.ndarray(
        (lanes, 2, chunk_size), dtype=nl.int32, buffer=nl.sbuf
    )
    workspace = nl.ndarray(
        (lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf
    )
    full_length = (share_size // chunk_size) * chunk_size

    def body(i):
        nisa.dma_copy(
            dst=data[0:1, 0, :],
            src=input_row.ap(
                pattern=[[0, 1], [1, chunk_size]],
                offset=base,
                scalar_offset=i,
                indirect_dim=1,
            ),
        )
        _encode_replicate16(
            data, low_mask, unit, group_shift, lanes, chunk_size
        )
        _accumulate_swar16(
            data, workspace, accumulator, labels,
            pair_shifts, quad_shifts, oct_shifts, final_shifts,
            pair_mask, quad_mask, oct_mask, final_mask,
            bin_groups, lanes, groups_per_chunk,
        )

    nl.fori_loop(0, full_length, body, step=chunk_size)

    tail = share_size - full_length
    if tail > 0:
        # Logical right shift maps padding outside every valid bin group.
        nisa.memset(dst=data[0:1, 0, :], value=-1)
        nisa.dma_copy(
            dst=data[0:1, 0, 0:tail],
            src=input_row[:, base + full_length:base + share_size],
        )
        _encode_replicate16(
            data, low_mask, unit, group_shift, lanes, chunk_size
        )
        _accumulate_swar16(
            data, workspace, accumulator, labels,
            pair_shifts, quad_shifts, oct_shifts, final_shifts,
            pair_mask, quad_mask, oct_mask, final_mask,
            bin_groups, lanes, groups_per_chunk,
        )

    for tile in range(tile_count):
        rows = min(lanes, bin_groups - tile * lanes)
        start = tile * lanes
        nisa.dma_copy(
            dst=partial_view[pid, start:start + rows, :],
            src=accumulator[0:rows, tile, :],
        )

    return partials


@nki.jit
def _reduce_partials_swar16(partials, num_bins):
    pid = nl.program_id(0)
    free = num_bins // 64

    output = nl.ndarray(
        (num_bins,), dtype=nl.int32, buffer=nl.shared_hbm
    )
    partial_view = partials.reshape((2, 2, 32, free))
    output_view = output.reshape((2, 32, free))

    first = nl.ndarray((32, free), dtype=nl.int32, buffer=nl.sbuf)
    second = nl.ndarray((32, free), dtype=nl.int32, buffer=nl.sbuf)

    nisa.dma_copy(dst=first, src=partial_view[0, pid, :, :])
    nisa.dma_copy(dst=second, src=partial_view[1, pid, :, :])
    nisa.tensor_tensor(
        dst=first,
        data1=first,
        data2=second,
        op=nl.add,
        engine=nisa.engine.gpsimd,
    )
    nisa.dma_copy(dst=output_view[pid, :, :], src=first)
    return output


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    global _LAST_CONFIG

    assert input.shape == (N,)
    assert input.dtype == torch.int32
    assert input.stride() == (1,)
    assert 262144 <= N <= 67108864 and N % 262144 == 0
    assert 64 <= num_bins <= 4096 and num_bins % 64 == 0

    groups_per_chunk = 5120
    lanes = min(128, num_bins // 16)

    _LAST_CONFIG = {
        "algorithm": "two_level_private_integer_histogram",
        "lnc": 2,
        "partials": 2,
        "bin_partitions": lanes,
        "bin_tiles": (num_bins // 16 + lanes - 1) // lanes,
        "packed_bins": 16,
        "initial_field_bits": 2,
        "initial_packing_span": 3,
        "field_widening_bits": [2, 4, 8, 16],
        "integer_reduction_factors": [3, 5, 16, 64],
        "groups_per_chunk": groups_per_chunk,
        "chunk_size": 3 * groups_per_chunk,
        "encoding_partitions": 1,
        "replication": "combined_sbuf_dma_partition_doubling",
        "count_reduction": "carry_free_integer_swar_tree",
        "reduction_partitions": 32,
        "N": N,
        "num_bins": num_bins,
    }

    partials = _private_histograms_swar16[2](
        input, N, num_bins, groups_per_chunk, lanes
    )
    return _reduce_partials_swar16[2](partials, num_bins)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
