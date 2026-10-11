import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _encode_and_replicate(
    groups, encoded, low_mask, shift_two, shift_three, unit,
    lanes, chunk_size,
):
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
        data2=shift_two.broadcast(1, chunk_size),
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
        data2=shift_three.broadcast(1, chunk_size),
        op=nl.right_shift,
        engine=nisa.engine.gpsimd,
    )

    # Ordinary SBUF-to-SBUF DMA supports arbitrary partition offsets.
    # Doubling avoids zero-stride partition access patterns on SBUF.
    copied = 1
    while copied < lanes:
        count = min(copied, lanes - copied)
        nisa.dma_copy(
            dst=groups[copied:copied + count, :],
            src=groups[0:count, :],
        )
        nisa.dma_copy(
            dst=encoded[copied:copied + count, :],
            src=encoded[0:count, :],
        )
        copied = copied + count


def _accumulate_encoded(
    groups, encoded, workspace, accumulator, labels,
    field_shifts, field_mask, bin_groups, lanes, groups_per_chunk,
):
    chunk_size = 15 * groups_per_chunk
    tile_count = (bin_groups + lanes - 1) // lanes

    for tile in range(tile_count):
        rows = min(lanes, bin_groups - tile * lanes)
        matched = workspace[0:rows, :]

        nisa.tensor_tensor(
            dst=matched,
            data1=groups[0:rows, :],
            data2=labels[0:rows, tile, :].broadcast(1, chunk_size),
            op=nl.equal,
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_tensor(
            dst=matched,
            data1=matched,
            data2=encoded[0:rows, :],
            op=nl.multiply,
            engine=nisa.engine.gpsimd,
        )

        # Each integer word contains eight independent four-bit fields.
        # Adding at most fifteen one-hot words cannot carry between fields.
        words = matched.reshape((rows, 15, groups_per_chunk))
        nisa.tensor_tensor(
            dst=words[:, 0:7, :],
            data1=words[:, 0:7, :],
            data2=words[:, 8:15, :],
            op=nl.add,
            engine=nisa.engine.gpsimd,
        )

        width = 8
        while width > 1:
            half = width // 2
            nisa.tensor_tensor(
                dst=words[:, 0:half, :],
                data1=words[:, 0:half, :],
                data2=words[:, half:width, :],
                op=nl.add,
                engine=nisa.engine.gpsimd,
            )
            width = half

        packed = words[:, 0:1, :].reshape((rows, groups_per_chunk))

        # Reuse dead storage, leaving the packed source block intact.
        fields = matched[
            :, groups_per_chunk:9 * groups_per_chunk
        ].reshape((rows, groups_per_chunk, 8))

        nisa.tensor_tensor(
            dst=fields,
            data1=packed.expand_dim(2).broadcast(2, 8),
            data2=field_shifts[0:rows, :, :].broadcast(
                1, groups_per_chunk
            ),
            op=nl.right_shift,
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_tensor(
            dst=fields,
            data1=fields,
            data2=field_mask[0:rows, :, :]
            .broadcast(1, groups_per_chunk)
            .broadcast(2, 8),
            op=nl.bitwise_and,
            engine=nisa.engine.gpsimd,
        )

        width = groups_per_chunk
        while width > 1:
            half = width // 2
            nisa.tensor_tensor(
                dst=fields[:, 0:half, :],
                data1=fields[:, 0:half, :],
                data2=fields[:, half:width, :],
                op=nl.add,
                engine=nisa.engine.gpsimd,
            )
            width = half

        counts = fields[:, 0:1, :].reshape((rows, 8))
        nisa.tensor_tensor(
            dst=accumulator[0:rows, tile, :],
            data1=accumulator[0:rows, tile, :],
            data2=counts,
            op=nl.add,
            engine=nisa.engine.gpsimd,
        )


@nki.jit
def _private_histograms(input, N, num_bins, groups_per_chunk, lanes):
    pid = nl.program_id(0)
    bin_groups = num_bins // 8
    tile_count = (bin_groups + lanes - 1) // lanes
    chunk_size = 15 * groups_per_chunk
    share_size = N // 2
    base = pid * share_size

    partials = nl.ndarray(
        (2, num_bins), dtype=nl.int32, buffer=nl.shared_hbm
    )
    partial_view = partials.reshape((2, bin_groups, 8))
    input_row = input.reshape((1, N))

    accumulator = nl.ndarray(
        (lanes, tile_count, 8), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.memset(dst=accumulator, value=0)

    # Each physical core initializes and exclusively owns one global row.
    # Reading it back makes counting explicitly depend on initialization.
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
    shift_two = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    shift_three = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    unit = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=low_mask, value=7)
    nisa.memset(dst=shift_two, value=2)
    nisa.memset(dst=shift_three, value=3)
    nisa.memset(dst=unit, value=1)

    field_shifts = nl.ndarray(
        (lanes, 1, 8), dtype=nl.int32, buffer=nl.sbuf
    )
    field_mask = nl.ndarray(
        (lanes, 1, 1), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.iota(
        dst=field_shifts,
        pattern=[[0, 1], [4, 8]],
        offset=0,
        channel_multiplier=0,
    )
    nisa.memset(dst=field_mask, value=15)

    groups = nl.ndarray(
        (lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf
    )
    encoded = nl.ndarray(
        (lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf
    )
    workspace = nl.ndarray(
        (lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf
    )

    full_length = (share_size // chunk_size) * chunk_size

    def body(i):
        nisa.dma_copy(
            dst=groups[0:1, :],
            src=input_row.ap(
                pattern=[[0, 1], [1, chunk_size]],
                offset=base,
                scalar_offset=i,
                indirect_dim=1,
            ),
        )
        _encode_and_replicate(
            groups, encoded, low_mask, shift_two, shift_three,
            unit, lanes, chunk_size,
        )
        _accumulate_encoded(
            groups, encoded, workspace, accumulator, labels,
            field_shifts, field_mask, bin_groups, lanes,
            groups_per_chunk,
        )

    nl.fori_loop(0, full_length, body, step=chunk_size)

    tail = share_size - full_length
    if tail > 0:
        # Logical right shift sends the -1 padding outside all valid groups.
        nisa.memset(dst=groups[0:1, :], value=-1)
        nisa.dma_copy(
            dst=groups[0:1, 0:tail],
            src=input_row[:, base + full_length:base + share_size],
        )
        _encode_and_replicate(
            groups, encoded, low_mask, shift_two, shift_three,
            unit, lanes, chunk_size,
        )
        _accumulate_encoded(
            groups, encoded, workspace, accumulator, labels,
            field_shifts, field_mask, bin_groups, lanes,
            groups_per_chunk,
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
def _reduce_partials(partials, num_bins):
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

    groups_per_chunk = 1024
    lanes = min(128, num_bins // 8)

    _LAST_CONFIG = {
        "algorithm": "two_level_private_integer_histogram",
        "lnc": 2,
        "partials": 2,
        "bin_partitions": lanes,
        "bin_tiles": (num_bins // 8 + lanes - 1) // lanes,
        "packed_bins": 8,
        "field_bits": 4,
        "packing_span": 15,
        "groups_per_chunk": groups_per_chunk,
        "chunk_size": 15 * groups_per_chunk,
        "encoding_partitions": 1,
        "replication": "sbuf_dma_partition_doubling",
        "count_reduction": "inplace_contiguous_integer_tree",
        "reduction_partitions": 32,
        "N": N,
        "num_bins": num_bins,
    }

    partials = _private_histograms[2](
        input, N, num_bins, groups_per_chunk, lanes
    )
    return _reduce_partials[2](partials, num_bins)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
