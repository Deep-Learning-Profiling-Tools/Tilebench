import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _accumulate_chunk(
    values,
    accumulator,
    labels,
    low_mask,
    shift_two,
    shift_three,
    unit,
    field_shifts,
    field_mask,
    bin_groups,
    lanes,
    groups_per_chunk,
):
    chunk_size = groups_per_chunk * 15

    shifts = nl.ndarray((lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf)
    encoded = nl.ndarray((lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf)
    input_groups = nl.ndarray((lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf)

    nisa.tensor_tensor(
        dst=shifts, data1=values, data2=low_mask,
        op=nl.bitwise_and, engine=nisa.engine.gpsimd,
    )
    nisa.tensor_tensor(
        dst=shifts, data1=shifts, data2=shift_two,
        op=nl.left_shift, engine=nisa.engine.gpsimd,
    )
    nisa.tensor_tensor(
        dst=encoded, data1=unit, data2=shifts,
        op=nl.left_shift, engine=nisa.engine.gpsimd,
    )
    nisa.tensor_tensor(
        dst=input_groups, data1=values, data2=shift_three,
        op=nl.right_shift, engine=nisa.engine.gpsimd,
    )

    tile_count = (bin_groups + lanes - 1) // lanes
    for tile in range(tile_count):
        rows = min(lanes, bin_groups - tile * lanes)

        matched = nl.ndarray(
            (rows, chunk_size), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=matched,
            data1=input_groups[0:rows, :],
            data2=labels[0:rows, tile, :],
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

        # Eight four-bit counters are packed into an int32 word.
        # Summing only fifteen inputs prevents carries between fields.
        matched3 = matched.reshape((rows, groups_per_chunk, 15))
        reduced = nl.ndarray(
            (rows, groups_per_chunk, 8), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=reduced[:, :, 0:7],
            data1=matched3[:, :, 0:7],
            data2=matched3[:, :, 8:15],
            op=nl.add,
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_copy(
            dst=reduced[:, :, 7:8],
            src=matched3[:, :, 7:8],
            engine=nisa.engine.gpsimd,
        )

        width = 8
        while width > 1:
            half = width // 2
            next_reduced = nl.ndarray(
                (rows, groups_per_chunk, half),
                dtype=nl.int32,
                buffer=nl.sbuf,
            )
            nisa.tensor_tensor(
                dst=next_reduced,
                data1=reduced[:, :, 0:half],
                data2=reduced[:, :, half:width],
                op=nl.add,
                engine=nisa.engine.gpsimd,
            )
            reduced = next_reduced
            width = half

        packed = reduced.reshape((rows, groups_per_chunk))
        replicated = packed.expand_dim(1).broadcast(1, 8)
        fields = nl.ndarray(
            (rows, 8, groups_per_chunk), dtype=nl.int32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=fields,
            data1=replicated,
            data2=field_shifts[0:rows, :, :],
            op=nl.right_shift,
            engine=nisa.engine.gpsimd,
        )
        nisa.tensor_tensor(
            dst=fields,
            data1=fields,
            data2=field_mask[0:rows, :, :],
            op=nl.bitwise_and,
            engine=nisa.engine.gpsimd,
        )

        # Unpacked counts are summed with native int32 additions.
        width = groups_per_chunk
        while width > 1:
            half = width // 2
            next_fields = nl.ndarray(
                (rows, 8, half), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.tensor_tensor(
                dst=next_fields,
                data1=fields[:, :, 0:half],
                data2=fields[:, :, half:width],
                op=nl.add,
                engine=nisa.engine.gpsimd,
            )
            fields = next_fields
            width = half

        counts = fields.reshape((rows, 8))
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
    partials = nl.ndarray(
        (2, num_bins), dtype=nl.int32, buffer=nl.shared_hbm
    )
    partial_view = partials.reshape((2, num_bins // 8, 8))

    bin_groups = num_bins // 8
    tile_count = (bin_groups + lanes - 1) // lanes
    chunk_size = groups_per_chunk * 15
    share_size = N // 2
    base = pid * share_size

    # Each core owns one global partial row and caches that row in SBUF.
    accumulator = nl.ndarray(
        (lanes, tile_count, 8), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.memset(dst=accumulator, value=0)
    for tile in range(tile_count):
        rows = min(lanes, bin_groups - tile * lanes)
        start = tile * lanes
        nisa.dma_copy(
            dst=partial_view[pid, start:start + rows, :],
            src=accumulator[0:rows, tile, :],
        )

    labels = nl.ndarray(
        (lanes, tile_count, chunk_size), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.iota(
        dst=labels,
        pattern=[[lanes, tile_count], [0, chunk_size]],
        offset=0,
        channel_multiplier=1,
    )

    low_mask = nl.ndarray(
        (lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf
    )
    shift_two = nl.ndarray(
        (lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf
    )
    shift_three = nl.ndarray(
        (lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf
    )
    unit = nl.ndarray(
        (lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.memset(dst=low_mask, value=7)
    nisa.memset(dst=shift_two, value=2)
    nisa.memset(dst=shift_three, value=3)
    nisa.memset(dst=unit, value=1)

    field_shifts = nl.ndarray(
        (lanes, 8, groups_per_chunk), dtype=nl.int32, buffer=nl.sbuf
    )
    field_mask = nl.ndarray(
        (lanes, 8, groups_per_chunk), dtype=nl.int32, buffer=nl.sbuf
    )
    nisa.iota(
        dst=field_shifts,
        pattern=[[4, 8], [0, groups_per_chunk]],
        offset=0,
        channel_multiplier=0,
    )
    nisa.memset(dst=field_mask, value=15)

    values = nl.ndarray(
        (lanes, chunk_size), dtype=nl.int32, buffer=nl.sbuf
    )
    full_length = (share_size // chunk_size) * chunk_size

    def body(i):
        nisa.dma_copy(
            dst=values,
            src=input.ap(
                pattern=[[0, lanes], [1, chunk_size]],
                offset=base,
                scalar_offset=i,
                indirect_dim=1,
            ),
        )
        _accumulate_chunk(
            values, accumulator, labels,
            low_mask, shift_two, shift_three, unit,
            field_shifts, field_mask,
            bin_groups, lanes, groups_per_chunk,
        )

    nl.fori_loop(0, full_length, body, step=chunk_size)

    tail = share_size - full_length
    if tail > 0:
        # -1 maps to no valid group under logical right shift.
        nisa.memset(dst=values, value=-1)
        nisa.dma_copy(
            dst=values[:, 0:tail],
            src=input.ap(
                pattern=[[0, lanes], [1, tail]],
                offset=base + full_length,
            ),
        )
        _accumulate_chunk(
            values, accumulator, labels,
            low_mask, shift_two, shift_three, unit,
            field_shifts, field_mask,
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
    result = nl.ndarray((32, free), dtype=nl.int32, buffer=nl.sbuf)

    nisa.dma_copy(dst=first, src=partial_view[0, pid, :, :])
    nisa.dma_copy(dst=second, src=partial_view[1, pid, :, :])
    nisa.tensor_tensor(
        dst=result,
        data1=first,
        data2=second,
        op=nl.add,
        engine=nisa.engine.gpsimd,
    )
    nisa.dma_copy(dst=output_view[pid, :, :], src=result)
    return output


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    global _LAST_CONFIG

    assert input.shape == (N,)
    assert input.dtype == torch.int32
    assert input.stride() == (1,)
    assert N % 262144 == 0
    assert num_bins >= 64 and num_bins % 64 == 0

    groups_per_chunk = 256 if num_bins <= 1024 else 128
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
