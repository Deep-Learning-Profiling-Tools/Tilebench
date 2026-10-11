import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _process_block(
    x, out, weights, length, tile_width, dma_width, prefetch_groups,
    matmul_width, psum_sets, start, loop_index, runtime_offset,
    left_edge, right_edge
):
    # Each DMA stages several GEMM tiles, sharing their internal halos.
    patches = nl.ndarray(
        (128, prefetch_groups, dma_width + 2),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )
    results = nl.ndarray(
        (128, prefetch_groups, dma_width),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )
    accumulators = nl.ndarray(
        (128, psum_sets, tile_width),
        dtype=nl.float32,
        buffer=nl.psum,
    )

    for group in range(prefetch_groups):
        position = start + group * dma_width

        if left_edge and group == 0:
            nisa.memset(dst=patches[:, group, 0:1], value=0.0)
            nisa.dma_copy(
                dst=patches[:, group, 1:dma_width + 2],
                src=x.ap(
                    pattern=[[length, 128], [1, dma_width + 1]],
                    offset=position,
                ),
            )
        elif right_edge and group == prefetch_groups - 1:
            nisa.dma_copy(
                dst=patches[:, group, 0:dma_width + 1],
                src=x.ap(
                    pattern=[[length, 128], [1, dma_width + 1]],
                    offset=position - 1,
                ),
            )
            nisa.memset(
                dst=patches[:, group, dma_width + 1:dma_width + 2],
                value=0.0,
            )
        elif runtime_offset:
            nisa.dma_copy(
                dst=patches[:, group, :],
                src=x.ap(
                    pattern=[[length, 128], [1, dma_width + 2]],
                    offset=position - 1,
                    scalar_offset=loop_index,
                    indirect_dim=1,
                ),
            )
        else:
            nisa.dma_copy(
                dst=patches[:, group, :],
                src=x.ap(
                    pattern=[[length, 128], [1, dma_width + 2]],
                    offset=position - 1,
                ),
            )

    tiles_per_dma = dma_width // tile_width

    for group in range(prefetch_groups):
        for tile in range(tiles_per_dma):
            t0 = tile * tile_width
            slot = (group * tiles_per_dma + tile) % psum_sets

            # One fp32 accumulator covers the complete tap-major reduction.
            # Alternating PSUM sets permits copies to overlap subsequent GEMMs.
            for tap in range(3):
                for column in range(tile_width // matmul_width):
                    c0 = column * matmul_width
                    nisa.nc_matmul(
                        dst=accumulators[
                            :, slot, c0:c0 + matmul_width
                        ],
                        stationary=weights[:, tap, :],
                        moving=patches[
                            :, group,
                            t0 + c0 + tap:t0 + c0 + tap + matmul_width
                        ],
                        accumulate=(tap != 0),
                    )

            nisa.tensor_copy(
                dst=results[:, group, t0:t0 + tile_width],
                src=accumulators[:, slot, :],
            )

        # Coalesce neighboring completed tiles into a larger output DMA.
        position = start + group * dma_width
        if runtime_offset:
            nisa.dma_copy(
                dst=out.ap(
                    pattern=[[length, 128], [1, dma_width]],
                    offset=position,
                    scalar_offset=loop_index,
                    indirect_dim=1,
                ),
                src=results[:, group, :],
            )
        else:
            nisa.dma_copy(
                dst=out.ap(
                    pattern=[[length, 128], [1, dma_width]],
                    offset=position,
                ),
                src=results[:, group, :],
            )


@nki.jit
def _conv1d_kernel(
    input, weight, tile_width, dma_width, prefetch_groups,
    matmul_width, psum_sets
):
    length = input.shape[2]
    output = nl.ndarray(
        (1, 128, length),
        dtype=input.dtype,
        buffer=nl.shared_hbm,
    )

    pid = nl.program_id(0)
    programs = nl.num_programs()
    span = length // programs
    base = pid * span
    block_width = dma_width * prefetch_groups

    x = input[0]
    out = output[0]

    weights = nl.ndarray(
        (128, 3, 128),
        dtype=weight.dtype,
        buffer=nl.sbuf,
    )
    for tap in range(3):
        source = nl.ndarray(
            (128, 128),
            dtype=weight.dtype,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(dst=source, src=weight[:, :, tap])

        transposed = nl.ndarray(
            (128, 128),
            dtype=weight.dtype,
            buffer=nl.psum,
        )
        nisa.nc_transpose(
            dst=transposed,
            data=source,
            engine=nisa.engine.tensor,
        )
        nisa.tensor_copy(dst=weights[:, tap, :], src=transposed)

    _process_block(
        x, out, weights, length, tile_width, dma_width, prefetch_groups,
        matmul_width, psum_sets, base, 0, False, pid == 0, False,
    )

    interior_base = base + block_width

    def body(i):
        _process_block(
            x, out, weights, length, tile_width, dma_width, prefetch_groups,
            matmul_width, psum_sets, interior_base, i, True, False, False,
        )

    if span > 2 * block_width:
        nl.fori_loop(
            0,
            span - 2 * block_width,
            body,
            step=block_width,
        )

    _process_block(
        x, out, weights, length, tile_width, dma_width, prefetch_groups,
        matmul_width, psum_sets, base + span - block_width, 0, False,
        False, pid == programs - 1,
    )

    return output


def run(
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 1,
    groups: int = 1,
    **kwargs
):
    global _LAST_CONFIG

    batch, channels, length = input.shape
    tile_width = 2048
    dma_width = 8192
    prefetch_groups = 4
    matmul_width = 512
    psum_sets = 2
    block_width = dma_width * prefetch_groups

    assert batch == 1 and channels == 128
    assert tuple(weight.shape) == (128, 128, 3)
    assert input.dtype == torch.float16
    assert weight.dtype == input.dtype
    assert stride == 1 and padding == 1 and groups == 1
    assert length >= 4 * block_width
    assert length % (2 * block_width) == 0

    _LAST_CONFIG = {
        "implementation": "implicit_gemm",
        "lnc": 2,
        "length": int(length),
        "output_channels": 128,
        "reduction_chunk": 128,
        "reduction_order": "tap_major",
        "matmul_width": matmul_width,
        "tile_width": tile_width,
        "dma_width": dma_width,
        "prefetch_groups": prefetch_groups,
        "prefetch_tiles": prefetch_groups * (dma_width // tile_width),
        "block_width": block_width,
        "psum_buffer_sets": psum_sets,
        "accumulation_dtype": "float32",
        "buffer_scope": "block_local",
        "loop_origin": "first_interior_block",
    }

    return _conv1d_kernel[2](
        input, weight, tile_width, dma_width, prefetch_groups,
        matmul_width, psum_sets,
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
