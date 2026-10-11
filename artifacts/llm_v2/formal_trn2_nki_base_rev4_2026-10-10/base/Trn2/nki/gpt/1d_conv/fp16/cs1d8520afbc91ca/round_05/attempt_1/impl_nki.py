import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _load_patch(
    x, patch, length, dma_width, position, loop_index,
    runtime_offset, left_edge, right_edge
):
    if left_edge:
        nisa.memset(dst=patch[:, 0:1], value=0.0)
        nisa.dma_copy(
            dst=patch[:, 1:dma_width + 2],
            src=x.ap(
                pattern=[[length, 128], [1, dma_width + 1]],
                offset=position,
            ),
        )
    elif right_edge:
        nisa.dma_copy(
            dst=patch[:, 0:dma_width + 1],
            src=x.ap(
                pattern=[[length, 128], [1, dma_width + 1]],
                offset=position - 1,
            ),
        )
        nisa.memset(
            dst=patch[:, dma_width + 1:dma_width + 2],
            value=0.0,
        )
    elif runtime_offset:
        nisa.dma_copy(
            dst=patch,
            src=x.ap(
                pattern=[[length, 128], [1, dma_width + 2]],
                offset=position - 1,
                scalar_offset=loop_index,
                indirect_dim=1,
            ),
        )
    else:
        nisa.dma_copy(
            dst=patch,
            src=x.ap(
                pattern=[[length, 128], [1, dma_width + 2]],
                offset=position - 1,
            ),
        )


def _compute_patch(
    patch, result, weights, accumulators,
    tile_width, dma_width, matmul_width, psum_sets
):
    for tile in range(dma_width // tile_width):
        t0 = tile * tile_width
        slot = tile % psum_sets

        # Each output tile has one fp32 accumulator covering all three
        # tap-major reduction chunks. No partial result leaves the core.
        for tap in range(3):
            for column in range(tile_width // matmul_width):
                c0 = column * matmul_width
                nisa.nc_matmul(
                    dst=accumulators[:, slot, c0:c0 + matmul_width],
                    stationary=weights[:, tap, :],
                    moving=patch[
                        :, t0 + c0 + tap:t0 + c0 + tap + matmul_width
                    ],
                    accumulate=(tap != 0),
                )

        nisa.tensor_copy(
            dst=result[:, t0:t0 + tile_width],
            src=accumulators[:, slot, :],
        )


def _store_result(
    out, result, length, dma_width, position, loop_index, runtime_offset
):
    if runtime_offset:
        nisa.dma_copy(
            dst=out.ap(
                pattern=[[length, 128], [1, dma_width]],
                offset=position,
                scalar_offset=loop_index,
                indirect_dim=1,
            ),
            src=result,
        )
    else:
        nisa.dma_copy(
            dst=out.ap(
                pattern=[[length, 128], [1, dma_width]],
                offset=position,
            ),
            src=result,
        )


@nki.jit
def _conv1d_kernel(
    input, weight, tile_width, dma_width,
    matmul_width, psum_sets, packets_per_loop
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
    block_width = dma_width * packets_per_loop

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

    # Persistent ping-pong buffers carry the prefetched patch and the
    # completed, deferred output across rolled-loop iterations.
    patch_a = nl.ndarray(
        (128, dma_width + 2), dtype=input.dtype, buffer=nl.sbuf
    )
    patch_b = nl.ndarray(
        (128, dma_width + 2), dtype=input.dtype, buffer=nl.sbuf
    )
    result_a = nl.ndarray(
        (128, dma_width), dtype=input.dtype, buffer=nl.sbuf
    )
    result_b = nl.ndarray(
        (128, dma_width), dtype=input.dtype, buffer=nl.sbuf
    )
    accumulators = nl.ndarray(
        (128, psum_sets, tile_width),
        dtype=nl.float32,
        buffer=nl.psum,
    )

    # Prologue: process the first pair, retaining its second output.
    _load_patch(
        x, patch_a, length, dma_width, base, 0,
        False, pid == 0, False,
    )
    _load_patch(
        x, patch_b, length, dma_width, base + dma_width, 0,
        False, False, False,
    )
    _compute_patch(
        patch_a, result_a, weights, accumulators,
        tile_width, dma_width, matmul_width, psum_sets,
    )
    _load_patch(
        x, patch_a, length, dma_width, base + 2 * dma_width, 0,
        False, False, False,
    )
    _store_result(
        out, result_a, length, dma_width, base, 0, False,
    )
    _compute_patch(
        patch_b, result_b, weights, accumulators,
        tile_width, dma_width, matmul_width, psum_sets,
    )

    interior_base = base + 2 * dma_width

    def body(i):
        for pair in range(packets_per_loop // 2):
            a_position = interior_base + 2 * pair * dma_width
            b_position = a_position + dma_width

            # Prefetch before draining the previous B output, so DMA
            # can start as soon as the old patch's last GEMM completes.
            _load_patch(
                x, patch_b, length, dma_width, b_position, i,
                True, False, False,
            )
            _store_result(
                out, result_b, length, dma_width,
                a_position - dma_width, i, True,
            )

            _compute_patch(
                patch_a, result_a, weights, accumulators,
                tile_width, dma_width, matmul_width, psum_sets,
            )
            _load_patch(
                x, patch_a, length, dma_width,
                a_position + 2 * dma_width, i,
                True, False, False,
            )
            _store_result(
                out, result_a, length, dma_width, a_position, i, True,
            )
            _compute_patch(
                patch_b, result_b, weights, accumulators,
                tile_width, dma_width, matmul_width, psum_sets,
            )

    if span > block_width:
        nl.fori_loop(
            0,
            span - block_width,
            body,
            step=block_width,
        )

    # The remaining packets complete the pipeline and handle the sole
    # right-padding boundary. All other halos come directly from input.
    tail_pairs = (packets_per_loop - 2) // 2
    tail_base = base + span - 2 * tail_pairs * dma_width

    for pair in range(tail_pairs):
        a_position = tail_base + 2 * pair * dma_width
        b_position = a_position + dma_width

        _load_patch(
            x, patch_b, length, dma_width, b_position, 0,
            False, False,
            pair == tail_pairs - 1 and pid == programs - 1,
        )
        _store_result(
            out, result_b, length, dma_width,
            a_position - dma_width, 0, False,
        )
        _compute_patch(
            patch_a, result_a, weights, accumulators,
            tile_width, dma_width, matmul_width, psum_sets,
        )

        if pair < tail_pairs - 1:
            _load_patch(
                x, patch_a, length, dma_width,
                a_position + 2 * dma_width, 0,
                False, False, False,
            )

        _store_result(
            out, result_a, length, dma_width, a_position, 0, False,
        )
        _compute_patch(
            patch_b, result_b, weights, accumulators,
            tile_width, dma_width, matmul_width, psum_sets,
        )

    _store_result(
        out, result_b, length, dma_width,
        base + span - dma_width, 0, False,
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
    matmul_width = 512
    psum_sets = 2
    packets_per_loop = 8
    block_width = dma_width * packets_per_loop

    assert batch == 1 and channels == 128
    assert tuple(weight.shape) == (128, 128, 3)
    assert input.dtype == torch.float16
    assert weight.dtype == input.dtype
    assert stride == 1 and padding == 1 and groups == 1
    assert length >= 2 * block_width
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
        "packets_per_loop": packets_per_loop,
        "block_width": block_width,
        "psum_buffer_sets": psum_sets,
        "input_buffer_sets": 2,
        "output_buffer_sets": 2,
        "deferred_output_packets": 1,
        "prologue_packets": 2,
        "epilogue_packets": packets_per_loop - 2,
        "accumulation_dtype": "float32",
        "buffer_scope": "kernel_local",
        "pipeline": "rolling_ping_pong",
    }

    return _conv1d_kernel[2](
        input, weight, tile_width, dma_width,
        matmul_width, psum_sets, packets_per_loop,
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
