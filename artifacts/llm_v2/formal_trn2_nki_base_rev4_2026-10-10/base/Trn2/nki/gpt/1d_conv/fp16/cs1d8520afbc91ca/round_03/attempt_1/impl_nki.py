import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _process_block(
    x, out, weights, length, tile_width, prefetch_tiles, matmul_width,
    start, loop_index, runtime_offset, left_edge, right_edge
):
    # These buffers are local to this invocation, including when invoked
    # inside the rolled loop. No PSUM value crosses a loop boundary.
    patches = nl.ndarray(
        (128, prefetch_tiles, tile_width + 2),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )
    results = nl.ndarray(
        (128, prefetch_tiles, tile_width),
        dtype=x.dtype,
        buffer=nl.sbuf,
    )
    accumulator = nl.ndarray(
        (128, tile_width),
        dtype=nl.float32,
        buffer=nl.psum,
    )

    for tile in range(prefetch_tiles):
        position = start + tile * tile_width

        if left_edge and tile == 0:
            nisa.memset(dst=patches[:, tile, 0:1], value=0.0)
            nisa.dma_copy(
                dst=patches[:, tile, 1:tile_width + 2],
                src=x.ap(
                    pattern=[[length, 128], [1, tile_width + 1]],
                    offset=position,
                ),
            )
        elif right_edge and tile == prefetch_tiles - 1:
            nisa.dma_copy(
                dst=patches[:, tile, 0:tile_width + 1],
                src=x.ap(
                    pattern=[[length, 128], [1, tile_width + 1]],
                    offset=position - 1,
                ),
            )
            nisa.memset(
                dst=patches[:, tile, tile_width + 1:tile_width + 2],
                value=0.0,
            )
        elif runtime_offset:
            nisa.dma_copy(
                dst=patches[:, tile, :],
                src=x.ap(
                    pattern=[[length, 128], [1, tile_width + 2]],
                    offset=position - 1,
                    scalar_offset=loop_index,
                    indirect_dim=1,
                ),
            )
        else:
            nisa.dma_copy(
                dst=patches[:, tile, :],
                src=x.ap(
                    pattern=[[length, 128], [1, tile_width + 2]],
                    offset=position - 1,
                ),
            )

    for tile in range(prefetch_tiles):
        # Tap-major chunks enumerate the complete 384-element reduction.
        # The first tap overwrites each PSUM region; subsequent taps
        # accumulate in fp32 without an intermediate rounding.
        for tap in range(3):
            for column in range(tile_width // matmul_width):
                c0 = column * matmul_width
                nisa.nc_matmul(
                    dst=accumulator[:, c0:c0 + matmul_width],
                    stationary=weights[:, tap, :],
                    moving=patches[
                        :, tile, c0 + tap:c0 + tap + matmul_width
                    ],
                    accumulate=(tap != 0),
                )

        nisa.tensor_copy(
            dst=results[:, tile, :],
            src=accumulator,
        )

        position = start + tile * tile_width
        if runtime_offset:
            nisa.dma_copy(
                dst=out.ap(
                    pattern=[[length, 128], [1, tile_width]],
                    offset=position,
                    scalar_offset=loop_index,
                    indirect_dim=1,
                ),
                src=results[:, tile, :],
            )
        else:
            nisa.dma_copy(
                dst=out.ap(
                    pattern=[[length, 128], [1, tile_width]],
                    offset=position,
                ),
                src=results[:, tile, :],
            )


@nki.jit
def _conv1d_kernel(input, weight, tile_width, prefetch_tiles, matmul_width):
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
    block_width = tile_width * prefetch_tiles

    x = input[0]
    out = output[0]

    # The small weight tensor is gathered and transposed entirely on chip.
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
        x, out, weights, length, tile_width, prefetch_tiles, matmul_width,
        base, 0, False, pid == 0, False,
    )

    interior_base = base + block_width

    def body(i):
        _process_block(
            x, out, weights,
            length, tile_width, prefetch_tiles, matmul_width,
            interior_base, i, True, False, False,
        )

    if span > 2 * block_width:
        nl.fori_loop(
            0,
            span - 2 * block_width,
            body,
            step=block_width,
        )

    _process_block(
        x, out, weights, length, tile_width, prefetch_tiles, matmul_width,
        base + span - block_width, 0, False,
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
    tile_width = 1024
    prefetch_tiles = 16
    matmul_width = 512
    block_width = tile_width * prefetch_tiles

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
        "prefetch_tiles": prefetch_tiles,
        "block_width": block_width,
        "psum_buffer_sets": 1,
        "accumulation_dtype": "float32",
        "buffer_scope": "block_local",
        "loop_origin": "first_interior_block",
    }

    return _conv1d_kernel[2](
        input, weight, tile_width, prefetch_tiles, matmul_width
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
