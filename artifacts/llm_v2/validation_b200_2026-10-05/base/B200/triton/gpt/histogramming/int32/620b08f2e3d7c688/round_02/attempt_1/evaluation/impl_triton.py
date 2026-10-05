import torch
import triton
import triton.language as tl


@triton.jit
def _count_partials(
    Input,
    Partials,
    N: tl.constexpr,
    NUM_BINS: tl.constexpr,
    P: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = Partials + pid * NUM_BINS
    bins = tl.arange(0, NUM_BINS)

    # Only this program initializes and updates its private row.
    tl.store(row + bins, tl.zeros((NUM_BINS,), tl.int32))
    tl.debug_barrier()

    offsets = tl.arange(0, CHUNK_SIZE)
    for chunk in tl.range(
        pid, tl.cdiv(N, CHUNK_SIZE), P, num_stages=2
    ):
        indices = chunk * CHUNK_SIZE + offsets
        values = tl.load(
            Input + indices,
            mask=indices < N,
            other=-1,
            cache_modifier=".cg",
        )
        valid = (indices < N) & (values >= 0) & (values < NUM_BINS)
        tl.atomic_add(
            row + values,
            1,
            mask=valid,
            sem="relaxed",
            scope="cta",
        )


@triton.jit
def _reduce_partials(
    Partials,
    Output,
    P: tl.constexpr,
    NUM_BINS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BIN_BLOCK: tl.constexpr,
):
    local_rows = tl.arange(0, ROW_BLOCK)
    bins = tl.program_id(0) * BIN_BLOCK + tl.arange(0, BIN_BLOCK)
    accumulated = tl.zeros((ROW_BLOCK, BIN_BLOCK), tl.int32)

    for start in range(0, P, ROW_BLOCK):
        rows = start + local_rows
        values = tl.load(
            Partials + rows[:, None] * NUM_BINS + bins[None, :],
            mask=(rows[:, None] < P) & (bins[None, :] < NUM_BINS),
            other=0,
        )
        accumulated = accumulated + values

    counts = tl.sum(accumulated, axis=0, dtype=tl.int32)
    tl.store(Output + bins, counts, mask=bins < NUM_BINS)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    assert input.is_cuda
    assert len(input.shape) == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    assert input.stride(0) == 1
    assert N == 67108864
    assert num_bins == 4096

    P = min(1184, triton.cdiv(N, 1024))
    output = torch.empty((num_bins,), device=input.device, dtype=torch.int32)
    partials = torch.empty((P, num_bins), device=input.device, dtype=torch.int32)

    _count_partials[(P,)](
        input,
        partials,
        N=N,
        NUM_BINS=num_bins,
        P=P,
        CHUNK_SIZE=1024,
        num_warps=4,
        num_stages=1,
    )
    _reduce_partials[(triton.cdiv(num_bins, 16),)](
        partials,
        output,
        P=P,
        NUM_BINS=num_bins,
        ROW_BLOCK=256,
        BIN_BLOCK=16,
        num_warps=4,
        num_stages=1,
    )
    return output


def get_last_config() -> dict:
    return {
        "partial_cap": 1184,
        "chunk_size": 1024,
        "count_num_warps": 4,
        "count_num_stages": 1,
        "count_loop_stages": 2,
        "count_method": "private_global_atomic_add",
        "atomic_semantics": "relaxed",
        "atomic_scope": "cta",
        "reduce_row_block": 256,
        "reduce_bin_block": 16,
        "reduce_num_warps": 4,
        "reduce_num_stages": 1,
        "scratch_initialization": "owning_program",
    }
