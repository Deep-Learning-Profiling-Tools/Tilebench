import torch
import cuda.tile as ct


@ct.kernel(num_ctas=1, occupancy=4)
def _count_private(
    input,
    partials,
    N: ct.Constant[int],
    P: ct.Constant[int],
    CHUNK: ct.Constant[int],
):
    pid = ct.bid(0)

    for chunk_id in range(pid, N // CHUNK, P):
        values = ct.load(
            input,
            (chunk_id,),
            (CHUNK,),
            allow_tma=False,
        )
        ct.atomic_add(
            partials,
            (pid, values),
            1,
            check_bounds=True,
            memory_order=ct.MemoryOrder.RELAXED,
            memory_scope=ct.MemoryScope.BLOCK,
        )


@ct.kernel(num_ctas=1, occupancy=4)
def _reduce_private(
    partials,
    output,
    ROWS: ct.Constant[int],
    BINS: ct.Constant[int],
):
    bin_tile = ct.bid(0)
    values = ct.load(
        partials,
        (0, bin_tile),
        (ROWS, BINS),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    counts = ct.sum(values, axis=0)
    ct.store(output, (bin_tile,), counts, allow_tma=False)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    assert input.is_cuda
    assert input.dtype == torch.int32
    assert len(input.shape) == 1
    assert input.shape[0] == N
    assert input.stride() == (1,)
    assert N == 67108864
    assert num_bins == 4096

    partial_cap = 512
    chunk = 2048
    reduce_rows = 512
    reduce_bins = 16
    P = min(partial_cap, ct.cdiv(N, chunk))

    output = torch.empty((num_bins,), dtype=torch.int32, device=input.device)
    partials = torch.zeros(
        (P, num_bins), dtype=torch.int32, device=input.device
    )
    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (P,),
        _count_private,
        (input, partials, N, P, chunk),
    )
    ct.launch(
        stream,
        (ct.cdiv(num_bins, reduce_bins),),
        _reduce_private,
        (partials, output, reduce_rows, reduce_bins),
    )
    return output


def get_last_config() -> dict:
    return {
        "partial_cap": 512,
        "count_chunk": 2048,
        "count_num_ctas": 1,
        "count_occupancy": 4,
        "atomic_order": "relaxed",
        "atomic_scope": "block",
        "scratch_initialization": "torch.zeros",
        "reduce_rows": 512,
        "reduce_bins": 16,
        "reduce_num_ctas": 1,
        "reduce_occupancy": 4,
        "allow_tma": False,
    }
