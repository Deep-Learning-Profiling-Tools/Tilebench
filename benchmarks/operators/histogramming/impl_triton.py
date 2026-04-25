import torch
import triton
import triton.language as tl

_LAST_CONFIG = None

_DEFAULT_CONFIG = {
    "BLOCK_SIZE": 1024,
    "NUM_PARTIAL": 256,
    "BLOCK_ROWS": 64,
    "BLOCK_BINS": 256,
    "num_warps_partial": 4,
    "num_warps_reduce": 4,
    "num_stages": 1,
}


@triton.jit
def histogram_partial_kernel(
    input_ptr,
    partial_ptr,
    N,
    num_bins,
    num_partials,
    stride_pr,
    stride_pb,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    row_base = partial_ptr + pid * stride_pr
    one = tl.full((BLOCK_SIZE,), 1, dtype=tl.int32)

    chunk_start = pid * BLOCK_SIZE
    chunk_step = num_partials * BLOCK_SIZE

    while chunk_start < N:
        idx = chunk_start + offs
        mask = idx < N

        vals = tl.load(input_ptr + idx, mask=mask, other=0)
        valid = mask & (vals >= 0) & (vals < num_bins)

        safe_bins = tl.where(valid, vals, 0)
        tl.atomic_add(row_base + safe_bins * stride_pb, one, mask=valid)

        chunk_start += chunk_step


@triton.jit
def histogram_reduce_kernel(
    partial_ptr,
    hist_ptr,
    num_partials,
    num_bins,
    stride_pr,
    stride_pb,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_BINS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs_b = pid_b * BLOCK_BINS + tl.arange(0, BLOCK_BINS)

    acc = tl.zeros((BLOCK_BINS,), dtype=tl.int32)

    for row_start in tl.range(0, num_partials, BLOCK_ROWS):
        offs_r = row_start + tl.arange(0, BLOCK_ROWS)

        ptrs = partial_ptr + offs_r[:, None] * stride_pr + offs_b[None, :] * stride_pb
        mask = (offs_r[:, None] < num_partials) & (offs_b[None, :] < num_bins)

        vals = tl.load(ptrs, mask=mask, other=0)
        acc += tl.sum(vals, axis=0)

    tl.store(hist_ptr + offs_b, acc, mask=offs_b < num_bins)


histogram_partial_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=1)
        for bs in [512, 1024, 2048]
        for nw in [2, 4, 8]
    ],
    key=["N", "num_bins"],
)(histogram_partial_kernel)


histogram_reduce_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_ROWS": br,
                "BLOCK_BINS": bb,
            },
            num_warps=nw,
            num_stages=1,
        )
        for br in [32, 64, 128]
        for bb in [64, 128, 256]
        for nw in [2, 4, 8]
    ],
    key=["num_partials", "num_bins"],
)(histogram_reduce_kernel)


def _launch_histogram(
    input: torch.Tensor,
    histogram: torch.Tensor,
    N: int,
    num_bins: int,
    BLOCK_SIZE: int,
    NUM_PARTIAL: int,
    BLOCK_ROWS: int,
    BLOCK_BINS: int,
    autotune: bool,
):
    num_partials = min(int(NUM_PARTIAL), triton.cdiv(N, BLOCK_SIZE))
    partial = torch.zeros((num_partials, num_bins), device=input.device, dtype=torch.int32)

    if autotune:
        histogram_partial_kernel_autotuned[(num_partials,)](
            input,
            partial,
            N,
            num_bins,
            num_partials,
            partial.stride(0),
            partial.stride(1),
        )
    else:
        histogram_partial_kernel[(num_partials,)](
            input,
            partial,
            N,
            num_bins,
            num_partials,
            partial.stride(0),
            partial.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=_DEFAULT_CONFIG["num_warps_partial"],
            num_stages=_DEFAULT_CONFIG["num_stages"],
        )

    if autotune:
        grid_reduce = lambda meta: (
            triton.cdiv(num_bins, meta["BLOCK_BINS"]),
        )
        histogram_reduce_kernel_autotuned[grid_reduce](
            partial,
            histogram,
            num_partials,
            num_bins,
            partial.stride(0),
            partial.stride(1),
        )
    else:
        histogram_reduce_kernel[(triton.cdiv(num_bins, BLOCK_BINS),)](
            partial,
            histogram,
            num_partials,
            num_bins,
            partial.stride(0),
            partial.stride(1),
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_BINS=BLOCK_BINS,
            num_warps=_DEFAULT_CONFIG["num_warps_reduce"],
            num_stages=_DEFAULT_CONFIG["num_stages"],
        )


def run(
    input,
    N: int,
    num_bins: int,
    BLOCK_SIZE: int = 1024,
    NUM_PARTIAL: int = 256,
    BLOCK_ROWS: int = 64,
    BLOCK_BINS: int = 256,
    block_size: int = None,
    autotune: bool = False,
    **kwargs,
):
    global _LAST_CONFIG

    if block_size is not None:
        BLOCK_SIZE = int(block_size)

    BLOCK_SIZE = int(BLOCK_SIZE)
    NUM_PARTIAL = int(NUM_PARTIAL)
    BLOCK_ROWS = int(BLOCK_ROWS)
    BLOCK_BINS = int(BLOCK_BINS)

    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    assert num_bins >= 1

    input = input.contiguous()
    histogram = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    _launch_histogram(
        input=input,
        histogram=histogram,
        N=N,
        num_bins=num_bins,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_PARTIAL=NUM_PARTIAL,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_BINS=BLOCK_BINS,
        autotune=autotune,
    )

    _LAST_CONFIG = {
        "BLOCK_SIZE": BLOCK_SIZE,
        "NUM_PARTIAL": NUM_PARTIAL,
        "BLOCK_ROWS": BLOCK_ROWS,
        "BLOCK_BINS": BLOCK_BINS,
        "autotune": bool(autotune),
    }
    return histogram


def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    BLOCK_SIZE = 1024
    NUM_PARTIAL = 256
    BLOCK_ROWS = 64
    BLOCK_BINS = 256

    _launch_histogram(
        input=input,
        histogram=histogram,
        N=N,
        num_bins=num_bins,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_PARTIAL=NUM_PARTIAL,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_BINS=BLOCK_BINS,
        autotune=False,
    )


def get_last_config() -> dict | None:
    cfg_partial = getattr(histogram_partial_kernel_autotuned, "best_config", None)
    cfg_reduce = getattr(histogram_reduce_kernel_autotuned, "best_config", None)

    if cfg_partial is None and cfg_reduce is None:
        return _LAST_CONFIG

    result = {}
    if cfg_partial is not None:
        result.update({
            "partial_BLOCK_SIZE": cfg_partial.kwargs["BLOCK_SIZE"],
            "partial_num_warps": cfg_partial.num_warps,
            "partial_num_stages": cfg_partial.num_stages,
        })
    if cfg_reduce is not None:
        result.update({
            "reduce_BLOCK_ROWS": cfg_reduce.kwargs["BLOCK_ROWS"],
            "reduce_BLOCK_BINS": cfg_reduce.kwargs["BLOCK_BINS"],
            "reduce_num_warps": cfg_reduce.num_warps,
            "reduce_num_stages": cfg_reduce.num_stages,
        })
    return result