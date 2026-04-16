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


def _launch_histogram(
    input: torch.Tensor,
    histogram: torch.Tensor,
    N: int,
    num_bins: int,
    BLOCK_SIZE: int,
    NUM_PARTIAL: int,
    BLOCK_ROWS: int,
    BLOCK_BINS: int,
    num_warps_partial: int,
    num_warps_reduce: int,
    num_stages: int,
):
    num_partials = min(int(NUM_PARTIAL), triton.cdiv(N, BLOCK_SIZE))
    partial = torch.zeros((num_partials, num_bins), device=input.device, dtype=torch.int32)

    histogram_partial_kernel[(num_partials,)](
        input,
        partial,
        N,
        num_bins,
        num_partials,
        partial.stride(0),
        partial.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps_partial,
        num_stages=num_stages,
    )

    histogram_reduce_kernel[(triton.cdiv(num_bins, BLOCK_BINS),)](
        partial,
        histogram,
        num_partials,
        num_bins,
        partial.stride(0),
        partial.stride(1),
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_BINS=BLOCK_BINS,
        num_warps=num_warps_reduce,
        num_stages=num_stages,
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
):
    global _LAST_CONFIG

    if block_size is not None:
        BLOCK_SIZE = int(block_size)

    cfg = dict(_DEFAULT_CONFIG)
    cfg["BLOCK_SIZE"] = int(BLOCK_SIZE)
    cfg["NUM_PARTIAL"] = int(NUM_PARTIAL)
    cfg["BLOCK_ROWS"] = int(BLOCK_ROWS)
    cfg["BLOCK_BINS"] = int(BLOCK_BINS)

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
        BLOCK_SIZE=cfg["BLOCK_SIZE"],
        NUM_PARTIAL=cfg["NUM_PARTIAL"],
        BLOCK_ROWS=cfg["BLOCK_ROWS"],
        BLOCK_BINS=cfg["BLOCK_BINS"],
        num_warps_partial=cfg["num_warps_partial"],
        num_warps_reduce=cfg["num_warps_reduce"],
        num_stages=cfg["num_stages"],
    )

    _LAST_CONFIG = {
        "BLOCK_SIZE": cfg["BLOCK_SIZE"],
        "NUM_PARTIAL": cfg["NUM_PARTIAL"],
        "BLOCK_ROWS": cfg["BLOCK_ROWS"],
        "BLOCK_BINS": cfg["BLOCK_BINS"],
        "num_warps_partial": cfg["num_warps_partial"],
        "num_warps_reduce": cfg["num_warps_reduce"],
        "num_stages": cfg["num_stages"],
    }
    return histogram


# input and histogram are tensors on the GPU
def solve(input: torch.Tensor, histogram: torch.Tensor, N: int, num_bins: int):
    BLOCK_SIZE = 1024
    NUM_PARTIAL = 256
    BLOCK_ROWS = 64
    BLOCK_BINS = 256

    assert input.is_cuda and histogram.is_cuda
    assert input.dtype == torch.int32
    assert histogram.dtype == torch.int32
    assert input.ndim == 1 and histogram.ndim == 1
    assert input.shape[0] == N
    assert histogram.shape[0] == num_bins

    _launch_histogram(
        input=input,
        histogram=histogram,
        N=N,
        num_bins=num_bins,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_PARTIAL=NUM_PARTIAL,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_BINS=BLOCK_BINS,
        num_warps_partial=4,
        num_warps_reduce=4,
        num_stages=1,
    )


def get_last_config() -> dict | None:
    return _LAST_CONFIG
