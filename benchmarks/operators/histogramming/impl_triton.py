import torch
import triton
import triton.language as tl


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
def _histogram_partial_kernel(
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

    for chunk_start in tl.range(pid * BLOCK_SIZE, N, num_partials * BLOCK_SIZE):
        idx = chunk_start + offs
        mask = idx < N

        vals = tl.load(input_ptr + idx, mask=mask, other=0)
        valid = mask & (vals >= 0) & (vals < num_bins)

        safe_bins = tl.where(valid, vals, 0)
        tl.atomic_add(row_base + safe_bins * stride_pb, one, mask=valid)


@triton.jit
def _histogram_reduce_kernel(
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


_histogram_partial_kernel_autotuned = triton.autotune(
    configs=[
        # Shrunk from 12 cfgs to 4 to match cuTile-side shrink (impl_cutile.py).
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=1)
        for bs in [1024, 2048]
        for nw in [4, 8]
    ],
    key=["N", "num_bins"],
    # Stage 1 uses tl.atomic_add into partial_ptr — each autotune-sweep run
    # accumulates into the same buffer. Zero it before every cfg trial so the
    # final replay sees a clean buffer.
    reset_to_zero=["partial_ptr"],
)(_histogram_partial_kernel)


# Stage 2's K-loop is a true `tl.load → reduce` pipeline, so num_stages
# sweep can give real load/compute overlap (unlike stage 1, which is
# atomic-write-bound and pins num_stages=1).
_histogram_reduce_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_ROWS": br, "BLOCK_BINS": bb},
            num_warps=nw,
            num_stages=ns,
        )
        # Shrunk from ~135 cfgs to 8 to match cuTile-side shrink.
        for br in [64, 128]
        for bb in [64, 128]
        for nw in [4, 8]
        for ns in [2]
        if br * bb <= 256 * 128
    ],
    key=["num_partials", "num_bins"],
    warmup=3,
    rep=10,
)(_histogram_reduce_kernel)


def run(input: torch.Tensor, N: int, num_bins: int,
        block_size: int = None, autotune: bool = False, **kwargs):
    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    assert num_bins >= 1

    cfg = _DEFAULT_CONFIG
    BLOCK_SIZE = int(block_size) if block_size is not None else cfg["BLOCK_SIZE"]
    NUM_PARTIAL = cfg["NUM_PARTIAL"]
    BLOCK_ROWS = cfg["BLOCK_ROWS"]
    BLOCK_BINS = cfg["BLOCK_BINS"]

    input = input.contiguous()
    histogram = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    # Allocate + zero the partial buffer per call. The TileBench engine has no
    # setup hook, so this memset gets recorded into the CUDA graph and inflates
    # absolute latency by a few us. We accept the cost because impl_cutile.py
    # does the exact same `torch.zeros(...)`, keeping the Triton-vs-cuTile
    # comparison symmetric.
    num_partials = min(NUM_PARTIAL, triton.cdiv(N, BLOCK_SIZE))
    partial = torch.zeros((num_partials, num_bins), device=input.device, dtype=torch.int32)

    if autotune:
        _histogram_partial_kernel_autotuned[(num_partials,)](
            input, partial, N, num_bins, num_partials,
            partial.stride(0), partial.stride(1),
        )
        grid_reduce = lambda meta: (triton.cdiv(num_bins, meta["BLOCK_BINS"]),)
        _histogram_reduce_kernel_autotuned[grid_reduce](
            partial, histogram, num_partials, num_bins,
            partial.stride(0), partial.stride(1),
        )
    else:
        _histogram_partial_kernel[(num_partials,)](
            input, partial, N, num_bins, num_partials,
            partial.stride(0), partial.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=cfg["num_warps_partial"],
            num_stages=cfg["num_stages"],
        )
        _histogram_reduce_kernel[(triton.cdiv(num_bins, BLOCK_BINS),)](
            partial, histogram, num_partials, num_bins,
            partial.stride(0), partial.stride(1),
            BLOCK_ROWS=BLOCK_ROWS,
            BLOCK_BINS=BLOCK_BINS,
            num_warps=cfg["num_warps_reduce"],
            num_stages=cfg["num_stages"],
        )

    return histogram


def get_last_config() -> dict | None:
    cfg_partial = getattr(_histogram_partial_kernel_autotuned, "best_config", None)
    cfg_reduce = getattr(_histogram_reduce_kernel_autotuned, "best_config", None)
    if cfg_partial is None and cfg_reduce is None:
        return None

    result = {}
    if cfg_partial is not None:
        result.update({
            "partial_BLOCK_SIZE": cfg_partial.kwargs["BLOCK_SIZE"],
            "partial_num_warps":  cfg_partial.num_warps,
        })
    if cfg_reduce is not None:
        result.update({
            "reduce_BLOCK_ROWS":  cfg_reduce.kwargs["BLOCK_ROWS"],
            "reduce_BLOCK_BINS":  cfg_reduce.kwargs["BLOCK_BINS"],
            "reduce_num_warps":   cfg_reduce.num_warps,
            "reduce_num_stages":  cfg_reduce.num_stages,
        })
    return result
