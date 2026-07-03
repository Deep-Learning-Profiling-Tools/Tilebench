"""Triton Stream-K GEMM (Osama et al., PPoPP 2023).

Two-kernel decomposition with the Stream-K scheduler computed INSIDE
each kernel from M/N/K/NUM_SMS/BLOCK_M/BLOCK_N/BLOCK_K. This lets
the autotuner sweep BLOCK_* freely — host code never sees those
values, so there's nothing to keep in sync.

  first_wave  - launches NUM_SMS programs; each owns a contiguous range
                of K-iterations spanning multiple (M, N) output tiles.
                Combines partial tiles via tl.atomic_add into a
                pre-zeroed C — order-independent, no locks needed.
  full_tiles  - data-parallel; one program per leftover whole tile
                after wave-quantization. Writes pre-zeroed disjoint
                regions via TMA store (non-atomic).

Both kernels must share BLOCK_* (otherwise their tile partitions
disagree and the result is corrupt), so a config choice affects both.
The tuner therefore times the **whole two-kernel pipeline**
(first_wave + full_tiles back-to-back) per config. Tuning a single
kernel as a proxy mis-ranks the other's preferences both ways: tuning
first_wave only picked BLOCK_M=64 at m=8192/n=28672 fp32 (~70% slower
end-to-end than default — full_tiles does >99% of the work there);
tuning full_tiles only picked configs that doubled the small shapes,
where first_wave is ~half the work.

The tuner is hand-rolled (mirroring cuTile's CutileAutotuner) because
`triton.autotune` can neither time a two-kernel sequence nor tune on a
SCRATCH buffer — and scratch is required: trial configs partition the
tile space differently, so their stores would leave garbage in regions
the winner's partition expects pre-zeroed.

A/B are loaded through host-side TMA descriptors (TensorDescriptor —
the tutorial-09 pattern, same as matmul_fp32_fp16_fp8; device-side
tl.make_tensor_descriptor is broken). B is consumed transposed (N, K)
so both operand boxes have the K axis innermost (≤128B ⇒ TMA swizzle
fast path). first_wave's output combine stays a pointer-based
tl.atomic_add (TMA stores can't atomic-add); full_tiles' disjoint tiles
go out via TMA store.

`tl.dot(..., input_precision="tf32")` is explicit so the precision
choice mirrors cuTile (which casts fp32 → ct.tfloat32 before ct.mma).
"""
import torch
import triton
from triton import language as tl
from triton.testing import do_bench
from triton.tools.tensor_descriptor import TensorDescriptor


_DEFAULT_CONFIG = {
    "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8,
    "num_warps": 8, "num_stages": 3,
}

# Tile space (BLOCK_M, BLOCK_N, BLOCK_K) is 1:1 with impl_cutile.py's
# _SEARCH_SPACE; num_warps is Triton's scheduling knob like cuTile's
# occupancy.
_SEARCH_SPACE = [
    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8,
     "num_warps": nw, "num_stages": 3}
    for bm in (64, 128)
    for bn in (128, 256)
    for bk in (32, 64)
    for nw in (4, 8)
]

# DT_ID keeps fp16 / bf16 / fp32 compilations unambiguously separate:
# TensorDescriptor args are not torch.Tensors, and C (a real tensor arg)
# is the fp32 staging buffer for every input dtype.
_DT_IDS = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}

# (M, N, K, DT_ID) -> winning config dict, filled by _tune_full_tiles.
_autotune_cache: dict = {}
_last_autotune_config: dict = {}

# The kernel consumes B transposed to (N, K) for the TMA swizzle fast
# path. Build the transposed copy once per input tensor so the unmeasured
# warmup call pays for it. Keyed by tensor IDENTITY, not data_ptr: the
# sweep repeats B's (shape, dtype) across cases (k, n recur while m
# varies) and the caching allocator readily hands a freed data_ptr to the
# next same-sized tensor, so a data_ptr key can return the STALE
# transpose (sporadic ~99%-mismatch verify failures). Weak keys drop the
# entry when the source B is collected.
_bt_cache = torch.utils.weak.WeakTensorKeyDictionary()


def _b_transposed(b: torch.Tensor) -> torch.Tensor:
    bt = _bt_cache.get(b)
    if bt is None:
        bt = b.t().contiguous()   # (N, K) row-major
        _bt_cache[b] = bt
    return bt


@triton.jit
def _swizzle_tile(tile_id, M, N,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  GROUP_M: tl.constexpr):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


@triton.jit
def first_wave(
    a_desc, b_desc, C, M, N, K, NUM_SMS: tl.constexpr,
    stride_cm, stride_cn,
    DT_ID: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
):
    total_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    iters_per_tile = tl.cdiv(K, BLOCK_K)
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS
    total_iters_streamk = streamk_tiles * iters_per_tile
    total_full_iters = total_iters_streamk // NUM_SMS
    total_partial_iters = total_iters_streamk % NUM_SMS

    pid = tl.program_id(0)
    start_iter = pid * total_full_iters + tl.minimum(pid, total_partial_iters)
    last_iter = (pid + 1) * total_full_iters + tl.minimum(pid + 1, total_partial_iters)

    while start_iter < last_iter:
        tile_id = start_iter // iters_per_tile
        rem = iters_per_tile - (start_iter % iters_per_tile)
        end_iter = tl.minimum(start_iter + rem, last_iter)

        pid_m, pid_n = _swizzle_tile(tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_M)
        offs_am = pid_m * BLOCK_M
        offs_bn = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        iter_in_tile = start_iter % iters_per_tile
        for current_iter in range(start_iter, end_iter):
            # TMA boxes zero-fill OOB lanes — no K/M/N masks needed.
            a = a_desc.load([offs_am, iter_in_tile * BLOCK_K])
            b = b_desc.load([offs_bn, iter_in_tile * BLOCK_K])
            acc = tl.dot(a, b.T, acc, input_precision="tf32")
            iter_in_tile += 1

        # Partial-tile combine must be atomic — pointer store, not TMA.
        rm = offs_am + tl.arange(0, BLOCK_M)
        rn = offs_bn + tl.arange(0, BLOCK_N)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        acc_typed = acc.to(C.dtype.element_ty)
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        tl.atomic_add(C_, acc_typed, mask=mask)

        start_iter = end_iter


@triton.jit
def full_tiles(
    a_desc, b_desc, c_desc, M, N, K, NUM_SMS: tl.constexpr,
    DT_ID: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
):
    total_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS

    tile_id = tl.program_id(0) + streamk_tiles
    if tile_id >= total_tiles:
        return

    pid_m, pid_n = _swizzle_tile(tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in tl.range(tl.cdiv(K, BLOCK_K)):
        a = a_desc.load([offs_am, k * BLOCK_K])
        b = b_desc.load([offs_bn, k * BLOCK_K])
        acc = tl.dot(a, b.T, acc, input_precision="tf32")

    # Whole tiles are disjoint — plain (non-atomic) TMA store.
    c_desc.store([offs_am, offs_bn], acc.to(c_desc.dtype))


def _device_sm_count() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _streamk_partition(M, N, BLK_M, BLK_N, NUM_SMS):
    """Mirror the kernel's scheduler. Returns (total_tiles, streamk_tiles)."""
    total_tiles = triton.cdiv(M, BLK_M) * triton.cdiv(N, BLK_N)
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS
    return total_tiles, streamk_tiles


def _launch_full_tiles(a, bt, c, M, N, K, NUM_SMS, dt_id, cfg,
                       blocking_tiles):
    a_desc = TensorDescriptor.from_tensor(a, [cfg["BLOCK_M"], cfg["BLOCK_K"]])
    b_desc = TensorDescriptor.from_tensor(bt, [cfg["BLOCK_N"], cfg["BLOCK_K"]])
    c_desc = TensorDescriptor.from_tensor(c, [cfg["BLOCK_M"], cfg["BLOCK_N"]])
    full_tiles[(blocking_tiles,)](
        a_desc, b_desc, c_desc, M, N, K, NUM_SMS,
        DT_ID=dt_id,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"], ACC_TYPE=tl.float32,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )


def _launch_first_wave(a, bt, c, M, N, K, NUM_SMS, dt_id, cfg):
    a_desc = TensorDescriptor.from_tensor(a, [cfg["BLOCK_M"], cfg["BLOCK_K"]])
    b_desc = TensorDescriptor.from_tensor(bt, [cfg["BLOCK_N"], cfg["BLOCK_K"]])
    first_wave[(NUM_SMS,)](
        a_desc, b_desc, c, M, N, K, NUM_SMS,
        c.stride(0), c.stride(1),
        DT_ID=dt_id,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"], ACC_TYPE=tl.float32,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )


def _tune_pipeline(a, bt, M, N, K, NUM_SMS, dt_id) -> dict:
    """Exhaustively time the first_wave + full_tiles pipeline over
    _SEARCH_SPACE on a scratch buffer (do_bench warmup=1/rep=3 ms — the
    repo-wide autotune budget) and return the winning config. Cached per
    (M, N, K, DT_ID); the engine's warmup calls pay for the sweep, so
    tuning stays out of the measured window."""
    key = (M, N, K, dt_id)
    cached = _autotune_cache.get(key)
    if cached is not None:
        return cached

    scratch = torch.empty((M, N), device=a.device, dtype=torch.float32)
    best_cfg, best_ms, failures = None, float("inf"), []
    for cfg in _SEARCH_SPACE:
        total_tiles, streamk_tiles = _streamk_partition(
            M, N, cfg["BLOCK_M"], cfg["BLOCK_N"], NUM_SMS)
        blocking_tiles = total_tiles - streamk_tiles

        def _pipeline():
            # Scratch values are garbage across trials — only the timing
            # matters, and GEMM/atomic runtime is data-independent.
            _launch_first_wave(a, bt, scratch, M, N, K, NUM_SMS, dt_id, cfg)
            if blocking_tiles > 0:
                _launch_full_tiles(a, bt, scratch, M, N, K, NUM_SMS,
                                   dt_id, cfg, blocking_tiles)

        try:
            ms = do_bench(_pipeline, warmup=1, rep=3)
        except Exception as e:   # OutOfResources / compile errors → skip
            failures.append((cfg, f"{type(e).__name__}: {e}"))
            continue
        if ms < best_ms:
            best_cfg, best_ms = cfg, ms
    if best_cfg is None:
        raise RuntimeError(
            f"streamk_matmul: all {len(_SEARCH_SPACE)} pipeline configs "
            f"failed to run; first error: {failures[0][1] if failures else 'n/a'}")
    if failures:
        print(f"  streamk_matmul triton autotune: skipped {len(failures)} "
              f"failing config(s), e.g. {failures[0][1][:80]}")
    _autotune_cache[key] = best_cfg
    return best_cfg


def run(a: torch.Tensor, b: torch.Tensor,
        block_size: int = None, autotune: bool = False, **kwargs):
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    _, N = b.shape
    NUM_SMS = _device_sm_count()
    DT_ID = _DT_IDS[a.dtype]

    # Mirror impl_cutile.py: route non-fp32 dtypes through an fp32 staging
    # buffer. cuTile *has* to do this (ct.atomic_add doesn't support bf16);
    # Triton matches it so both backends do identical memory traffic and
    # use fp32-precision atomics — otherwise the fp16/bf16 cases would
    # compare Triton's lower-precision fp16 atomic against cuTile's fp32
    # accumulate, plus 2× the C-write bandwidth.
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if a.dtype == torch.float32:
        c = out
        c.zero_()
    else:
        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)

    bt = _b_transposed(b)   # (N, K); cached, so warmup pays the copy

    if autotune:
        cfg = _tune_pipeline(a, bt, M, N, K, NUM_SMS, DT_ID)
        _last_autotune_config.clear()
        _last_autotune_config.update(cfg)
    else:
        cfg = _DEFAULT_CONFIG
    BLK_M, BLK_N = cfg["BLOCK_M"], cfg["BLOCK_N"]

    # ---- Stage 1: first_wave (Stream-K) ----
    _launch_first_wave(a, bt, c, M, N, K, NUM_SMS, DT_ID, cfg)

    # ---- Stage 2: full_tiles (data-parallel) ----
    total_tiles, streamk_tiles = _streamk_partition(M, N, BLK_M, BLK_N, NUM_SMS)
    blocking_tiles = total_tiles - streamk_tiles
    if blocking_tiles > 0:
        _launch_full_tiles(a, bt, c, M, N, K, NUM_SMS, DT_ID, cfg,
                           blocking_tiles)

    if c is not out:
        out.copy_(c)
    return out


def get_last_config() -> dict | None:
    # Hand-rolled tuner (no triton.autotune wrapper to read best_config
    # from) — mutable-dict pattern, same as the cuTile side.
    return dict(_last_autotune_config) if _last_autotune_config else None
