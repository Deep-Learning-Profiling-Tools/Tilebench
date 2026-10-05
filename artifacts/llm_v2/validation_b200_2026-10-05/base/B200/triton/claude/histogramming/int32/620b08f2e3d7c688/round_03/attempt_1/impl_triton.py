import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fixed configuration literals (never changed between calls)
# ---------------------------------------------------------------------------
P_CAP = 592          # max number of private partial histograms (4 per SM on 148 SMs)
BLOCK = 2048         # input elements per tile in the counting kernel (8 per thread)
COUNT_WARPS = 8      # warps per counting program
COUNT_MAXNREG = 64   # register cap so that 4 programs (1024 threads) stay resident per SM
PREFETCH_DEPTH = 2   # tiles kept in flight (register double-buffering) during atomics
ZBLOCK = 4096        # elements per zero-fill step of the owning program
RED_BB = 8           # bins per reduce program (one 32B sector per row)
RED_R = 1024         # partial rows per reduce step (covers all P rows in one pass)
RED_WARPS = 8        # warps per reduce program

_CONFIG = {
    "P_CAP": P_CAP,
    "BLOCK": BLOCK,
    "COUNT_WARPS": COUNT_WARPS,
    "COUNT_MAXNREG": COUNT_MAXNREG,
    "PREFETCH_DEPTH": PREFETCH_DEPTH,
    "ZBLOCK": ZBLOCK,
    "RED_BB": RED_BB,
    "RED_R": RED_R,
    "RED_WARPS": RED_WARPS,
}


@triton.jit
def _count_kernel(x_ptr, scratch_ptr, N, num_tiles, num_bins,
                  BLOCK: tl.constexpr, ZBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    base = tl.arange(0, BLOCK)

    # Prefetch the first two tiles of this program's grid-stride sequence
    # before the zero fill so their latency overlaps it.
    o0 = pid * BLOCK + base
    v0 = tl.load(x_ptr + o0, mask=o0 < N, other=-1, eviction_policy="evict_first")
    o1 = (pid + nprog) * BLOCK + base
    v1 = tl.load(x_ptr + o1, mask=o1 < N, other=-1, eviction_policy="evict_first")

    # Stage 1 (fused): the owning program zero-fills its private row.
    row_ptr = scratch_ptr + pid.to(tl.int64) * num_bins
    zeros = tl.zeros((ZBLOCK,), dtype=tl.int32)
    for z in range(0, num_bins, ZBLOCK):
        zo = z + tl.arange(0, ZBLOCK)
        tl.store(row_ptr + zo, zeros, mask=zo < num_bins)
    # Zero fill visible to every thread of this program before counting.
    tl.debug_barrier()

    # Stage 2: private counting, grid-stride over tiles, with tiles t+nprog and
    # t+2*nprog in flight while the atomics of tile t are issued.
    ones = tl.full((BLOCK,), 1, tl.int32)
    for t in range(pid, num_tiles, nprog):
        o2 = (t + 2 * nprog) * BLOCK + base
        v2 = tl.load(x_ptr + o2, mask=o2 < N, other=-1, eviction_policy="evict_first")
        valid = (v0 >= 0) & (v0 < num_bins)
        # Only this program ever touches this row -> CTA-scope relaxed atomics.
        tl.atomic_add(row_ptr + v0, ones, mask=valid, sem="relaxed", scope="cta")
        v0 = v1
        v1 = v2


@triton.jit
def _reduce_kernel(scratch_ptr, out_ptr, P, num_bins,
                   BB: tl.constexpr, R: tl.constexpr):
    # Stage 3: column-wise exact int32 sum over the P private rows.
    pid = tl.program_id(0)
    cols = pid * BB + tl.arange(0, BB)
    cmask = cols < num_bins
    acc = tl.zeros((R, BB), dtype=tl.int32)
    for r0 in range(0, P, R):
        rows = r0 + tl.arange(0, R)
        ptrs = scratch_ptr + rows[:, None].to(tl.int64) * num_bins + cols[None, :]
        m = (rows[:, None] < P) & cmask[None, :]
        acc += tl.load(ptrs, mask=m, other=0)
    tot = tl.sum(acc, axis=0)
    tl.store(out_ptr + cols, tot, mask=cmask)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    assert num_bins >= 1
    x = input.contiguous()  # no-op for the contiguous benchmark input
    assert x.data_ptr() == input.data_ptr()

    num_tiles = triton.cdiv(N, BLOCK)
    P = max(1, min(P_CAP, num_tiles))

    out = torch.empty((num_bins,), dtype=torch.int32, device=input.device)
    # Zero-filled in-kernel by each owning program (stage 1 fused into stage 2).
    scratch = torch.empty((P, num_bins), dtype=torch.int32, device=input.device)

    _count_kernel[(P,)](
        x, scratch, N, num_tiles, num_bins,
        BLOCK=BLOCK, ZBLOCK=ZBLOCK,
        num_warps=COUNT_WARPS,
        maxnreg=COUNT_MAXNREG,
    )

    grid_r = (triton.cdiv(num_bins, RED_BB),)
    _reduce_kernel[grid_r](
        scratch, out, P, num_bins,
        BB=RED_BB, R=RED_R,
        num_warps=RED_WARPS,
    )
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
