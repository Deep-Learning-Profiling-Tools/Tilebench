import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# ---------------------------------------------------------------------------
# Fixed configuration literals (never changed between calls)
# ---------------------------------------------------------------------------
P_CAP = 296          # max number of private partial histograms (2 per SM on 148 SMs)
BW = 32              # inner width of the 2D TMA view of the input (32 x int32 = 128 B)
BM = 128             # rows per TMA box -> 4096 input elements per loop step
GRAIN_ROWS = 32      # partition granularity in rows (1024 elements)
COUNT_WARPS = 16     # warps per counting program
COUNT_MAXNREG = 64   # register cap so that 2 programs stay resident per SM
COUNT_STAGES = 4     # TMA pipeline depth (16 KB per stage)
TBLOCK = 4096        # elements per step of the (tiny) tail loop
ZBLOCK = 4096        # elements per zero-fill step of the owning program
RED_BB = 16          # bins per reduce program (64B per partial row)
RED_R = 512          # partial rows per reduce step (covers all P rows in one pass)
RED_WARPS = 8        # warps per reduce program

_CONFIG = {
    "P_CAP": P_CAP,
    "BW": BW,
    "BM": BM,
    "GRAIN_ROWS": GRAIN_ROWS,
    "COUNT_WARPS": COUNT_WARPS,
    "COUNT_MAXNREG": COUNT_MAXNREG,
    "COUNT_STAGES": COUNT_STAGES,
    "TBLOCK": TBLOCK,
    "ZBLOCK": ZBLOCK,
    "RED_BB": RED_BB,
    "RED_R": RED_R,
    "RED_WARPS": RED_WARPS,
}


@triton.jit
def _count_kernel(x_desc, x_ptr, scratch_ptr, N, num_grains, P, num_bins,
                  BM: tl.constexpr, BW: tl.constexpr, GRAIN_ROWS: tl.constexpr,
                  TBLOCK: tl.constexpr, ZBLOCK: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)

    # Stage 1 (fused): the owning program zero-fills its private row.
    row_ptr = scratch_ptr + pid.to(tl.int64) * num_bins
    zeros = tl.zeros((ZBLOCK,), dtype=tl.int32)
    for z in range(0, num_bins, ZBLOCK):
        zo = z + tl.arange(0, ZBLOCK)
        tl.store(row_ptr + zo, zeros, mask=zo < num_bins)
    # Zero fill visible to every thread of this program before counting.
    tl.debug_barrier()

    nb = num_bins.to(tl.uint32)
    ones = tl.full((BM, BW), 1, tl.int32)
    rbase = tl.arange(0, BM)

    # Stage 2: balanced contiguous partition in units of GRAIN_ROWS rows of BW elements.
    g0 = (pid * num_grains) // P
    g1 = ((pid + 1) * num_grains) // P
    r_start = g0 * GRAIN_ROWS
    r_end = g1 * GRAIN_ROWS
    for r in tl.range(r_start, r_end, BM, num_stages=STAGES):
        v = x_desc.load([r, 0])                      # [BM, BW] via TMA
        rows = r + rbase
        # rows beyond this program's range (or TMA zero padding) are excluded;
        # single unsigned compare covers v < 0 and v >= num_bins
        valid = (rows < r_end)[:, None] & (v.to(tl.uint32, bitcast=True) < nb)
        # Only this program ever touches this row -> CTA-scope relaxed atomics.
        tl.atomic_add(row_ptr + v, ones, mask=valid, sem="relaxed", scope="cta")

    # Tail (< GRAIN elements beyond the last full grain), last program, own row.
    if pid == P - 1:
        tail_start = num_grains * (GRAIN_ROWS * BW)
        base = tl.arange(0, TBLOCK)
        ones1 = tl.full((TBLOCK,), 1, tl.int32)
        for s in range(tail_start, N, TBLOCK):
            offs = s + base
            m = offs < N
            v1 = tl.load(x_ptr + offs, mask=m, other=0)
            valid1 = m & (v1.to(tl.uint32, bitcast=True) < nb)
            tl.atomic_add(row_ptr + v1, ones1, mask=valid1, sem="relaxed", scope="cta")


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
        ptrs = scratch_ptr + rows[:, None] * num_bins + cols[None, :]
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
    assert N < 2 ** 31
    x = input.contiguous()  # no-op for the contiguous benchmark input
    assert x.data_ptr() == input.data_ptr()

    GRAIN = GRAIN_ROWS * BW
    M = N // BW                      # full rows of the 2D TMA view
    num_grains = N // GRAIN
    P = max(1, min(P_CAP, num_grains))
    assert (P + RED_R) * num_bins < 2 ** 31

    out = torch.empty((num_bins,), dtype=torch.int32, device=input.device)
    # Zero-filled in-kernel by each owning program (stage 1 fused into stage 2).
    scratch = torch.empty((P, num_bins), dtype=torch.int32, device=input.device)

    # 2D TMA descriptor over the input storage: [M, BW] row-major, boxes of [BM, BW].
    x_desc = TensorDescriptor(x, [max(M, 1), BW], [BW, 1], [BM, BW])

    _count_kernel[(P,)](
        x_desc, x, scratch, N, num_grains, P, num_bins,
        BM=BM, BW=BW, GRAIN_ROWS=GRAIN_ROWS,
        TBLOCK=TBLOCK, ZBLOCK=ZBLOCK, STAGES=COUNT_STAGES,
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
