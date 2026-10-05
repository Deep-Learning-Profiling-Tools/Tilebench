import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# ---------------------------------------------------------------------------
# Fixed configuration literals (never changed between calls)
# ---------------------------------------------------------------------------
P_CAP = 296          # max number of private partial histograms (2 per SM on 148 SMs)
ROW_W = 32           # int32 elements per descriptor row (128 B = one TMA swizzle span)
BLOCK_ROWS = 128     # descriptor rows per TMA tile -> 4096 elements (16 KB) per tile
GRAIN_ROWS = 32      # partition granularity in rows (1024 elements)
COUNT_WARPS = 16     # warps per counting program
COUNT_MAXNREG = 64   # register cap so that 2 programs (1024 threads) stay resident per SM
COUNT_STAGES = 5     # TMA pipeline depth of the input tiles (async bulk copies into SMEM)
TAIL_BLOCK = 1024    # elements per step of the (< 1024 element) tail
ZBLOCK = 4096        # elements per zero-fill step of the owning program
RED_BB = 16          # bins per reduce program (64B per partial row)
RED_R = 512          # partial rows per reduce step (covers all P rows in one pass)
RED_WARPS = 8        # warps per reduce program

_CONFIG = {
    "P_CAP": P_CAP,
    "ROW_W": ROW_W,
    "BLOCK_ROWS": BLOCK_ROWS,
    "GRAIN_ROWS": GRAIN_ROWS,
    "COUNT_WARPS": COUNT_WARPS,
    "COUNT_MAXNREG": COUNT_MAXNREG,
    "COUNT_STAGES": COUNT_STAGES,
    "TAIL_BLOCK": TAIL_BLOCK,
    "ZBLOCK": ZBLOCK,
    "RED_BB": RED_BB,
    "RED_R": RED_R,
    "RED_WARPS": RED_WARPS,
}


@triton.jit
def _count_kernel(x_desc, x_ptr, scratch_ptr, N, num_grains, P, num_bins,
                  BLOCK_ROWS: tl.constexpr, ROW_W: tl.constexpr,
                  GRAIN_ROWS: tl.constexpr, TAIL_BLOCK: tl.constexpr,
                  ZBLOCK: tl.constexpr, STAGES: tl.constexpr):
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
    ones = tl.full((BLOCK_ROWS, ROW_W), 1, tl.int32)
    rbase = tl.arange(0, BLOCK_ROWS)

    # Stage 2: balanced contiguous partition in units of GRAIN_ROWS rows.
    g0 = (pid * num_grains) // P
    g1 = ((pid + 1) * num_grains) // P
    r_start = g0 * GRAIN_ROWS
    r_end = g1 * GRAIN_ROWS
    for r in tl.range(r_start, r_end, BLOCK_ROWS, num_stages=STAGES):
        v = x_desc.load([r, 0])                       # [BLOCK_ROWS, ROW_W] via TMA
        rows = r + rbase
        rmask = rows < r_end                          # rows past this program's share
        # single unsigned compare covers v < 0 and v >= num_bins
        valid = rmask[:, None] & (v.to(tl.uint32, bitcast=True) < nb)
        # Only this program ever touches this row -> CTA-scope relaxed atomics.
        tl.atomic_add(row_ptr + v, ones, mask=valid, sem="relaxed", scope="cta")

    # Tail (< 1024 elements beyond the last full grain): last program, own row.
    if pid == P - 1:
        tail_start = num_grains * (GRAIN_ROWS * ROW_W)
        tbase = tl.arange(0, TAIL_BLOCK)
        tones = tl.full((TAIL_BLOCK,), 1, tl.int32)
        for s in range(tail_start, N, TAIL_BLOCK):
            offs = s + tbase
            m = offs < N
            tv = tl.load(x_ptr + offs, mask=m, other=0)
            tvalid = m & (tv.to(tl.uint32, bitcast=True) < nb)
            tl.atomic_add(row_ptr + tv, tones, mask=tvalid, sem="relaxed", scope="cta")


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
    assert x.data_ptr() % 16 == 0

    num_rows = N // ROW_W
    num_grains = num_rows // GRAIN_ROWS
    P = max(1, min(P_CAP, num_grains))
    assert (P + RED_R) * num_bins < 2 ** 31

    out = torch.empty((num_bins,), dtype=torch.int32, device=input.device)
    # Zero-filled in-kernel by each owning program (stage 1 fused into stage 2).
    scratch = torch.empty((P, num_bins), dtype=torch.int32, device=input.device)

    # Host-side TMA descriptor over the input viewed as [N // ROW_W, ROW_W]
    # (metadata only; the tail N % (GRAIN_ROWS*ROW_W) is read via plain loads).
    x_desc = TensorDescriptor(x, [max(1, num_rows), ROW_W], [ROW_W, 1],
                              [BLOCK_ROWS, ROW_W])

    _count_kernel[(P,)](
        x_desc, x, scratch, N, num_grains, P, num_bins,
        BLOCK_ROWS=BLOCK_ROWS, ROW_W=ROW_W, GRAIN_ROWS=GRAIN_ROWS,
        TAIL_BLOCK=TAIL_BLOCK, ZBLOCK=ZBLOCK, STAGES=COUNT_STAGES,
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
