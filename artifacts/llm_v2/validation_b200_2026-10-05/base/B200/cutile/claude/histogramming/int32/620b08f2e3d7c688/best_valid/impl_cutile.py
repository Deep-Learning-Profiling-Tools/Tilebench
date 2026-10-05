import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

# ---------------------------------------------------------------------------
# Fixed configuration literals
# ---------------------------------------------------------------------------
_P_CAP = 1184         # max number of private partial histograms (148 SMs x 8)
_TILE = 1024          # input elements per counting step
_COUNT_OCC = 8        # occupancy hint for the counting kernel
_PREFETCH = 3         # tiles in flight ahead of the tile being counted
_ZERO_TILE_CAP = 4096 # max columns zeroed per store in the in-kernel fill
_RED_BINS = 8         # bins handled per reduce program (one 32B sector per row)
_RED_MAIN_CAP = 1024  # max rows of the main reduce tile

_CONFIG = {
    "P_CAP": _P_CAP,
    "TILE": _TILE,
    "COUNT_OCCUPANCY": _COUNT_OCC,
    "PREFETCH_DEPTH": _PREFETCH,
    "ZERO_TILE_CAP": _ZERO_TILE_CAP,
    "RED_BINS": _RED_BINS,
    "RED_MAIN_CAP": _RED_MAIN_CAP,
    "partitioning": "grid-stride over tiles, 3-deep software prefetch",
    "atomic_memory_order": "RELAXED",
    "atomic_memory_scope": "DEVICE",
    "zero_fill": "in-kernel by the owning program (row store before counting)",
    "reduce": "two concurrent power-of-two row tiles (main + remainder), tree-sum",
}


# ---------------------------------------------------------------------------
# Stage 1+2: each program zeroes its own private row, then counts its
# grid-stride share of the input into that row. Three tiles are kept in
# flight ahead of the tile whose atomics are issued. Prefetch indices past
# the end are redirected to the current tile so no load is ever out of range.
# ---------------------------------------------------------------------------
@ct.kernel(occupancy=8)
def _count_kernel(inp, scratch, n, num_tiles,
                  P: ConstInt, NB: ConstInt, TILE: ConstInt, EVEN: ConstBool,
                  ZT: ConstInt, NZC: ConstInt):
    pid = ct.bid(0)
    lane = ct.arange(TILE, dtype=ct.int32)
    t1 = pid + P
    t2 = pid + 2 * P
    i1 = ct.where(t1 < num_tiles, t1, pid)
    i2 = ct.where(t2 < num_tiles, t2, pid)
    v0 = ct.load(inp, (pid,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
    v1 = ct.load(inp, (i1,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
    v2 = ct.load(inp, (i2,), (TILE,), padding_mode=ct.PaddingMode.ZERO)

    # Stage 1 (fused): zero this program's private row before any counting.
    zrow = ct.zeros((1, ZT), ct.int32)
    if NZC == 1:
        ct.store(scratch, (pid, 0), zrow, allow_tma=False)
    else:
        for c in range(NZC):
            ct.store(scratch, (pid, c), zrow, allow_tma=False)

    # Stage 2: private counting into row `pid`.
    for t in range(pid, num_tiles, P):
        t3 = t + 3 * P
        i3 = ct.where(t3 < num_tiles, t3, t)
        v3 = ct.load(inp, (i3,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
        if EVEN:
            valid = (v0 >= 0) & (v0 < NB)
        else:
            offs = lane + t * TILE
            valid = (offs < n) & (v0 >= 0) & (v0 < NB)
        idx = ct.where(valid, v0, 0)
        upd = valid.astype(ct.int32)
        ct.atomic_add(scratch, (pid, idx), upd,
                      check_bounds=False,
                      memory_order=ct.MemoryOrder.RELAXED,
                      memory_scope=ct.MemoryScope.DEVICE)
        v0 = v1
        v1 = v2
        v2 = v3


# ---------------------------------------------------------------------------
# Stage 3: cross-partial reduction (column-wise sum over the P rows).
# Each program owns BINS columns; the P rows are covered by one main tile of
# R1 rows plus (optionally) one remainder tile of R2 rows starting at row R1.
# ---------------------------------------------------------------------------
@ct.kernel
def _reduce_kernel(scratch, out, R1: ConstInt, R2: ConstInt, R2IDX: ConstInt,
                   HAS_REM: ConstBool, BINS: ConstInt):
    b = ct.bid(0)
    if HAS_REM:
        a = ct.load(scratch, (0, b), (R1, BINS),
                    padding_mode=ct.PaddingMode.ZERO)
        c = ct.load(scratch, (R2IDX, b), (R2, BINS),
                    padding_mode=ct.PaddingMode.ZERO)
        s = ct.sum(a, axis=0) + ct.sum(c, axis=0)
    else:
        a = ct.load(scratch, (0, b), (R1, BINS),
                    padding_mode=ct.PaddingMode.ZERO)
        s = ct.sum(a, axis=0)
    ct.store(out, (b,), s.astype(out.dtype))


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    assert num_bins >= 1

    x = input.contiguous()  # no-op for contiguous input
    device = input.device

    num_tiles = ct.cdiv(N, _TILE) if N > 0 else 0
    P = max(1, min(_P_CAP, num_tiles))

    out = torch.empty((num_bins,), dtype=torch.int32, device=device)
    stream = torch.cuda.current_stream()

    if N > 0:
        # Stage 1 is fused into the counting kernel (owning program zeroes
        # its row), so the scratch is allocated uninitialised here.
        scratch = torch.empty((P, num_bins), dtype=torch.int32, device=device)
        nbp = 1 << (num_bins - 1).bit_length()
        zt = min(nbp, _ZERO_TILE_CAP)
        nzc = ct.cdiv(num_bins, zt)
        even = (N % _TILE) == 0
        ct.launch(stream, (P,), _count_kernel,
                  (x, scratch, N, num_tiles, P, num_bins, _TILE, even, zt, nzc))
    else:
        scratch = torch.zeros((P, num_bins), dtype=torch.int32, device=device)

    # Stage 3: row decomposition P = R1 + rem, R1 power of two, rem < R1
    r1 = 1 << (P.bit_length() - 1)
    r1 = min(r1, _RED_MAIN_CAP)
    rem = P - r1
    assert rem < r1 or (rem == 0)
    if rem > 0:
        r2 = 1 << (rem - 1).bit_length()
        if r2 > r1:
            r2 = r1
        has_rem = True
    else:
        r2 = 1
        has_rem = False
    r2idx = r1 // r2
    ct.launch(stream, (ct.cdiv(num_bins, _RED_BINS),), _reduce_kernel,
              (scratch, out, r1, r2, r2idx, has_rem, _RED_BINS))
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
