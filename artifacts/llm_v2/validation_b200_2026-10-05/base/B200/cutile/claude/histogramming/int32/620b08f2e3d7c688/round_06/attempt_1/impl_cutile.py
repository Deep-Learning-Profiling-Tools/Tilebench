import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

# ---------------------------------------------------------------------------
# Fixed configuration literals
# ---------------------------------------------------------------------------
_P_CAP = 592          # max number of private partial histograms (148 SMs x 4)
_TILE = 2048          # input elements per counting step
_COUNT_OCC = 4        # occupancy hint for the counting kernel
_PREFETCH = 2         # tiles in flight ahead of the one being counted
_RED_BINS = 8         # bins handled per reduce program (one 32B sector per row)
_RED_RCH = 1024       # rows per reduce load (single shot for P <= 1024)

_CONFIG = {
    "P_CAP": _P_CAP,
    "TILE": _TILE,
    "COUNT_OCCUPANCY": _COUNT_OCC,
    "PREFETCH_DEPTH": _PREFETCH,
    "RED_BINS": _RED_BINS,
    "RED_RCH": _RED_RCH,
    "partitioning": "grid-stride over tiles, 2-deep software prefetch (clamped)",
    "atomic_memory_order": "RELAXED",
    "atomic_memory_scope": "DEVICE",
    "atomic_bounds": "check_bounds=True (out-of-range values dropped)",
    "zero_fill": "torch.zeros device fill",
    "reduce": "single (RCH x BINS) padded load per column slice, tree-sum",
}


# ---------------------------------------------------------------------------
# Stage 2: private counting. Program `pid` owns row `pid` of scratch.
# Two tiles are kept in flight ahead of the tile whose atomics are issued.
# Prefetch indices are clamped to the last tile so no load is fully OOB.
# ---------------------------------------------------------------------------
@ct.kernel(occupancy=4)
def _count_kernel(inp, scratch, n, num_tiles,
                  P: ConstInt, TILE: ConstInt, EVEN: ConstBool):
    pid = ct.bid(0)
    last = num_tiles - 1
    cnt = (num_tiles - pid + (P - 1)) // P
    v0 = ct.load(inp, (pid,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
    v1 = ct.load(inp, (min(pid + P, last),), (TILE,),
                 padding_mode=ct.PaddingMode.ZERO)
    t = pid
    for k in range(cnt):
        v2 = ct.load(inp, (min(t + 2 * P, last),), (TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
        if EVEN:
            idx = v0
        else:
            offs = ct.arange(TILE, dtype=ct.int32) + t * TILE
            idx = ct.where(offs < n, v0, -1)
        # out-of-range bins (negative or >= num_bins) are skipped by the bounds check
        ct.atomic_add(scratch, (pid, idx), 1,
                      check_bounds=True,
                      memory_order=ct.MemoryOrder.RELAXED,
                      memory_scope=ct.MemoryScope.DEVICE)
        v0 = v1
        v1 = v2
        t = t + P


# ---------------------------------------------------------------------------
# Stage 3: cross-partial reduction (column-wise sum over the P rows).
# Each program owns BINS columns; rows are loaded in RCH-row padded chunks.
# ---------------------------------------------------------------------------
@ct.kernel
def _reduce_kernel(scratch, out, NCH: ConstInt, RCH: ConstInt, BINS: ConstInt):
    b = ct.bid(0)
    acc = ct.load(scratch, (0, b), (RCH, BINS),
                  padding_mode=ct.PaddingMode.ZERO)
    for k in range(1, NCH):
        t = ct.load(scratch, (k, b), (RCH, BINS),
                    padding_mode=ct.PaddingMode.ZERO)
        acc = acc + t
    s = ct.sum(acc, axis=0)
    ct.store(out, (b,), s.astype(out.dtype))


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    assert num_bins >= 1

    x = input.contiguous()  # no-op for contiguous input
    device = input.device

    num_tiles = ct.cdiv(N, _TILE)
    P = max(1, min(_P_CAP, num_tiles))

    # Stage 1: zero-initialised private partial histograms (fresh every call)
    scratch = torch.zeros((P, num_bins), dtype=torch.int32, device=device)
    out = torch.empty((num_bins,), dtype=torch.int32, device=device)

    stream = torch.cuda.current_stream()
    even = (N % _TILE) == 0

    # Stage 2
    if N > 0:
        ct.launch(stream, (P,), _count_kernel,
                  (x, scratch, N, num_tiles, P, _TILE, even))

    # Stage 3
    nch = ct.cdiv(P, _RED_RCH)
    ct.launch(stream, (ct.cdiv(num_bins, _RED_BINS),), _reduce_kernel,
              (scratch, out, nch, _RED_RCH, _RED_BINS))
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
