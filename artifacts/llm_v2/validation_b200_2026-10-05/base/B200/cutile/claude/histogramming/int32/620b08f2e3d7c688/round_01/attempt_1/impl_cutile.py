import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

# ---------------------------------------------------------------------------
# Fixed configuration literals
# ---------------------------------------------------------------------------
_P_CAP = 592          # max number of private partial histograms (148 SMs x 4)
_TILE = 4096          # input elements per counting step
_COUNT_OCC = 4        # occupancy hint for the counting kernel
_RED_ROWS = 64        # partial rows summed per reduce step
_RED_BINS = 32        # bins handled per reduce program

_CONFIG = {
    "P_CAP": _P_CAP,
    "TILE": _TILE,
    "COUNT_OCCUPANCY": _COUNT_OCC,
    "RED_ROWS": _RED_ROWS,
    "RED_BINS": _RED_BINS,
    "partitioning": "grid-stride over tiles",
    "atomic_memory_order": "RELAXED",
    "atomic_memory_scope": "DEVICE",
    "zero_fill": "torch.zeros device fill",
}


# ---------------------------------------------------------------------------
# Stage 2: private counting. Program `pid` owns row `pid` of scratch.
# ---------------------------------------------------------------------------
@ct.kernel(occupancy=4)
def _count_kernel(inp, scratch, n, num_tiles,
                  P: ConstInt, NB: ConstInt, TILE: ConstInt, EVEN: ConstBool):
    pid = ct.bid(0)
    lane = ct.arange(TILE, dtype=ct.int32)
    for t in range(pid, num_tiles, P):
        v = ct.load(inp, (t,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
        if EVEN:
            valid = (v >= 0) & (v < NB)
        else:
            offs = lane + t * TILE
            valid = (offs < n) & (v >= 0) & (v < NB)
        # invalid lanes add 0 to bin 0 (never corrupts in-range counts)
        idx = ct.where(valid, v, 0)
        upd = valid.astype(ct.int32)
        ct.atomic_add(scratch, (pid, idx), upd,
                      check_bounds=False,
                      memory_order=ct.MemoryOrder.RELAXED,
                      memory_scope=ct.MemoryScope.DEVICE)


# ---------------------------------------------------------------------------
# Stage 3: cross-partial reduction (column-wise sum over the P rows).
# ---------------------------------------------------------------------------
@ct.kernel
def _reduce_kernel(scratch, out, ITERS: ConstInt, ROWS: ConstInt, BINS: ConstInt):
    b = ct.bid(0)
    acc = ct.zeros((ROWS, BINS), ct.int32)
    for r in range(ITERS):
        t = ct.load(scratch, (r, b), (ROWS, BINS), padding_mode=ct.PaddingMode.ZERO)
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
    ct.launch(stream, (P,), _count_kernel,
              (x, scratch, N, num_tiles, P, num_bins, _TILE, even))

    # Stage 3
    red_iters = ct.cdiv(P, _RED_ROWS)
    ct.launch(stream, (ct.cdiv(num_bins, _RED_BINS),), _reduce_kernel,
              (scratch, out, red_iters, _RED_ROWS, _RED_BINS))
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
