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
_RED_BINS = 8         # bins handled per reduce program (one 32B sector per row)

_CONFIG = {
    "P_CAP": _P_CAP,
    "TILE": _TILE,
    "COUNT_OCCUPANCY": _COUNT_OCC,
    "RED_BINS": _RED_BINS,
    "RED_ROWS": "next_pow2(P), single padded load",
    "partitioning": "grid-stride over tiles, unroll 3, prefetch distance 2 (register LDG)",
    "input_load_tma": False,
    "atomic_memory_order": "RELAXED",
    "atomic_memory_scope": "DEVICE",
    "atomic_masking": "check_bounds (invalid lanes redirected out of range)",
    "zero_fill": "torch.zeros device fill",
}

_ZERO = ct.PaddingMode.ZERO
_RELAXED = ct.MemoryOrder.RELAXED
_DEVICE = ct.MemoryScope.DEVICE


# ---------------------------------------------------------------------------
# Stage 2: private counting. Program `pid` owns row `pid` of scratch.
# Three rotating register buffers (a, b, c): each load is issued two tiles of
# processing ahead of its use, without loop-carried register moves.
# Out-of-range bins / tail lanes are dropped via check_bounds.
# ---------------------------------------------------------------------------
@ct.kernel(occupancy=4)
def _count_kernel(inp, scratch, n, num_tiles,
                  P: ConstInt, NB: ConstInt, TILE: ConstInt, EVEN: ConstBool):
    pid = ct.bid(0)
    last = num_tiles - 1
    lane = ct.arange(TILE, dtype=ct.int32)
    ones = ct.full((TILE,), 1, ct.int32)

    a = ct.load(inp, (pid,), (TILE,), padding_mode=_ZERO, allow_tma=False)
    b = ct.load(inp, (min(pid + P, last),), (TILE,), padding_mode=_ZERO,
                allow_tma=False)

    for t in range(pid, num_tiles, 3 * P):
        c = ct.load(inp, (min(t + 2 * P, last),), (TILE,), padding_mode=_ZERO,
                    allow_tma=False)

        # ---- tile t (always valid inside the loop) ----
        if EVEN:
            ia = a
        else:
            ia = ct.where(lane + t * TILE < n, a, NB)
        ct.atomic_add(scratch, (pid, ia), ones,
                      check_bounds=True,
                      memory_order=_RELAXED,
                      memory_scope=_DEVICE)

        a = ct.load(inp, (min(t + 3 * P, last),), (TILE,), padding_mode=_ZERO,
                    allow_tma=False)

        # ---- tile t + P ----
        t1 = t + P
        if t1 < num_tiles:
            if EVEN:
                ib = b
            else:
                ib = ct.where(lane + t1 * TILE < n, b, NB)
            ct.atomic_add(scratch, (pid, ib), ones,
                          check_bounds=True,
                          memory_order=_RELAXED,
                          memory_scope=_DEVICE)

        b = ct.load(inp, (min(t + 4 * P, last),), (TILE,), padding_mode=_ZERO,
                    allow_tma=False)

        # ---- tile t + 2P ----
        t2 = t + 2 * P
        if t2 < num_tiles:
            if EVEN:
                ic = c
            else:
                ic = ct.where(lane + t2 * TILE < n, c, NB)
            ct.atomic_add(scratch, (pid, ic), ones,
                          check_bounds=True,
                          memory_order=_RELAXED,
                          memory_scope=_DEVICE)


# ---------------------------------------------------------------------------
# Stage 3: cross-partial reduction (column-wise sum over the P rows).
# One padded (ROWS, BINS) load per program, tree-sum over rows.
# ---------------------------------------------------------------------------
@ct.kernel
def _reduce_kernel(scratch, out, ROWS: ConstInt, BINS: ConstInt):
    b = ct.bid(0)
    t = ct.load(scratch, (0, b), (ROWS, BINS), padding_mode=ct.PaddingMode.ZERO)
    s = ct.sum(t, axis=0)
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
                  (x, scratch, N, num_tiles, P, num_bins, _TILE, even))

    # Stage 3
    rows = 1 << (P - 1).bit_length()
    ct.launch(stream, (ct.cdiv(num_bins, _RED_BINS),), _reduce_kernel,
              (scratch, out, rows, _RED_BINS))
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
