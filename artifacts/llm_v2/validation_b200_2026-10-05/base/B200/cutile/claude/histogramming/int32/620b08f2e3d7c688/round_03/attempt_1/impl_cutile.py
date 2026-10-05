import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

# ---------------------------------------------------------------------------
# Fixed configuration literals
# ---------------------------------------------------------------------------
_P_CAP = 592          # max number of private partial histograms (148 SMs x 4)
_TILE = 4096          # input elements per counting step (16 KB per load -> more bytes in flight)
_COUNT_OCC = 4        # occupancy hint for the counting kernel
_RED_BINS = 8         # bins handled per reduce program (one 32B sector per row)

_CONFIG = {
    "P_CAP": _P_CAP,
    "TILE": _TILE,
    "COUNT_OCCUPANCY": _COUNT_OCC,
    "RED_BINS": _RED_BINS,
    "RED_ROWS": "next_pow2(P), single padded load",
    "partitioning": "grid-stride over tiles, 1-deep software prefetch",
    "atomic_memory_order": "RELAXED",
    "atomic_memory_scope": "BLOCK",
    "zero_fill": "torch.zeros device fill",
}


# ---------------------------------------------------------------------------
# Stage 2: private counting. Program `pid` owns row `pid` of scratch.
# The next tile is loaded before the atomics of the current tile are issued.
# Only this program ever touches row `pid`, so block-scope atomics suffice;
# visibility to the reduce kernel is guaranteed by the kernel boundary.
# ---------------------------------------------------------------------------
@ct.kernel(occupancy=4)
def _count_kernel(inp, scratch, n, num_tiles,
                  P: ConstInt, NB: ConstInt, TILE: ConstInt, EVEN: ConstBool):
    pid = ct.bid(0)
    lane = ct.arange(TILE, dtype=ct.int32)
    v = ct.load(inp, (pid,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
    t_cur = pid
    for t in range(pid + P, num_tiles, P):
        vn = ct.load(inp, (t,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
        if EVEN:
            valid = (v >= 0) & (v < NB)
        else:
            offs = lane + t_cur * TILE
            valid = (offs < n) & (v >= 0) & (v < NB)
        idx = ct.where(valid, v, 0)
        upd = valid.astype(ct.int32)
        ct.atomic_add(scratch, (pid, idx), upd,
                      check_bounds=False,
                      memory_order=ct.MemoryOrder.RELAXED,
                      memory_scope=ct.MemoryScope.BLOCK)
        v = vn
        t_cur = t
    # last tile of this program
    if EVEN:
        valid = (v >= 0) & (v < NB)
    else:
        offs = lane + t_cur * TILE
        valid = (offs < n) & (v >= 0) & (v < NB)
    idx = ct.where(valid, v, 0)
    upd = valid.astype(ct.int32)
    ct.atomic_add(scratch, (pid, idx), upd,
                  check_bounds=False,
                  memory_order=ct.MemoryOrder.RELAXED,
                  memory_scope=ct.MemoryScope.BLOCK)


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
    ct.launch(stream, (P,), _count_kernel,
              (x, scratch, N, num_tiles, P, num_bins, _TILE, even))

    # Stage 3
    rows = 1 << (P - 1).bit_length()
    ct.launch(stream, (ct.cdiv(num_bins, _RED_BINS),), _reduce_kernel,
              (scratch, out, rows, _RED_BINS))
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
