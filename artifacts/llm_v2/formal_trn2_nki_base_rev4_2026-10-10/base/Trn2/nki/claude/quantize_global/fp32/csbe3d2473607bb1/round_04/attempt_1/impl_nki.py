import nki
import nki.language as nl
import nki.isa as nisa
import torch

# fp32 -> fp16 elementwise conversion (single logical traversal).
#
# Two deterministic mappings selected by problem size:
#  * direct  (small n): HBM -> HBM DMAs, the DMA engine converts in flight.
#  * phased  (large n): each program (physical core) converts its contiguous
#    half by DMA-casting fp32 HBM rows into an fp16 SBUF buffer that holds the
#    whole half (read phase: 16 fully-contiguous row-group loads), then writes
#    the fp16 buffer back with column-slab stores (write phase). Every store
#    reads all partitions, so it depends on all loads: HBM sees a pure read
#    stream followed by a pure write stream instead of interleaved traffic.
#    Each element is still read once, converted once and written once.

P = 128
ASSUMED_PROGRAMS = 2      # bare kernel(...) call runs with LNC=2 on this host

# direct path parameters
F_MAX = 4096
F_MIN = 512
MIN_DMAS_PER_PROG = 8

# phased path parameters
PHASE_MIN_N = 6 * 1048576  # use the phased mapping from this size on
PHASE_B = 4096             # innermost contiguous run (elements)
PHASE_MAX_A = 24           # max slabs per program: 24 * 4096 * 2 B = 192 KiB / partition
PHASE_ROWS = 8             # partitions per load DMA (128 / 8 = 16 loads per program)

_LAST_CONFIG = {}


@nki.jit
def _cast_direct_kernel(x, F):
    n = x.shape[0]
    out = nl.ndarray((n,), dtype=nl.float16, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    chunk = P * F
    T = n // (n_prog * chunk)
    main_tiles = n_prog * T
    main_end = main_tiles * chunk

    if T > 0:
        if main_end == n:
            x3 = x.reshape((main_tiles, P, F))
            o3 = out.reshape((main_tiles, P, F))
            for t in range(T):
                g = pid * T + t
                nisa.dma_copy(dst=o3[g], src=x3[g])
        else:
            for t in range(T):
                g = pid * T + t
                off = g * chunk
                nisa.dma_copy(dst=out.ap(pattern=[[F, P], [1, F]], offset=off),
                              src=x.ap(pattern=[[F, P], [1, F]], offset=off))

    if main_end < n:
        pos = main_end
        rem = n - main_end
        while rem >= F:
            rows = min(P, rem // F)
            nisa.dma_copy(dst=out.ap(pattern=[[F, rows], [1, F]], offset=pos),
                          src=x.ap(pattern=[[F, rows], [1, F]], offset=pos))
            pos = pos + rows * F
            rem = rem - rows * F
        if rem > 0:
            nisa.dma_copy(dst=out.ap(pattern=[[rem, 1], [1, rem]], offset=pos),
                          src=x.ap(pattern=[[rem, 1], [1, rem]], offset=pos))

    return out


@nki.jit
def _cast_phase_kernel(x, F):
    n = x.shape[0]
    out = nl.ndarray((n,), dtype=nl.float16, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    R = n_prog * P
    use_phase = 0
    A = 0
    if n % (R * PHASE_B) == 0:
        A = n // (R * PHASE_B)
        if A <= PHASE_MAX_A:
            if A > 0:
                use_phase = 1

    if use_phase == 1:
        x3 = x.reshape((R, A, PHASE_B))
        o3 = out.reshape((R, A, PHASE_B))
        r0 = pid * P
        res = nl.ndarray((P, A, PHASE_B), dtype=nl.float16, buffer=nl.sbuf)
        # read phase: contiguous fp32 row groups -> fp16 SBUF (DMA cast, RNE)
        for i in range(P // PHASE_ROWS):
            g0 = i * PHASE_ROWS
            nisa.dma_copy(dst=res[g0:g0 + PHASE_ROWS, :, :],
                          src=x3[r0 + g0:r0 + g0 + PHASE_ROWS, :, :])
        # write phase: each slab store spans all partitions (depends on all loads)
        for j in range(A):
            nisa.dma_copy(dst=o3[r0:r0 + P, j:j + 1, :], src=res[:, j:j + 1, :])
    else:
        # generic fallback: direct HBM -> HBM DMA cast
        chunk = P * F
        T = n // (n_prog * chunk)
        main_tiles = n_prog * T
        main_end = main_tiles * chunk
        if T > 0:
            for t in range(T):
                g = pid * T + t
                off = g * chunk
                nisa.dma_copy(dst=out.ap(pattern=[[F, P], [1, F]], offset=off),
                              src=x.ap(pattern=[[F, P], [1, F]], offset=off))
        if main_end < n:
            pos = main_end
            rem = n - main_end
            while rem >= F:
                rows = min(P, rem // F)
                nisa.dma_copy(dst=out.ap(pattern=[[F, rows], [1, F]], offset=pos),
                              src=x.ap(pattern=[[F, rows], [1, F]], offset=pos))
                pos = pos + rows * F
                rem = rem - rows * F
            if rem > 0:
                nisa.dma_copy(dst=out.ap(pattern=[[rem, 1], [1, rem]], offset=pos),
                              src=x.ap(pattern=[[rem, 1], [1, rem]], offset=pos))

    return out


def _choose_free(n):
    per_prog = n // ASSUMED_PROGRAMS
    if per_prog >= MIN_DMAS_PER_PROG * P * F_MAX:
        return F_MAX
    f = (per_prog // (MIN_DMAS_PER_PROG * P)) // F_MIN * F_MIN
    if f < F_MIN:
        f = F_MIN
    if f > F_MAX:
        f = F_MAX
    return int(f)


def _phase_slabs(n):
    unit = ASSUMED_PROGRAMS * P * PHASE_B
    if n < PHASE_MIN_N or n % unit != 0:
        return 0
    a = n // unit
    if a > PHASE_MAX_A:
        return 0
    return int(a)


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    global _LAST_CONFIG
    x = x.contiguous()
    n = x.numel()
    F = _choose_free(n)
    A = _phase_slabs(n)
    if A > 0:
        out = _cast_phase_kernel(x, F)
        _LAST_CONFIG = {
            "method": "phased_hbm_sbuf_dma_cast",
            "programs": ASSUMED_PROGRAMS,
            "load_rows": PHASE_ROWS,
            "loads_per_program": P // PHASE_ROWS,
            "slab_free": PHASE_B,
            "stores_per_program": A,
            "sbuf_bytes_per_partition": A * PHASE_B * 2,
            "phase_min_n": PHASE_MIN_N,
        }
    else:
        out = _cast_direct_kernel(x, F)
        chunk = P * F
        blocks_per_prog = n // (ASSUMED_PROGRAMS * chunk)
        _LAST_CONFIG = {
            "method": "hbm_to_hbm_dma_cast",
            "block_rows": P,
            "block_free": F,
            "programs": ASSUMED_PROGRAMS,
            "dmas_per_program": int(blocks_per_prog),
            "tail_elements": int(n - ASSUMED_PROGRAMS * blocks_per_prog * chunk),
            "phase_min_n": PHASE_MIN_N,
        }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
