import nki
import nki.language as nl
import nki.isa as nisa
import torch

# Flat streaming fp32 -> fp16 conversion.
# Each program (physical core under LNC=2) converts its contiguous half of the
# input with direct HBM -> HBM DMAs; the DMA engine performs the native
# fp32 -> fp16 (round-to-nearest-even) conversion in flight, so there is a
# single logical traversal and no SBUF staging / engine synchronisation.

P = 128            # rows per DMA block
F_MAX = 4096       # max contiguous elements per row of a DMA block
F_MIN = 512
MIN_DMAS_PER_PROG = 8
ASSUMED_PROGRAMS = 2  # bare kernel(...) call runs with LNC=2 on this host

_LAST_CONFIG = {}


@nki.jit
def _cast_f32_to_f16_dma_kernel(x, F):
    n = x.shape[0]
    out = nl.ndarray((n,), dtype=nl.float16, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    chunk = P * F
    T = n // (n_prog * chunk)          # full [P, F] blocks per program
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

    # Tail (not reached for the configured cases; both programs write identical values)
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
    # Deterministic rule: aim for at least MIN_DMAS_PER_PROG blocks per program,
    # row length a multiple of F_MIN, capped at F_MAX.
    per_prog = n // ASSUMED_PROGRAMS
    if per_prog >= MIN_DMAS_PER_PROG * P * F_MAX:
        return F_MAX
    f = (per_prog // (MIN_DMAS_PER_PROG * P)) // F_MIN * F_MIN
    if f < F_MIN:
        f = F_MIN
    if f > F_MAX:
        f = F_MAX
    return int(f)


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    global _LAST_CONFIG
    x = x.contiguous()
    n = x.numel()
    F = _choose_free(n)
    out = _cast_f32_to_f16_dma_kernel(x, F)
    chunk = P * F
    blocks_per_prog = n // (ASSUMED_PROGRAMS * chunk)
    _LAST_CONFIG = {
        "method": "hbm_to_hbm_dma_cast",
        "block_rows": P,
        "block_free": F,
        "programs": ASSUMED_PROGRAMS,
        "dmas_per_program": int(blocks_per_prog),
        "tail_elements": int(n - ASSUMED_PROGRAMS * blocks_per_prog * chunk),
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
