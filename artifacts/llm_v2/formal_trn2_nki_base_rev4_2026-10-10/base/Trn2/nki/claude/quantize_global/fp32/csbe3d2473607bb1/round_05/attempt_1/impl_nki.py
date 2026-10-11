import nki
import nki.language as nl
import nki.isa as nisa
import torch

# Flat streaming fp32 -> fp16 conversion (single logical traversal).
# Each program (physical core under LNC=2) converts its contiguous half of the
# input with direct HBM -> HBM DMAs; the DMA engine performs the native
# fp32 -> fp16 (round-to-nearest-even) conversion in flight. No SBUF staging,
# no extra DMA traffic.
#
# Mapping: every program issues a fixed number (16) of equally sized,
# fully contiguous DMA blocks of shape [R, F], so that the DMA work is evenly
# balanced over the DMA engines of each physical core regardless of n.
# R <= 128 rows; F (contiguous elements per row) is kept >= 512 for
# reasonably large descriptors.

P = 128                 # max rows per DMA block
ASSUMED_PROGRAMS = 2    # bare kernel(...) call runs with LNC=2 on this host
DMAS_PER_PROG = 16      # one block per DMA engine of a physical core
F_MIN = 512             # min contiguous elements per row (2 KB fp32 reads)

_LAST_CONFIG = {}


@nki.jit
def _cast_f32_to_f16_dma_kernel(x, R, F):
    n = x.shape[0]
    out = nl.ndarray((n,), dtype=nl.float16, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    chunk = R * F
    T = n // (n_prog * chunk)          # full [R, F] blocks per program
    main_tiles = n_prog * T
    main_end = main_tiles * chunk

    if T > 0:
        if main_end == n:
            x3 = x.reshape((main_tiles, R, F))
            o3 = out.reshape((main_tiles, R, F))
            for t in range(T):
                g = pid * T + t
                nisa.dma_copy(dst=o3[g], src=x3[g])
        else:
            for t in range(T):
                g = pid * T + t
                off = g * chunk
                nisa.dma_copy(dst=out.ap(pattern=[[F, R], [1, F]], offset=off),
                              src=x.ap(pattern=[[F, R], [1, F]], offset=off))

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


def _plan(n):
    # Deterministic rule: split each program's half into DMAS_PER_PROG equal
    # contiguous blocks [R, F] with R <= 128 and F >= F_MIN where possible.
    per_prog = n // ASSUMED_PROGRAMS
    chunk = per_prog // DMAS_PER_PROG
    if chunk >= P * F_MIN:
        return P, int(chunk // P)
    if chunk >= F_MIN:
        return int(chunk // F_MIN), F_MIN
    return 1, int(max(chunk, 1))


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    global _LAST_CONFIG
    x = x.contiguous()
    n = x.numel()
    R, F = _plan(n)
    out = _cast_f32_to_f16_dma_kernel(x, R, F)
    chunk = R * F
    blocks_per_prog = n // (ASSUMED_PROGRAMS * chunk)
    _LAST_CONFIG = {
        "method": "hbm_to_hbm_dma_cast_balanced",
        "block_rows": int(R),
        "block_free": int(F),
        "programs": ASSUMED_PROGRAMS,
        "dmas_per_program": int(blocks_per_prog),
        "tail_elements": int(n - ASSUMED_PROGRAMS * blocks_per_prog * chunk),
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
