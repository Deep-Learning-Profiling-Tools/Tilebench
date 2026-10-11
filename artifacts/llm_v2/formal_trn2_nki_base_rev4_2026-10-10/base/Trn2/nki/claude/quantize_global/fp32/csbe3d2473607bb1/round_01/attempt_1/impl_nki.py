import nki
import nki.language as nl
import nki.isa as nisa
import torch

P = 128
NBUF = 4
F_MAX = 4096
F_MIN = 512
MIN_TILES_PER_PROG = 8
ASSUMED_PROGRAMS = 2  # bare kernel(...) call runs with LNC=2 on this host

_LAST_CONFIG = {}


@nki.jit
def _cast_f32_to_f16_kernel(x, F):
    n = x.shape[0]
    out = nl.ndarray((n,), dtype=nl.float16, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    chunk = P * F
    T = n // (n_prog * chunk)          # full [P, F] tiles per program
    main_tiles = n_prog * T
    main_end = main_tiles * chunk

    if T > 0:
        # rotating SBUF buffers (bounded memory, allows load/cast/store overlap)
        in0 = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        in1 = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        in2 = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        in3 = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
        ob0 = nl.ndarray((P, F), dtype=nl.float16, buffer=nl.sbuf)
        ob1 = nl.ndarray((P, F), dtype=nl.float16, buffer=nl.sbuf)
        ob2 = nl.ndarray((P, F), dtype=nl.float16, buffer=nl.sbuf)
        ob3 = nl.ndarray((P, F), dtype=nl.float16, buffer=nl.sbuf)

        if main_end == n:
            x3 = x.reshape((main_tiles, P, F))
            o3 = out.reshape((main_tiles, P, F))
            for t in range(T):
                g = pid * T + t
                b = t % 4
                if b == 0:
                    ti = in0
                    to = ob0
                elif b == 1:
                    ti = in1
                    to = ob1
                elif b == 2:
                    ti = in2
                    to = ob2
                else:
                    ti = in3
                    to = ob3
                nisa.dma_copy(dst=ti, src=x3[g])
                if t % 2 == 0:
                    nisa.tensor_copy(dst=to, src=ti, engine=nisa.engine.vector)
                else:
                    nisa.tensor_copy(dst=to, src=ti, engine=nisa.engine.scalar)
                nisa.dma_copy(dst=o3[g], src=to)
        else:
            for t in range(T):
                g = pid * T + t
                b = t % 4
                if b == 0:
                    ti = in0
                    to = ob0
                elif b == 1:
                    ti = in1
                    to = ob1
                elif b == 2:
                    ti = in2
                    to = ob2
                else:
                    ti = in3
                    to = ob3
                off = g * chunk
                nisa.dma_copy(dst=ti, src=x.ap(pattern=[[F, P], [1, F]], offset=off))
                if t % 2 == 0:
                    nisa.tensor_copy(dst=to, src=ti, engine=nisa.engine.vector)
                else:
                    nisa.tensor_copy(dst=to, src=ti, engine=nisa.engine.scalar)
                nisa.dma_copy(dst=out.ap(pattern=[[F, P], [1, F]], offset=off), src=to)

    # Tail (never reached for the configured cases; both programs write identical values)
    if main_end < n:
        pos = main_end
        rem = n - main_end
        while rem >= F:
            rows = min(P, rem // F)
            tti = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
            tto = nl.ndarray((P, F), dtype=nl.float16, buffer=nl.sbuf)
            nisa.dma_copy(dst=tti[0:rows, :],
                          src=x.ap(pattern=[[F, rows], [1, F]], offset=pos))
            nisa.tensor_copy(dst=tto[0:rows, :], src=tti[0:rows, :],
                             engine=nisa.engine.vector)
            nisa.dma_copy(dst=out.ap(pattern=[[F, rows], [1, F]], offset=pos),
                          src=tto[0:rows, :])
            pos = pos + rows * F
            rem = rem - rows * F
        if rem > 0:
            lti = nl.ndarray((1, rem), dtype=nl.float32, buffer=nl.sbuf)
            lto = nl.ndarray((1, rem), dtype=nl.float16, buffer=nl.sbuf)
            nisa.dma_copy(dst=lti, src=x.ap(pattern=[[rem, 1], [1, rem]], offset=pos))
            nisa.tensor_copy(dst=lto, src=lti, engine=nisa.engine.vector)
            nisa.dma_copy(dst=out.ap(pattern=[[rem, 1], [1, rem]], offset=pos), src=lto)

    return out


def _choose_free(n):
    # Deterministic rule: aim for at least MIN_TILES_PER_PROG tiles per program,
    # tile free size a multiple of F_MIN, capped at F_MAX.
    per_prog = n // ASSUMED_PROGRAMS
    if per_prog >= MIN_TILES_PER_PROG * P * F_MAX:
        return F_MAX
    f = (per_prog // (MIN_TILES_PER_PROG * P)) // F_MIN * F_MIN
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
    out = _cast_f32_to_f16_kernel(x, F)
    chunk = P * F
    tiles_per_prog = n // (ASSUMED_PROGRAMS * chunk)
    _LAST_CONFIG = {
        "tile_partitions": P,
        "tile_free": F,
        "num_buffers": NBUF,
        "programs": ASSUMED_PROGRAMS,
        "tiles_per_program": int(tiles_per_prog),
        "tail_elements": int(n - ASSUMED_PROGRAMS * tiles_per_prog * chunk),
        "cast_engines": "vector/scalar alternating",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
