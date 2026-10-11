import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Partition count of an on-chip tile (axis 0).
P = 128

_LAST_CONFIG = {}


def _choose_free(n):
    # Deterministic function of the shape only: the per-partition chunk size
    # (fp16 elements) of one on-chip tile. Assumes the default LNC=2 split
    # (two programs). Prefer large DMA rows but at least ~8 tiles per program
    # so loads, compute and stores pipeline well.
    cols = n // (2 * P)
    for f in (4096, 2048):
        if cols % f == 0 and cols // f >= 8:
            return f
    return 1024


def _process_block(xv, ov, pid, cols, free_tile, dtype):
    # xv / ov: HBM views of shape (nprog, P, cols); this program handles xv[pid].
    nchunks = (cols + free_tile - 1) // free_tile
    for c in range(nchunks):
        c0 = c * free_tile
        sz = min(free_tile, cols - c0)
        t = nl.ndarray((P, sz), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=xv[pid, :, c0:c0 + sz])
        if c % 2 == 0:
            # VectorE: max(x, 0)
            nisa.tensor_scalar(dst=t, data=t, op0=nl.maximum, operand0=0.0,
                               engine=nisa.engine.vector)
        else:
            # ScalarE: relu(x)
            nisa.activation(dst=t, op=nl.relu, data=t)
        nisa.dma_copy(dst=ov[pid, :, c0:c0 + sz], src=t)


@nki.jit
def _relu_kernel(x, free_tile):
    n = x.shape[0]
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nprog = nl.num_programs()
    pid = nl.program_id(0)
    cols = n // (nprog * P)
    main = cols * nprog * P
    if cols > 0:
        if main == n:
            _process_block(x.reshape((nprog, P, cols)),
                           out.reshape((nprog, P, cols)),
                           pid, cols, free_tile, x.dtype)
        else:
            _process_block(x[0:main].reshape((nprog, P, cols)),
                           out[0:main].reshape((nprog, P, cols)),
                           pid, cols, free_tile, x.dtype)
    rem = n - main
    if rem > 0:
        # Tail (< nprog*128 elements); never reached for the configured cases.
        tt = nl.ndarray((1, rem), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tt, src=x[main:n])
        nisa.tensor_scalar(dst=tt, data=tt, op0=nl.maximum, operand0=0.0,
                           engine=nisa.engine.vector)
        nisa.dma_copy(dst=out[main:n], src=tt)
    return out


def run(x: torch.Tensor):
    global _LAST_CONFIG
    n = x.numel()
    free_tile = _choose_free(n)
    out = _relu_kernel(x, free_tile)
    _LAST_CONFIG = {
        "programs": 2,
        "partitions": P,
        "tile_free_elems": free_tile,
        "tile_bytes": P * free_tile * 2,
        "layout": "(programs, 128, n/(programs*128)) contiguous rows per partition",
        "compute": "alternating VectorE max(x,0) / ScalarE relu, in-place SBUF",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
