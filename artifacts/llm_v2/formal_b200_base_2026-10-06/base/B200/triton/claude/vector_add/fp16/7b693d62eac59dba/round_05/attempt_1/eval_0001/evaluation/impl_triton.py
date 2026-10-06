import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# Fixed configuration literals
_BN = 64            # inner (contiguous) tile extent in elements: 128 bytes of fp16
_BM = 64            # rows per tile -> 4096 elements (8 KB) per tensor per tile
_PROGRAMS = 148     # persistent grid: one program per SM
_STAGES = 8         # TMA load pipeline depth (smem ring of 8 stages x 16 KB)
_NUM_WARPS = 4
_TAIL_BLOCK = 64    # masked tail kernel block (only used if n % _BN != 0)

_CONFIG = {
    "mode": "tma_persistent",
    "BM": _BM,
    "BN": _BN,
    "PROGRAMS": _PROGRAMS,
    "num_stages": _STAGES,
    "num_warps": _NUM_WARPS,
    "TAIL_BLOCK": _TAIL_BLOCK,
}


@triton.jit
def _vadd_tma_kernel(x_desc, y_desc, o_desc, num_tiles,
                     BM: tl.constexpr, NPROG: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    # Persistent grid-stride loop; descriptor loads are software-pipelined
    # into a multi-stage shared-memory ring via asynchronous TMA copies.
    for t in tl.range(pid, num_tiles, NPROG, num_stages=STAGES):
        off = t * BM
        a = x_desc.load([off, 0])
        b = y_desc.load([off, 0])
        o_desc.store([off, 0], a + b)


@triton.jit
def _vadd_tail_kernel(x_ptr, y_ptr, o_ptr, start, n, BLOCK: tl.constexpr):
    offs = start + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    xv = tl.load(x_ptr + offs, mask=m, other=0.0)
    yv = tl.load(y_ptr + offs, mask=m, other=0.0)
    tl.store(o_ptr + offs, xv + yv, mask=m)


def run(x, y):
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    rows = n // _BN
    if rows > 0:
        num_tiles = triton.cdiv(rows, _BM)
        nprog = min(_PROGRAMS, num_tiles)
        x_desc = TensorDescriptor(x, [rows, _BN], [_BN, 1], [_BM, _BN])
        y_desc = TensorDescriptor(y, [rows, _BN], [_BN, 1], [_BM, _BN])
        o_desc = TensorDescriptor(out, [rows, _BN], [_BN, 1], [_BM, _BN])
        _vadd_tma_kernel[(nprog,)](
            x_desc, y_desc, o_desc, num_tiles,
            BM=_BM, NPROG=nprog, STAGES=_STAGES,
            num_warps=_NUM_WARPS, num_stages=_STAGES,
        )

    start = rows * _BN
    rem = n - start
    if rem > 0:
        _vadd_tail_kernel[(triton.cdiv(rem, _TAIL_BLOCK),)](
            x, y, out, start, n,
            BLOCK=_TAIL_BLOCK, num_warps=1,
        )
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
