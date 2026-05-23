import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _stats_partial(x, psum, psumsq,
                   BLOCK_N: ConstInt, BLOCK_C: ConstInt):
    pid_n = ct.bid(0)
    pid_c = ct.bid(1)
    x_tile = ct.load(x, index=(pid_n, pid_c),
                     shape=(BLOCK_N, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x_tile, np.float32)
    sx = ct.sum(x_f, axis=0, keepdims=True)
    sxx = ct.sum(x_f * x_f, axis=0, keepdims=True)
    ct.store(psum, index=(pid_n, pid_c), tile=sx)
    ct.store(psumsq, index=(pid_n, pid_c), tile=sxx)


@ct.kernel(occupancy=2)
def _finalize(psum, psumsq, gamma, beta, scale_out, bias_out, eps,
              N: ConstInt, NUM_NB: ConstInt,
              BLOCK_C: ConstInt, BLOCK_NB: ConstInt):
    pid = ct.bid(0)
    sx = ct.full((BLOCK_C,), 0.0, dtype=np.float32)
    sxx = ct.full((BLOCK_C,), 0.0, dtype=np.float32)
    num_tiles = ct.cdiv(NUM_NB, BLOCK_NB)
    for j in range(num_tiles):
        ps = ct.load(psum, index=(j, pid),
                     shape=(BLOCK_NB, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
        psq = ct.load(psumsq, index=(j, pid),
                      shape=(BLOCK_NB, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
        sx = sx + ct.sum(ps, axis=0)
        sxx = sxx + ct.sum(psq, axis=0)
    inv_N = 1.0 / N
    m = sx * inv_N
    var = sxx * inv_N - m * m
    r = ct.rsqrt(var + eps)
    g = ct.astype(ct.load(gamma, index=(pid,), shape=(BLOCK_C,)), np.float32)
    b = ct.astype(ct.load(beta, index=(pid,), shape=(BLOCK_C,)), np.float32)
    sc = r * g
    bs = b - m * sc
    ct.store(scale_out, index=(pid,), tile=sc)
    ct.store(bias_out, index=(pid,), tile=bs)


@ct.kernel(occupancy=4)
def _apply(x, out, scale_in, bias_in,
           BLOCK_N: ConstInt, BLOCK_C: ConstInt):
    pid_n = ct.bid(0)
    scale = ct.load(scale_in, index=(0,), shape=(BLOCK_C,), latency=1)
    bias = ct.load(bias_in, index=(0,), shape=(BLOCK_C,), latency=1)
    x_tile = ct.load(x, index=(pid_n, 0),
                     shape=(BLOCK_N, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO,
                     latency=10)
    x_f = ct.astype(x_tile, np.float32)
    y = x_f * scale[None, :] + bias[None, :]
    y_out = ct.astype(y, x.dtype)
    ct.store(out, index=(pid_n, 0), tile=y_out, latency=10)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def run(input, gamma, beta, N, C, eps, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    BLOCK_N_STATS = 256
    BLOCK_C_STATS = 128
    BLOCK_N_APPLY = 16
    BLOCK_C_APPLY = _next_pow2(C)
    BLOCK_C_FIN = 64
    num_nb = _cdiv(N, BLOCK_N_STATS)
    BLOCK_NB_FIN = max(32, _next_pow2(num_nb))
    if BLOCK_NB_FIN > 256:
        BLOCK_NB_FIN = 256

    psum = torch.empty((num_nb, C), device=input.device, dtype=torch.float32)
    psumsq = torch.empty((num_nb, C), device=input.device, dtype=torch.float32)
    scale = torch.empty(C, device=input.device, dtype=torch.float32)
    bias_buf = torch.empty(C, device=input.device, dtype=torch.float32)

    grid_s = (num_nb, _cdiv(C, BLOCK_C_STATS), 1)
    ct.launch(stream, grid_s, _stats_partial,
              (input, psum, psumsq, BLOCK_N_STATS, BLOCK_C_STATS))

    grid_f = (_cdiv(C, BLOCK_C_FIN), 1, 1)
    ct.launch(stream, grid_f, _finalize,
              (psum, psumsq, gamma, beta, scale, bias_buf, float(eps),
               int(N), int(num_nb), BLOCK_C_FIN, BLOCK_NB_FIN))

    grid_a = (_cdiv(N, BLOCK_N_APPLY), 1, 1)
    ct.launch(stream, grid_a, _apply,
              (input, output, scale, bias_buf,
               BLOCK_N_APPLY, BLOCK_C_APPLY))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N_STATS": BLOCK_N_STATS, "BLOCK_C_STATS": BLOCK_C_STATS,
        "BLOCK_N_APPLY": BLOCK_N_APPLY, "BLOCK_C_APPLY": BLOCK_C_APPLY,
        "BLOCK_C_FIN": BLOCK_C_FIN, "BLOCK_NB_FIN": BLOCK_NB_FIN,
        "occupancy_stats": 2, "occupancy_finalize": 2, "occupancy_apply": 4,
        "latency_hints": True,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
