import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


# Single-pass kernel: entire row in one tile. Reads x exactly once.
@ct.kernel(occupancy=1)
def _rmsnorm_single(x, w, out, eps: float, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    xt = ct.load(x, index=(row, 0), shape=(1, TILE),
                 padding_mode=ct.PaddingMode.ZERO)
    xf = ct.astype(xt, np.float32)
    mean_sq = ct.sum(xf * xf, axis=1, keepdims=True) / N
    rstd = ct.rsqrt(mean_sq + eps)
    wt = ct.load(w, index=(0,), shape=(TILE,),
                 padding_mode=ct.PaddingMode.ZERO)
    wf = ct.astype(wt, np.float32)
    yt = xf * rstd * wf[None, :]
    ct.store(out, index=(row, 0), tile=ct.astype(yt, x.dtype))


# Two-pass fallback (used for fp32 / for very large K that exceeds the single-pass cap).
@ct.kernel(occupancy=2)
def _rmsnorm_two_pass(x, w, out, eps: float, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        acc = acc + xj * xj

    mean_sq = ct.sum(acc, axis=1, keepdims=True) / N
    rstd = ct.rsqrt(mean_sq + eps)

    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        wj = ct.astype(
            ct.load(w, index=(j,), shape=(TILE,),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        yj = xj * rstd * wj[None, :]
        ct.store(out, index=(row, j), tile=ct.astype(yj, x.dtype))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    output = torch.empty_like(x)
    batch, M, K = x.shape
    n_rows = batch * M

    x2 = x.reshape(n_rows, K)
    o2 = output.reshape(n_rows, K)

    stream = torch.cuda.current_stream()
    grid = (n_rows, 1, 1)

    pow2_K = _next_pow2(K)

    # Decide single-pass vs two-pass.
    # Single-pass requires the whole padded row in one tile.
    # For fp16/bf16 (2 B/elem) we can afford TILE up to 16384 (32 KB tile of x).
    # For fp32 (4 B/elem) tiles get heavy; the two-pass kernel was already
    # ~94% of roofline, so keep it on the two-pass path.
    if x.dtype != torch.float32 and pow2_K <= 16384:
        TILE = pow2_K if pow2_K >= 512 else 512
        ct.launch(stream, grid, _rmsnorm_single,
                  (x2, rms_w, o2, float(eps), K, TILE))
        mode = "single_pass"
        occupancy = 1
    else:
        TILE = 2048
        ct.launch(stream, grid, _rmsnorm_two_pass,
                  (x2, rms_w, o2, float(eps), K, TILE))
        mode = "two_pass"
        occupancy = 2

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy, "mode": mode})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
