import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

# Process 2 rows per CTA to better amortize the per-tile rms_w load
# (rms_w is loaded once per CTA but reused across all rows in that CTA)
# and to give the cuTile compiler more independent loads to pipeline.
@ct.kernel(occupancy=4)
def _rmsnorm_kernel_2r(x, w, out, eps: float, N: ConstInt, TILE: ConstInt):
    rblk = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    # Pass 1: sum of squares in fp32 over 2 rows simultaneously
    acc = ct.full((2, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(rblk, j), shape=(2, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1),
            np.float32,
        )
        acc = acc + xj * xj

    mean_sq = ct.sum(acc, axis=1, keepdims=True) / N    # (2, 1)
    rstd = ct.rsqrt(mean_sq + eps)                      # (2, 1)

    # Pass 2: normalize and scale; rms_w is reused for both rows.
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(rblk, j), shape=(2, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1),
            np.float32,
        )
        wj = ct.astype(
            ct.load(w, index=(j,), shape=(TILE,),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1),
            np.float32,
        )
        yj = xj * rstd * wj[None, :]
        ct.store(out, index=(rblk, j), tile=ct.astype(yj, x.dtype))


# Fallback 1-row kernel (iter-4 proven config) for odd n_rows
@ct.kernel(occupancy=4)
def _rmsnorm_kernel_1r(x, w, out, eps: float, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1),
            np.float32,
        )
        acc = acc + xj * xj

    mean_sq = ct.sum(acc, axis=1, keepdims=True) / N
    rstd = ct.rsqrt(mean_sq + eps)

    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1),
            np.float32,
        )
        wj = ct.astype(
            ct.load(w, index=(j,), shape=(TILE,),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=1),
            np.float32,
        )
        yj = xj * rstd * wj[None, :]
        ct.store(out, index=(row, j), tile=ct.astype(yj, x.dtype))


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    output = torch.empty_like(x)
    batch, M, K = x.shape
    n_rows = batch * M

    x2 = x.reshape(n_rows, K)
    o2 = output.reshape(n_rows, K)

    TILE = 2048
    occupancy = 4
    rows_per_cta = 2 if (n_rows % 2 == 0) else 1

    stream = torch.cuda.current_stream()
    if rows_per_cta == 2:
        grid = (n_rows // 2, 1, 1)
        ct.launch(stream, grid, _rmsnorm_kernel_2r,
                  (x2, rms_w, o2, float(eps), K, TILE))
    else:
        grid = (n_rows, 1, 1)
        ct.launch(stream, grid, _rmsnorm_kernel_1r,
                  (x2, rms_w, o2, float(eps), K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
        "rows_per_cta": rows_per_cta,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
