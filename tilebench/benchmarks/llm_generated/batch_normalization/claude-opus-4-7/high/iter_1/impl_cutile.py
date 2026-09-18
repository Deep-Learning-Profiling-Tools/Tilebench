import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _stats_kernel(x, mean_out, rstd_out, eps,
                  N: ConstInt, BLOCK_N: ConstInt, BLOCK_C: ConstInt):
    pid_c = ct.bid(0)

    sum_x = ct.full((BLOCK_C,), 0.0, dtype=np.float32)
    sum_xx = ct.full((BLOCK_C,), 0.0, dtype=np.float32)

    num_n_tiles = ct.cdiv(N, BLOCK_N)
    for n_tile in range(0, num_n_tiles):
        x_tile = ct.load(x, index=(n_tile, pid_c),
                         shape=(BLOCK_N, BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)
        x_f = ct.astype(x_tile, np.float32)
        sum_x = sum_x + ct.sum(x_f, axis=0)
        sum_xx = sum_xx + ct.sum(x_f * x_f, axis=0)

    inv_N = 1.0 / N
    mean = sum_x * inv_N
    var = sum_xx * inv_N - mean * mean
    rstd = ct.rsqrt(var + eps)

    ct.store(mean_out, index=(pid_c,), tile=mean)
    ct.store(rstd_out, index=(pid_c,), tile=rstd)


@ct.kernel
def _apply_kernel(x, out, mean_in, rstd_in, gamma, beta,
                  BLOCK_N: ConstInt, BLOCK_C: ConstInt):
    pid_n = ct.bid(0)
    pid_c = ct.bid(1)

    mean = ct.load(mean_in, index=(pid_c,), shape=(BLOCK_C,))
    rstd = ct.load(rstd_in, index=(pid_c,), shape=(BLOCK_C,))
    gamma_t = ct.astype(ct.load(gamma, index=(pid_c,), shape=(BLOCK_C,)), np.float32)
    beta_t = ct.astype(ct.load(beta, index=(pid_c,), shape=(BLOCK_C,)), np.float32)

    x_tile = ct.load(x, index=(pid_n, pid_c),
                     shape=(BLOCK_N, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x_tile, np.float32)

    y = (x_f - mean[None, :]) * rstd[None, :] * gamma_t[None, :] + beta_t[None, :]
    y_out = ct.astype(y, x.dtype)
    ct.store(out, index=(pid_n, pid_c), tile=y_out)


def run(input, gamma, beta, N, C, eps, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    BLOCK_N_STATS = 256
    BLOCK_C_STATS = 64
    BLOCK_N_APPLY = 64
    BLOCK_C_APPLY = 128
    occupancy_stats = 2
    occupancy_apply = 4

    mean = torch.empty(C, device=input.device, dtype=torch.float32)
    rstd = torch.empty(C, device=input.device, dtype=torch.float32)

    grid_s = (ct.cdiv(C, BLOCK_C_STATS), 1, 1)
    stats_kernel = _stats_kernel.with_hints(occupancy=occupancy_stats)
    ct.launch(stream, grid_s, stats_kernel,
              (input, mean, rstd, float(eps), int(N),
               BLOCK_N_STATS, BLOCK_C_STATS))

    grid_a = (ct.cdiv(N, BLOCK_N_APPLY), ct.cdiv(C, BLOCK_C_APPLY), 1)
    apply_kernel = _apply_kernel.with_hints(occupancy=occupancy_apply)
    ct.launch(stream, grid_a, apply_kernel,
              (input, output, mean, rstd, gamma, beta,
               BLOCK_N_APPLY, BLOCK_C_APPLY))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N_STATS": BLOCK_N_STATS, "BLOCK_C_STATS": BLOCK_C_STATS,
        "BLOCK_N_APPLY": BLOCK_N_APPLY, "BLOCK_C_APPLY": BLOCK_C_APPLY,
        "occupancy_stats": occupancy_stats, "occupancy_apply": occupancy_apply,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
