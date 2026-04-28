"""cuTile KL divergence forward (mirrors impl_triton.py).

Convention (PyTorch F.kl_div with log_target=False):
    log_y_pred  is log-probabilities (log_softmax output)
    y_true      is plain probabilities (softmax output)
    loss[b] = sum_s y_true[b,s] * (log(y_true[b,s]) - log_y_pred[b,s])

Each CTA handles one row, with an inner tile loop iterating TILE-sized
chunks across the cols axis. TILE is a real autotune knob (decoupled
from cols), so cols=16384 doesn't blow up SMEM/registers.
"""
from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

# Cartesian product, mirrors impl_triton.py.
# Triton sweeps (BLOCK_SIZE, num_warps); cuTile sweeps (tile, occupancy)
# with nw * occ ~= 64 (Triton's nw in [2, 4, 8] pairs with occ in [32, 16, 8]).
_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [512, 1024, 2048, 4096]
    for occ in [4, 8, 16, 32]
    if t * occ >= 2048
]


@ct.kernel
def _kl_divergence_kernel(log_y_pred, y_true, loss, n_cols, TILE: ConstInt):
    bid = ct.bid(0)

    acc = ct.full((1, TILE), 0.0, dtype=ct.float32)
    n_tiles = ct.cdiv(n_cols, TILE)

    t = 0
    while t < n_tiles:
        log_pred_tile = ct.load(
            log_y_pred, index=(bid, t), shape=(1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
        )
        y_true_tile = ct.load(
            y_true, index=(bid, t), shape=(1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
        )

        log_pred_f32 = ct.astype(log_pred_tile, ct.float32)
        y_true_f32 = ct.astype(y_true_tile, ct.float32)

        # OOB lanes load as 0 (padding_mode=ZERO). For y_true==0 we want
        # loss=0 by KL convention -- guard log explicitly so log(0)*0
        # doesn't go through NaN.
        safe_log = ct.where(y_true_f32 > 0.0, ct.log(y_true_f32), 0.0)
        acc = acc + y_true_f32 * (safe_log - log_pred_f32)
        t = t + 1

    row_sum = ct.sum(acc, axis=1)  # (1,)
    ct.store(loss, index=(bid,), tile=row_sum)


_tuner = CutileAutotuner(_kl_divergence_kernel)


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    global _last_autotune_config

    rows, cols = log_y_pred.shape
    loss = torch.empty(rows, device=log_y_pred.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(rows, cols),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (rows, 1, 1),
            args_fn=lambda cfg: (log_y_pred, y_true, loss, cols, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config = {"tile": cfg.tile, "occupancy": cfg.occupancy}
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, (rows, 1, 1), kernel, (log_y_pred, y_true, loss, cols, cfg.tile))
    return loss


def get_last_config() -> dict | None:
    return _last_autotune_config
