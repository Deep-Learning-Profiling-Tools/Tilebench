from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(occupancy=8)

_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [4, 8, 16, 32]]


@ct.kernel
def cross_entropy_kernel(logits, targets, output, num_classes, BLOCK_CLASSES: ConstInt):
    bid = ct.bid(0)


    logits_tile = ct.load(
        logits, index=(bid, 0), shape=(1, BLOCK_CLASSES),
        padding_mode=ct.PaddingMode.NEG_INF,
    )
    logits_f32 = ct.astype(logits_tile, ct.float32)

    row_max = ct.max(logits_f32, axis=1)
    shifted = logits_f32 - ct.reshape(row_max, (1, 1))
    row_sum = ct.sum(ct.exp(shifted), axis=1)

    target_cls = ct.load(targets, index=(bid,), shape=())


    target_ok = (target_cls >= 0) & (target_cls < num_classes)
    safe_target = ct.where(target_ok, target_cls, 0)
    target_logit_raw = ct.gather(logits, (bid, safe_target), check_bounds=False)
    target_logit = ct.astype(target_logit_raw, ct.float32)
    target_logit = ct.where(target_ok, target_logit, -float("inf"))

    loss = -(target_logit - row_max - ct.log(row_sum))
    ct.store(output, index=(bid,), tile=ct.astype(loss, output.dtype))


_tuner = CutileAutotuner(cross_entropy_kernel)


def run(
    logits: torch.Tensor,
    targets: torch.Tensor,
    block_size: int = 1024,
    autotune: bool = False,
) -> torch.Tensor:
    batch_size, num_classes = logits.shape


    block_classes = 1
    while block_classes < num_classes:
        block_classes *= 2

    logits = logits.contiguous()
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)
    stream = torch.cuda.current_stream()
    grid = (batch_size, 1, 1)

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(batch_size, num_classes, str(logits.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: grid,
            args_fn=lambda cfg: (logits, targets, output, num_classes, block_classes),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({"occupancy": cfg.occupancy})
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream, grid, kernel,
        (logits, targets, output, num_classes, block_classes),
    )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
