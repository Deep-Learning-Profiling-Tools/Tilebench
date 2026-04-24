from types import SimpleNamespace

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(occupancy=8)

_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [4, 8, 16, 32]]


@ct.kernel
def _cross_entropy_kernel(logits, targets, output, BLOCK_CLASSES: ConstInt):
    bid = ct.bid(0)
    logits_tile = ct.load(logits, index=(bid, 0), shape=(1, BLOCK_CLASSES))

    row_max = ct.max(logits_tile, axis=1)
    shifted = logits_tile - ct.reshape(row_max, (1, 1))
    row_sum = ct.sum(ct.exp(shifted), axis=1)

    target_cls = ct.load(targets, index=(bid,), shape=())
    target_logit = ct.gather(logits, (bid, target_cls), check_bounds=False)

    loss = -(target_logit - row_max - ct.log(row_sum))
    ct.store(output, index=(bid,), tile=loss)


def run(
    logits: torch.Tensor,
    targets: torch.Tensor,
    block_size: int = 1024,
    autotune: bool = False,
) -> torch.Tensor:
    global _last_autotune_config
    batch_size, num_classes = logits.shape

    # BLOCK_CLASSES must be a power of 2 >= num_classes.
    block_classes = 1
    while block_classes < num_classes:
        block_classes *= 2

    # Pad logits with -inf so out-of-bounds positions don't affect max/sum.
    if block_classes != num_classes:
        logits_padded = torch.full(
            (batch_size, block_classes), float("-inf"),
            dtype=logits.dtype, device=logits.device,
        )
        logits_padded[:, :num_classes] = logits
    else:
        logits_padded = logits.contiguous()

    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)
    stream = torch.cuda.current_stream()
    grid = (batch_size, 1, 1)

    if autotune:
        result = ct.tune.exhaustive_search(
            _SEARCH_SPACE,
            stream,
            grid_fn=lambda cfg: grid,
            kernel=_cross_entropy_kernel,
            args_fn=lambda cfg: (logits_padded, targets, output, block_classes),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config = {"occupancy": result.best.config.occupancy}

    ct.launch(
        stream, grid, _cross_entropy_kernel,
        (logits_padded, targets, output, block_classes),
    )

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
