import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def cross_entropy_kernel(logits, targets, output, BLOCK_CLASSES: ConstInt):
    bid = ct.bid(0)
    logits_tile = ct.load(logits, index=(bid, 0), shape=(1, BLOCK_CLASSES))

    row_max = ct.max(logits_tile, axis=1)
    shifted = logits_tile - ct.reshape(row_max, (1, 1))
    row_sum = ct.sum(ct.exp(shifted), axis=1)

    target_cls = ct.load(targets, index=(bid,), shape=())
    target_logit = ct.gather(logits, (bid, target_cls), check_bounds=False)

    loss = -(target_logit - row_max - ct.log(row_sum))
    ct.store(output, index=(bid,), tile=loss)


def run(logits: torch.Tensor, targets: torch.Tensor, block_size: int = 1024):
    output = torch.empty((logits.shape[0],), device=logits.device, dtype=logits.dtype)
    grid = (logits.shape[0], 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        cross_entropy_kernel,
        (logits, targets, output, block_size),
    )
    return output
