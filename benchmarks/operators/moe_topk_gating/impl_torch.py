import torch


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):


    topk_vals, topk_idxs = torch.topk(
        logits,
        k=k,
        dim=-1,
        largest=True,
        sorted=True,
    )

    topk_weights = torch.softmax(
        topk_vals.to(torch.float32),
        dim=-1,
    ).to(logits.dtype)

    return (topk_weights, topk_idxs.to(torch.int32))
