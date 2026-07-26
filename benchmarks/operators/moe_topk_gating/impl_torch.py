import torch


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):


    logits_f32 = logits.to(torch.float32, copy=True)
    topk_vals = torch.empty(M, k, dtype=torch.float32, device=logits.device)
    topk_idxs = torch.empty(M, k, dtype=torch.int32, device=logits.device)

    for i in range(k):
        vals, idxs = logits_f32.max(dim=-1)
        topk_vals[:, k - 1 - i] = vals
        topk_idxs[:, k - 1 - i] = idxs.to(torch.int32)

        logits_f32.scatter_(1, idxs.unsqueeze(1), float("-inf"))

    topk_weights = torch.softmax(topk_vals, dim=-1).to(logits.dtype)
    return (topk_weights, topk_idxs)
