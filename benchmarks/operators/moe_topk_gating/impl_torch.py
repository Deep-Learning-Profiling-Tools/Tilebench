import torch


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    """
    MoE Top-K gating reference using the same iterative (max, argmax + mask-out)
    algorithm as the Triton and cuTile kernels. This matches their tie-breaking
    behavior (first-occurrence argmax) so that fp16 / bf16 runs verify bit-exact
    against the GPU kernels even when input quantization creates tied values.
    """
    logits_f32 = logits.float().clone()
    topk_vals = torch.empty(M, k, dtype=torch.float32, device=logits.device)
    topk_idxs = torch.empty(M, k, dtype=torch.int32, device=logits.device)

    for i in range(k):
        vals, idxs = logits_f32.max(dim=-1)          # ties → first occurrence
        topk_vals[:, i] = vals
        topk_idxs[:, i] = idxs.to(torch.int32)
        # Mask out the chosen position so the next iteration picks the next max.
        logits_f32.scatter_(1, idxs.unsqueeze(1), float("-inf"))

    topk_weights = torch.softmax(topk_vals, dim=-1).to(logits.dtype)
    return (topk_weights, topk_idxs)
