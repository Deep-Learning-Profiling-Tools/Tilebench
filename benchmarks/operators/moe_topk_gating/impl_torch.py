import torch


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    """
    MoE Top-K gating reference using the same iterative (max, argmax + mask-out)
    algorithm as the Triton and cuTile kernels. This matches their tie-breaking
    behavior (first-occurrence argmax) so that fp16 / bf16 runs verify bit-exact
    against the GPU kernels even when input quantization creates tied values.
    """
    # One forced copy serves both the fp32 upcast (matches the DSL kernels' fp32
    # compare path) and the mutation guard for the scatter_ below — the old
    # .float().clone() double-copied fp16/bf16 inputs.
    logits_f32 = logits.to(torch.float32, copy=True)
    topk_vals = torch.empty(M, k, dtype=torch.float32, device=logits.device)
    topk_idxs = torch.empty(M, k, dtype=torch.int32, device=logits.device)

    for i in range(k):
        vals, idxs = logits_f32.max(dim=-1)          # ties → first occurrence
        topk_vals[:, k - 1 - i] = vals
        topk_idxs[:, k - 1 - i] = idxs.to(torch.int32)
        # Mask out the chosen position so the next iteration picks the next max.
        logits_f32.scatter_(1, idxs.unsqueeze(1), float("-inf"))

    topk_weights = torch.softmax(topk_vals, dim=-1).to(logits.dtype)
    return (topk_weights, topk_idxs)
