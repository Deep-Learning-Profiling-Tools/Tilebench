import torch


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    # repeat_interleave (broadcast+reshape) instead of S[row_idx[:, None], col_idx[None, :]]:
    # that advanced-indexing gather compiles fine under torch_xla but faults at
    # runtime on Neuron with "scatter/gather (indirect memory copy via vector DGE)
    # out-of-bound access" -- see weight_dequant torch-baseline investigation.
    scale = S.repeat_interleave(TILE_SIZE, dim=0).repeat_interleave(TILE_SIZE, dim=1)
    scale = scale[:M, :N]
    return X * scale
