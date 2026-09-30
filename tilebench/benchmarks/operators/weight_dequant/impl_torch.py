import torch


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    if X.device.type == "xla":
        scale = S.repeat_interleave(TILE_SIZE, dim=0).repeat_interleave(TILE_SIZE, dim=1)
        return X * scale[:M, :N]
    row_idx = torch.arange(M, device=X.device) // TILE_SIZE
    col_idx = torch.arange(N, device=X.device) // TILE_SIZE
    scale = S[row_idx[:, None], col_idx[None, :]]
    return X * scale
