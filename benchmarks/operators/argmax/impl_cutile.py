import math

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _argmax_rowwise_kernel(
    input_tensor,
    output_tensor,
    N_COLS: ConstInt,
    TILE_SIZE: ConstInt,
):
    """
    One CTA per row.
    Loads a padded row (NEG_INF padding), finds the argmax index within
    [0, N_COLS), and stores that index (as float) into output_tensor[:,0].
    """
    row_idx = ct.bid(0)

    tile = ct.load(
        input_tensor,
        index=(row_idx, 0),
        shape=(1, TILE_SIZE),
        padding_mode=ct.PaddingMode.NEG_INF,
    )

    argmax_idx = ct.argmax(tile)
    argmax_idx = ct.minimum(argmax_idx, N_COLS - 1)

    # Write a single float into output_tensor[row_idx, 0].
    idx_tile = tile * 0 + argmax_idx
    ct.store(output_tensor, index=(row_idx, 0), tile=idx_tile)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    """
    cuTile row-wise argmax.
    Input:  (M, N) float tensor
    Output: (M,) int64 indices
    """
    assert x.is_cuda, "x must be on CUDA"

    if dim == 1:
        x2d = x.contiguous()
    else:
        x2d = x.transpose(0, 1).contiguous()

    # Always use float32 so that the index (stored as float) is exact.
    # fp16 can only exactly represent integers up to 2048; larger N would corrupt indices.
    x2d = x2d.float()

    M, N = x2d.shape
    TILE_SIZE = _next_power_of_2(N)

    out2d = torch.empty_like(x2d)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (M, 1, 1), _argmax_rowwise_kernel,
              (x2d, out2d, N, TILE_SIZE))

    return out2d[:, 0].to(torch.int64)


def get_last_config() -> dict | None:
    return None
