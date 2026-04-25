import torch

_LAST_CONFIG = None


def run(
    input,
    N: int,
    num_bins: int,
    BLOCK_SIZE: int = 1024,
    NUM_PARTIAL: int = 256,
    BLOCK_ROWS: int = 64,
    BLOCK_BINS: int = 256,
    block_size: int = None,
    autotune: bool = False,
    **kwargs,
):
    global _LAST_CONFIG

    if block_size is not None:
        BLOCK_SIZE = int(block_size)

    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    assert num_bins >= 1

    input = input.contiguous()
    histogram = torch.bincount(input.to(torch.int64), minlength=num_bins).to(torch.int32)

    _LAST_CONFIG = {
        "BLOCK_SIZE": int(BLOCK_SIZE),
        "NUM_PARTIAL": int(NUM_PARTIAL),
        "BLOCK_ROWS": int(BLOCK_ROWS),
        "BLOCK_BINS": int(BLOCK_BINS),
    }
    return histogram


def get_last_config() -> dict | None:
    return _LAST_CONFIG