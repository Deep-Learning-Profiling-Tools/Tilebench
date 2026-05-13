import torch

_LAST_CONFIG = None


def run(
    input,
    N: int,
    k: int,
    BLOCK_SIZE: int = 1024,
    block_size: int = None,
    autotune: bool = False,
    **kwargs,
):
    global _LAST_CONFIG

    if block_size is not None:
        BLOCK_SIZE = int(block_size)

    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.float32
    assert 1 <= k <= N

    input = input.contiguous()
    output = torch.topk(input, k, largest=True, sorted=True).values

    _LAST_CONFIG = {
        "BLOCK_SIZE": int(BLOCK_SIZE),
    }
    return output


def get_last_config() -> dict | None:
    return _LAST_CONFIG