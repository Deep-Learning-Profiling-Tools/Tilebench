import torch

_LAST_CFG: dict | None = None


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    # cuTile lacks a built-in sort primitive; bitonic-sort variants via
    # reshape/extract/cat/where fail to compile. Skip cleanly so triton is unaffected.
    raise NotImplementedError(
        "cuTile top_k_selection: no sort primitive; bitonic emulation fails to compile."
    )


def get_last_config() -> dict | None:
    return None
