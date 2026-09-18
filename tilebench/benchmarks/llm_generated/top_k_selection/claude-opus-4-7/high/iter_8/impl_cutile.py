import torch

_LAST_CFG: dict | None = None


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    # cuTile lacks a built-in sort primitive; hand-rolled bitonic sort
    # via reshape/extract/cat/where consistently fails to compile on the
    # toolchain. Skip cuTile cleanly so it doesn't block triton.
    raise NotImplementedError(
        "cuTile top_k_selection: bitonic sort variants fail to compile; "
        "skipping until cuTile exposes a sort primitive."
    )


def get_last_config() -> dict | None:
    return None
