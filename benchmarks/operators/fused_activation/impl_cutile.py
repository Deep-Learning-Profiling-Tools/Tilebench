from types import SimpleNamespace

import cuda.tile as ct
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=2)

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048, 4096, 8192]
    for occ in [1, 2, 4]
]


@ct.kernel
def _fused_kernel(x, gate, bias, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.astype(ct.load(x, index=(bid,), shape=(TILE,)), ct.float32)
    gate_tile = ct.astype(ct.load(gate, index=(bid,), shape=(TILE,)), ct.float32)
    bias_tile = ct.astype(ct.load(bias, index=(bid,), shape=(TILE,)), ct.float32)
    z = x_tile * gate_tile + bias_tile
    out_tile = ct.maximum(z, 0.0)
    ct.store(output, index=(bid,), tile=out_tile)


def run(
    x: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
    block_size: int = 1024,
    autotune: bool = False,
) -> torch.Tensor:
    global _last_autotune_config
    if x.shape != gate.shape or x.shape != bias.shape:
        raise ValueError("All input tensors must have the same shape.")
    x = x.contiguous()
    gate = gate.contiguous()
    bias = bias.contiguous()
    output = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1),
            kernel=_fused_kernel,
            args_fn=lambda cfg: (x, gate, bias, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile": result.tuned_config.tile,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        ct.launch(stream, ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1),
                  _fused_kernel, (x, gate, bias, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
