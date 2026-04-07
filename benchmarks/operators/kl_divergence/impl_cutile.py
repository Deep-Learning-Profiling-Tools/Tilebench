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
def _kl_divergence_kernel(p, q, output, eps, TILE: ConstInt):
    bid = ct.bid(0)
    p_tile = ct.astype(ct.load(p, index=(bid,), shape=(TILE,)), ct.float32)
    q_tile = ct.astype(ct.load(q, index=(bid,), shape=(TILE,)), ct.float32)
    p_tile = ct.maximum(p_tile, eps)
    q_tile = ct.maximum(q_tile, eps)
    out_tile = p_tile * (ct.log(p_tile) - ct.log(q_tile))
    ct.store(output, index=(bid,), tile=out_tile)


def run(p: torch.Tensor, q: torch.Tensor, eps: float,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_autotune_config
    if p.shape != q.shape:
        raise ValueError("Input tensors must have the same shape.")
    p = p.contiguous()
    q = q.contiguous()
    output = torch.empty(p.shape, device=p.device, dtype=torch.float32)
    n_elements = p.numel()
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1),
            kernel=_kl_divergence_kernel,
            args_fn=lambda cfg: (p, q, output, eps, cfg.tile),
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
                  _kl_divergence_kernel, (p, q, output, eps, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
