```python title="impl_triton.py"
import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y

    tl.store(out_ptr + offsets, out, mask=mask)


_vector_add_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [512, 1024, 2048, 4096, 8192]
        for nw in [4, 8]
    ],
    key=["n_elements"],
)(_vector_add_kernel)


def run(x, y, autotune: bool = False):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _vector_add_kernel_autotuned[grid](x, y, output, n_elements)
    return output


def get_last_config() -> dict | None:
    cfg = getattr(_vector_add_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps": cfg.num_warps,
    }
```


```python title="impl_cutile.py"
from types import SimpleNamespace

import torch
import cuda.tile as ct
from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [512, 1024, 2048, 4096, 8192]
    for occ in [4, 8, 16]
]


@ct.kernel
def _vector_add_kernel(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    y_tile = ct.load(
        y,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    out_tile = x_tile + y_tile
    ct.store(output, index=(bid,), tile=out_tile, allow_tma=False)


_tuner = CutileAutotuner(_vector_add_kernel)


def run(x, y, autotune: bool = False):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    cfg = _tuner.tune_or_cached(
        shape_key=(n_elements,),
        search_space=_SEARCH_SPACE,
        stream=stream,
        grid_fn=lambda cfg: (ct.cdiv(n_elements, cfg.tile), 1, 1),
        args_fn=lambda cfg: (x, y, output, cfg.tile),
        hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
    )

    _last_autotune_config.clear()
    _last_autotune_config.update({"tile": cfg.tile, "occupancy": cfg.occupancy})

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream,
        (ct.cdiv(n_elements, cfg.tile), 1, 1),
        kernel,
        (x, y, output, cfg.tile),
    )
    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
```
