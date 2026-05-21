```python title="impl_triton.py"
import torch
import triton
import triton.language as tl


_MAX_BLOCK_SIZE = 4096


@triton.jit
def _vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_units,
    DTYPE_CODE: tl.constexpr,
    MODE: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if MODE == 1:
        # int8 fast path: operate on four packed int8 lanes per int32 word.
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint32)
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint32)
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.uint32)
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.uint32)

        lo_mask = tl.full((BLOCK_SIZE,), 0x00FF00FF, tl.uint32)
        hi_mask = tl.full((BLOCK_SIZE,), 0xFF00FF00, tl.uint32)

        lo = (x_vals & lo_mask) + (y_vals & lo_mask)
        hi = (x_vals & hi_mask) + (y_vals & hi_mask)
        out_vals = (lo & lo_mask) | (hi & hi_mask)

        if MASKED:
            tl.store(out_ptr + offsets, out_vals, mask=mask, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + offsets, out_vals, eviction_policy="evict_first")
    else:
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            )
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            )
            tl.store(out_ptr + offsets, x_vals + y_vals, mask=mask, eviction_policy="evict_first")
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first")
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first")
            tl.store(out_ptr + offsets, x_vals + y_vals, eviction_policy="evict_first")


_TRITON_CONFIGS = [
    triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
    for bs, warps in [
        (64, [2]),
        (128, [2, 4]),
        (256, [2, 4]),
        (512, [2, 4]),
        (1024, [4, 8]),
        (2048, [4, 8]),
        (4096, [4, 8]),
    ]
    for nw in warps
    for ns in [1, 3]
]

_vector_add_kernel_autotuned = triton.autotune(
    configs=_TRITON_CONFIGS,
    key=["n_units", "DTYPE_CODE", "MODE", "MASKED"],
)(_vector_add_kernel)


def _dtype_code(dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    if dtype == torch.int8:
        return 3
    return 4


def run(x, y, autotune: bool = False):
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    dtype_code = _dtype_code(x.dtype)
    mode = 0
    n_units = n_elements
    x_arg = x
    y_arg = y
    out_arg = output

    if (
        x.dtype == torch.int8
        and n_elements % 4 == 0
        and x.is_contiguous()
        and y.is_contiguous()
        and output.is_contiguous()
    ):
        mode = 1
        n_units = n_elements // 4
        x_arg = x.view(torch.int32)
        y_arg = y.view(torch.int32)
        out_arg = output.view(torch.int32)

    masked = (n_units % _MAX_BLOCK_SIZE) != 0
    grid = lambda meta: (triton.cdiv(n_units, meta["BLOCK_SIZE"]),)

    _vector_add_kernel_autotuned[grid](
        x_arg,
        y_arg,
        out_arg,
        n_units,
        dtype_code,
        mode,
        masked,
    )
    return output


def get_last_config() -> dict | None:
    cfg = getattr(_vector_add_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
```

```python title="impl_cutile.py"
from types import SimpleNamespace

import torch
import cuda.tile as ct
from core.cutile_autotune import CutileAutotuner


ConstInt = ct.Constant[int]

_MAX_TILE = 4096
_last_autotune_config: dict = {}

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [64, 128, 256, 512, 1024, 2048, 4096]
    for occ in [4, 8, 16]
]


@ct.kernel
def _vector_add_kernel(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(x, index=(bid,), shape=(TILE,))
    y_tile = ct.load(y, index=(bid,), shape=(TILE,))
    ct.store(output, index=(bid,), tile=x_tile + y_tile)


@ct.kernel
def _vector_add_kernel_padded(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    y_tile = ct.load(
        y,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(output, index=(bid,), tile=x_tile + y_tile)


@ct.kernel
def _vector_add_i8x4_kernel(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(x, index=(bid,), shape=(TILE,))
    y_tile = ct.load(y, index=(bid,), shape=(TILE,))

    lo_mask = 16711935
    hi_mask = -16711936

    lo = (x_tile & lo_mask) + (y_tile & lo_mask)
    hi = (x_tile & hi_mask) + (y_tile & hi_mask)
    out_tile = (lo & lo_mask) + (hi & hi_mask)

    ct.store(output, index=(bid,), tile=out_tile)


@ct.kernel
def _vector_add_i8x4_kernel_padded(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    y_tile = ct.load(
        y,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    lo_mask = 16711935
    hi_mask = -16711936

    lo = (x_tile & lo_mask) + (y_tile & lo_mask)
    hi = (x_tile & hi_mask) + (y_tile & hi_mask)
    out_tile = (lo & lo_mask) + (hi & hi_mask)

    ct.store(output, index=(bid,), tile=out_tile)


_tuner = CutileAutotuner(_vector_add_kernel)
_tuner_padded = CutileAutotuner(_vector_add_kernel_padded)
_tuner_i8x4 = CutileAutotuner(_vector_add_i8x4_kernel)
_tuner_i8x4_padded = CutileAutotuner(_vector_add_i8x4_kernel_padded)


def _dtype_code(dtype) -> int:
    if dtype == torch.float32:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.bfloat16:
        return 2
    if dtype == torch.int8:
        return 3
    return 4


def _launch_tuned(
    tuner,
    mode: int,
    dtype_code: int,
    padded: bool,
    x_arg,
    y_arg,
    output_arg,
    n_units: int,
    stream,
):
    cfg = tuner.tune_or_cached(
        shape_key=(mode, dtype_code, n_units, 1 if padded else 0),
        search_space=_SEARCH_SPACE,
        stream=stream,
        grid_fn=lambda cfg: (ct.cdiv(n_units, cfg.tile), 1, 1),
        args_fn=lambda cfg: (x_arg, y_arg, output_arg, cfg.tile),
        hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
    )

    _last_autotune_config.clear()
    _last_autotune_config.update(
        {
            "tile": cfg.tile,
            "occupancy": cfg.occupancy,
            "mode": mode,
            "padded": 1 if padded else 0,
        }
    )

    kernel = tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream,
        (ct.cdiv(n_units, cfg.tile), 1, 1),
        kernel,
        (x_arg, y_arg, output_arg, cfg.tile),
    )


def run(x, y, autotune: bool = False):
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    dtype_code = _dtype_code(x.dtype)
    mode = 0
    n_units = n_elements
    x_arg = x
    y_arg = y
    out_arg = output

    if (
        x.dtype == torch.int8
        and n_elements % 4 == 0
        and x.is_contiguous()
        and y.is_contiguous()
        and output.is_contiguous()
    ):
        mode = 1
        n_units = n_elements // 4
        x_arg = x.view(torch.int32)
        y_arg = y.view(torch.int32)
        out_arg = output.view(torch.int32)

    padded = (n_units % _MAX_TILE) != 0
    stream = torch.cuda.current_stream()

    if mode == 1:
        tuner = _tuner_i8x4_padded if padded else _tuner_i8x4
    else:
        tuner = _tuner_padded if padded else _tuner

    _launch_tuned(
        tuner,
        mode,
        dtype_code,
        padded,
        x_arg,
        y_arg,
        out_arg,
        n_units,
        stream,
    )
    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
```
