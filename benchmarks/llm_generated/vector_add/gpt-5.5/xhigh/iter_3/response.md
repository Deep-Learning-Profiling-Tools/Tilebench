```python title="impl_triton.py"
import torch
import triton
import triton.language as tl


_MAX_BLOCK_SIZE = 256


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
        # int8 packed as int64: 8 independent wrapping byte additions/word.
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)

        lo_mask = tl.full((BLOCK_SIZE,), 0x00FF00FF00FF00FF, tl.uint64)
        hi_mask = lo_mask << 8

        lo = (x_vals & lo_mask) + (y_vals & lo_mask)
        hi = (x_vals & hi_mask) + (y_vals & hi_mask)
        out_vals = (lo & lo_mask) | (hi & hi_mask)

        if MASKED:
            tl.store(out_ptr + offsets, out_vals, mask=mask, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + offsets, out_vals, eviction_policy="evict_first")

    elif MODE == 2:
        # fp32 packed as int64: 2 float32 additions/word with one 64-bit load/store.
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)

        mask32 = tl.full((BLOCK_SIZE,), 0xFFFFFFFF, tl.uint64)

        x_lo_bits = (x_vals & mask32).to(tl.uint32)
        y_lo_bits = (y_vals & mask32).to(tl.uint32)
        x_hi_bits = (x_vals >> 32).to(tl.uint32)
        y_hi_bits = (y_vals >> 32).to(tl.uint32)

        x_lo = x_lo_bits.to(tl.float32, bitcast=True)
        y_lo = y_lo_bits.to(tl.float32, bitcast=True)
        x_hi = x_hi_bits.to(tl.float32, bitcast=True)
        y_hi = y_hi_bits.to(tl.float32, bitcast=True)

        out_lo_bits = (x_lo + y_lo).to(tl.uint32, bitcast=True)
        out_hi_bits = (x_hi + y_hi).to(tl.uint32, bitcast=True)
        out_vals = out_lo_bits.to(tl.uint64) | (out_hi_bits.to(tl.uint64) << 32)

        if MASKED:
            tl.store(out_ptr + offsets, out_vals, mask=mask, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + offsets, out_vals, eviction_policy="evict_first")

    elif MODE == 3:
        # fp16 packed as int64: 4 fp16 additions/word via two f16x2 instructions.
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)

        mask32 = tl.full((BLOCK_SIZE,), 0xFFFFFFFF, tl.uint64)
        x_lo = (x_vals & mask32).to(tl.uint32)
        y_lo = (y_vals & mask32).to(tl.uint32)
        x_hi = (x_vals >> 32).to(tl.uint32)
        y_hi = (y_vals >> 32).to(tl.uint32)

        out_lo = tl.inline_asm_elementwise(
            "add.rn.f16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[x_lo, y_lo],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        out_hi = tl.inline_asm_elementwise(
            "add.rn.f16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[x_hi, y_hi],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        out_vals = out_lo.to(tl.uint64) | (out_hi.to(tl.uint64) << 32)

        if MASKED:
            tl.store(out_ptr + offsets, out_vals, mask=mask, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + offsets, out_vals, eviction_policy="evict_first")

    elif MODE == 4:
        # bf16 packed as int64: 4 bf16 additions/word via two bf16x2 instructions.
        if MASKED:
            mask = offsets < n_units
            x_vals = tl.load(
                x_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
            y_vals = tl.load(
                y_ptr + offsets,
                mask=mask,
                other=0,
                eviction_policy="evict_first",
            ).to(tl.uint64)
        else:
            x_vals = tl.load(x_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)
            y_vals = tl.load(y_ptr + offsets, eviction_policy="evict_first").to(tl.uint64)

        mask32 = tl.full((BLOCK_SIZE,), 0xFFFFFFFF, tl.uint64)
        x_lo = (x_vals & mask32).to(tl.uint32)
        y_lo = (y_vals & mask32).to(tl.uint32)
        x_hi = (x_vals >> 32).to(tl.uint32)
        y_hi = (y_vals >> 32).to(tl.uint32)

        out_lo = tl.inline_asm_elementwise(
            "add.rn.bf16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[x_lo, y_lo],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        out_hi = tl.inline_asm_elementwise(
            "add.rn.bf16x2 $0, $1, $2;",
            constraints="=r,r,r",
            args=[x_hi, y_hi],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        out_vals = out_lo.to(tl.uint64) | (out_hi.to(tl.uint64) << 32)

        if MASKED:
            tl.store(out_ptr + offsets, out_vals, mask=mask, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + offsets, out_vals, eviction_policy="evict_first")

    else:
        # Generic scalar fallback.
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
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
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

    if x.dtype == torch.int8 and n_elements % 8 == 0:
        mode = 1
        n_units = n_elements // 8
        x_arg = x.view(torch.int64)
        y_arg = y.view(torch.int64)
        out_arg = output.view(torch.int64)
    elif x.dtype == torch.float32 and n_elements % 2 == 0:
        mode = 2
        n_units = n_elements // 2
        x_arg = x.view(torch.int64)
        y_arg = y.view(torch.int64)
        out_arg = output.view(torch.int64)
    elif x.dtype == torch.float16 and n_elements % 4 == 0:
        mode = 3
        n_units = n_elements // 4
        x_arg = x.view(torch.int64)
        y_arg = y.view(torch.int64)
        out_arg = output.view(torch.int64)
    elif x.dtype == torch.bfloat16 and n_elements % 4 == 0:
        mode = 4
        n_units = n_elements // 4
        x_arg = x.view(torch.int64)
        y_arg = y.view(torch.int64)
        out_arg = output.view(torch.int64)

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

_last_autotune_config: dict = {}

_VEC_PAIRS = [
    (16, 4),
    (32, 4),
    (64, 4),
    (16, 8),
    (32, 8),
    (64, 8),
    (16, 16),
    (32, 16),
]
_I8X8_PAIRS = [
    (16, 1),
    (32, 1),
    (64, 1),
    (16, 2),
    (32, 2),
    (64, 2),
    (16, 4),
    (32, 4),
]

_MAX_VEC_TILE_ELEMENTS = 512
_MAX_I8X8_WORD_TILE = 128
_MAX_1D_TILE = 512

_SEARCH_SPACE_VEC = [
    SimpleNamespace(rows=r, vec=v, occupancy=occ)
    for r, v in _VEC_PAIRS
    for occ in [4, 8, 16]
]

_SEARCH_SPACE_I8X8 = [
    SimpleNamespace(rows=r, vec=v, occupancy=occ)
    for r, v in _I8X8_PAIRS
    for occ in [4, 8, 16]
]

_SEARCH_SPACE_1D = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [128, 256, 512]
    for occ in [4, 8, 16]
]


@ct.kernel
def _vector_add_vec2d_kernel(x, y, output, ROWS: ConstInt, VEC: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(x, index=(bid, 0), shape=(ROWS, VEC))
    y_tile = ct.load(y, index=(bid, 0), shape=(ROWS, VEC))
    ct.store(output, index=(bid, 0), tile=x_tile + y_tile)


@ct.kernel
def _vector_add_vec2d_kernel_padded(x, y, output, ROWS: ConstInt, VEC: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid, 0),
        shape=(ROWS, VEC),
        padding_mode=ct.PaddingMode.ZERO,
    )
    y_tile = ct.load(
        y,
        index=(bid, 0),
        shape=(ROWS, VEC),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(output, index=(bid, 0), tile=x_tile + y_tile)


@ct.kernel
def _vector_add_i8x8_vec2d_kernel(x, y, output, ROWS: ConstInt, VEC: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(x, index=(bid, 0), shape=(ROWS, VEC))
    y_tile = ct.load(y, index=(bid, 0), shape=(ROWS, VEC))

    lo_mask = 71777214294589695
    hi_mask = -71777214294589696

    lo = (x_tile & lo_mask) + (y_tile & lo_mask)
    hi = (x_tile & hi_mask) + (y_tile & hi_mask)
    out_tile = (lo & lo_mask) + (hi & hi_mask)

    ct.store(output, index=(bid, 0), tile=out_tile)


@ct.kernel
def _vector_add_i8x8_vec2d_kernel_padded(x, y, output, ROWS: ConstInt, VEC: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid, 0),
        shape=(ROWS, VEC),
        padding_mode=ct.PaddingMode.ZERO,
    )
    y_tile = ct.load(
        y,
        index=(bid, 0),
        shape=(ROWS, VEC),
        padding_mode=ct.PaddingMode.ZERO,
    )

    lo_mask = 71777214294589695
    hi_mask = -71777214294589696

    lo = (x_tile & lo_mask) + (y_tile & lo_mask)
    hi = (x_tile & hi_mask) + (y_tile & hi_mask)
    out_tile = (lo & lo_mask) + (hi & hi_mask)

    ct.store(output, index=(bid, 0), tile=out_tile)


@ct.kernel
def _vector_add_1d_kernel(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(x, index=(bid,), shape=(TILE,))
    y_tile = ct.load(y, index=(bid,), shape=(TILE,))
    ct.store(output, index=(bid,), tile=x_tile + y_tile)


@ct.kernel
def _vector_add_1d_kernel_padded(x, y, output, TILE: ConstInt):
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


_tuner_vec = CutileAutotuner(_vector_add_vec2d_kernel)
_tuner_vec_padded = CutileAutotuner(_vector_add_vec2d_kernel_padded)
_tuner_i8x8 = CutileAutotuner(_vector_add_i8x8_vec2d_kernel)
_tuner_i8x8_padded = CutileAutotuner(_vector_add_i8x8_vec2d_kernel_padded)
_tuner_1d = CutileAutotuner(_vector_add_1d_kernel)
_tuner_1d_padded = CutileAutotuner(_vector_add_1d_kernel_padded)


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


def _launch_vec_tuned(
    tuner,
    mode: int,
    dtype_code: int,
    padded: bool,
    x,
    y,
    output,
    n_elements: int,
    stream,
):
    cfg = tuner.tune_or_cached(
        shape_key=(mode, dtype_code, n_elements, 1 if padded else 0),
        search_space=_SEARCH_SPACE_VEC,
        stream=stream,
        grid_fn=lambda cfg: (ct.cdiv(n_elements // cfg.vec, cfg.rows), 1, 1),
        args_fn=lambda cfg: (
            x.view(-1, cfg.vec),
            y.view(-1, cfg.vec),
            output.view(-1, cfg.vec),
            cfg.rows,
            cfg.vec,
        ),
        hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
    )

    _last_autotune_config.clear()
    _last_autotune_config.update(
        {
            "rows": cfg.rows,
            "vec": cfg.vec,
            "occupancy": cfg.occupancy,
            "mode": mode,
            "padded": 1 if padded else 0,
        }
    )

    kernel = tuner.kernel_with_hints(occupancy=cfg.occupancy)
    row_count = n_elements // cfg.vec
    ct.launch(
        stream,
        (ct.cdiv(row_count, cfg.rows), 1, 1),
        kernel,
        (
            x.view(-1, cfg.vec),
            y.view(-1, cfg.vec),
            output.view(-1, cfg.vec),
            cfg.rows,
            cfg.vec,
        ),
    )


def _launch_i8x8_tuned(
    tuner,
    mode: int,
    dtype_code: int,
    padded: bool,
    x_words,
    y_words,
    output_words,
    n_words: int,
    stream,
):
    cfg = tuner.tune_or_cached(
        shape_key=(mode, dtype_code, n_words, 1 if padded else 0),
        search_space=_SEARCH_SPACE_I8X8,
        stream=stream,
        grid_fn=lambda cfg: (ct.cdiv(n_words // cfg.vec, cfg.rows), 1, 1),
        args_fn=lambda cfg: (
            x_words.view(-1, cfg.vec),
            y_words.view(-1, cfg.vec),
            output_words.view(-1, cfg.vec),
            cfg.rows,
            cfg.vec,
        ),
        hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
    )

    _last_autotune_config.clear()
    _last_autotune_config.update(
        {
            "rows": cfg.rows,
            "vec": cfg.vec,
            "occupancy": cfg.occupancy,
            "mode": mode,
            "padded": 1 if padded else 0,
        }
    )

    kernel = tuner.kernel_with_hints(occupancy=cfg.occupancy)
    row_count = n_words // cfg.vec
    ct.launch(
        stream,
        (ct.cdiv(row_count, cfg.rows), 1, 1),
        kernel,
        (
            x_words.view(-1, cfg.vec),
            y_words.view(-1, cfg.vec),
            output_words.view(-1, cfg.vec),
            cfg.rows,
            cfg.vec,
        ),
    )


def _launch_1d_tuned(
    tuner,
    mode: int,
    dtype_code: int,
    padded: bool,
    x,
    y,
    output,
    n_elements: int,
    stream,
):
    cfg = tuner.tune_or_cached(
        shape_key=(mode, dtype_code, n_elements, 1 if padded else 0),
        search_space=_SEARCH_SPACE_1D,
        stream=stream,
        grid_fn=lambda cfg: (ct.cdiv(n_elements, cfg.tile), 1, 1),
        args_fn=lambda cfg: (x, y, output, cfg.tile),
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
        (ct.cdiv(n_elements, cfg.tile), 1, 1),
        kernel,
        (x, y, output, cfg.tile),
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
    stream = torch.cuda.current_stream()

    if x.dtype == torch.int8 and n_elements % 32 == 0:
        n_words = n_elements // 8
        x_words = x.view(torch.int64)
        y_words = y.view(torch.int64)
        output_words = output.view(torch.int64)
        padded = (n_words % _MAX_I8X8_WORD_TILE) != 0
        tuner = _tuner_i8x8_padded if padded else _tuner_i8x8
        _launch_i8x8_tuned(
            tuner,
            1,
            dtype_code,
            padded,
            x_words,
            y_words,
            output_words,
            n_words,
            stream,
        )
    elif n_elements % 16 == 0:
        padded = (n_elements % _MAX_VEC_TILE_ELEMENTS) != 0
        tuner = _tuner_vec_padded if padded else _tuner_vec
        _launch_vec_tuned(
            tuner,
            2,
            dtype_code,
            padded,
            x,
            y,
            output,
            n_elements,
            stream,
        )
    else:
        padded = (n_elements % _MAX_1D_TILE) != 0
        tuner = _tuner_1d_padded if padded else _tuner_1d
        _launch_1d_tuned(
            tuner,
            0,
            dtype_code,
            padded,
            x,
            y,
            output,
            n_elements,
            stream,
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
```
