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
