```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _relu_kernel_exact(x, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        allow_tma=False,
    )

    # Match torch.relu/threshold semantics: x <= 0 becomes +0, NaNs pass through.
    y_tile = ct.where(x_tile <= 0, 0, x_tile)

    ct.store(
        output,
        index=(bid,),
        tile=y_tile,
        allow_tma=False,
    )


@ct.kernel(occupancy=8)
def _relu_kernel_padded(x, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )

    # Match torch.relu/threshold semantics: x <= 0 becomes +0, NaNs pass through.
    y_tile = ct.where(x_tile <= 0, 0, x_tile)

    ct.store(
        output,
        index=(bid,),
        tile=y_tile,
        allow_tma=False,
    )


@ct.kernel(occupancy=8)
def _relu_int8x4_kernel_exact(x_words, output_words, TILE_WORDS: ConstInt):
    bid = ct.bid(0)

    w_tile = ct.load(
        x_words,
        index=(bid,),
        shape=(TILE_WORDS,),
        allow_tma=False,
    )

    # Four signed int8 lanes are packed in each int32 word.  ReLU for int8 is
    # byte & 0xff for non-negative lanes and 0 for negative lanes.
    u_tile = ct.bitcast(w_tile, ct.uint32)
    high_mask = ct.full((TILE_WORDS,), np.uint32(0x80808080), dtype=ct.uint32)
    byte_ff = ct.full((TILE_WORDS,), np.uint32(0x000000FF), dtype=ct.uint32)

    sign_lsb = ct.bitwise_rshift(ct.bitwise_and(u_tile, high_mask), 7)
    neg_mask = sign_lsb * byte_ff
    y_u = ct.bitwise_and(u_tile, ct.bitwise_not(neg_mask))

    ct.store(
        output_words,
        index=(bid,),
        tile=ct.bitcast(y_u, ct.int32),
        allow_tma=False,
    )


@ct.kernel(occupancy=8)
def _relu_int8x4_kernel_padded(x_words, output_words, TILE_WORDS: ConstInt):
    bid = ct.bid(0)

    w_tile = ct.load(
        x_words,
        index=(bid,),
        shape=(TILE_WORDS,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )

    u_tile = ct.bitcast(w_tile, ct.uint32)
    high_mask = ct.full((TILE_WORDS,), np.uint32(0x80808080), dtype=ct.uint32)
    byte_ff = ct.full((TILE_WORDS,), np.uint32(0x000000FF), dtype=ct.uint32)

    sign_lsb = ct.bitwise_rshift(ct.bitwise_and(u_tile, high_mask), 7)
    neg_mask = sign_lsb * byte_ff
    y_u = ct.bitwise_and(u_tile, ct.bitwise_not(neg_mask))

    ct.store(
        output_words,
        index=(bid,),
        tile=ct.bitcast(y_u, ct.int32),
        allow_tma=False,
    )


def run(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE_BYTES = 8192
    occupancy = 8
    allow_tma = False

    if x.dtype == torch.int8 and (n_elements % 4) == 0:
        x_words = x.view(torch.int32)
        output_words = output.view(torch.int32)

        TILE_WORDS = TILE_BYTES // 4
        n_words = n_elements // 4
        use_exact_kernel = (n_words % TILE_WORDS) == 0

        if use_exact_kernel:
            grid = (n_words // TILE_WORDS, 1, 1)
            ct.launch(stream, grid, _relu_int8x4_kernel_exact, (x_words, output_words, TILE_WORDS))
        else:
            grid = (ct.cdiv(n_words, TILE_WORDS), 1, 1)
            ct.launch(stream, grid, _relu_int8x4_kernel_padded, (x_words, output_words, TILE_WORDS))

        _LAST_CFG.clear()
        _LAST_CFG.update({
            "TILE_BYTES": TILE_BYTES,
            "TILE_WORDS": TILE_WORDS,
            "elements_per_cta": TILE_WORDS * 4,
            "pack_factor": 4,
            "occupancy": occupancy,
            "allow_tma": allow_tma,
            "tail_padding": not use_exact_kernel,
        })
        return output

    elem_size = x.element_size()
    TILE = TILE_BYTES // elem_size
    use_exact_kernel = (n_elements % TILE) == 0

    if use_exact_kernel:
        grid = (n_elements // TILE, 1, 1)
        ct.launch(stream, grid, _relu_kernel_exact, (x, output, TILE))
    else:
        grid = (ct.cdiv(n_elements, TILE), 1, 1)
        ct.launch(stream, grid, _relu_kernel_padded, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE_BYTES": TILE_BYTES,
        "TILE": TILE,
        "elements_per_cta": TILE,
        "pack_factor": 1,
        "occupancy": occupancy,
        "allow_tma": allow_tma,
        "tail_padding": not use_exact_kernel,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
