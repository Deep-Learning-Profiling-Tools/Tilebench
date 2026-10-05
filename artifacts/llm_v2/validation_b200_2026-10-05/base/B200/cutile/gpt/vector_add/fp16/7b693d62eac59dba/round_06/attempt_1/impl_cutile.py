from typing import Annotated

import torch
import cuda.tile as ct


FlatArray = Annotated[
    ct.Array,
    ct.ArrayAnnotation(index_dtype=ct.int32, static_shape_dims=(0,)),
]


@ct.function
def _add_chunk(x, y, out, index):
    a = ct.load(
        x, (index,), (8192,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    b = ct.load(
        y, (index,), (8192,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    ct.store(out, (index,), a + b, allow_tma=False)


@ct.kernel(num_ctas=1, occupancy=4, opt_level=3)
def _vector_add(
    x: FlatArray,
    y: FlatArray,
    out: FlatArray,
    N: ct.Constant[int],
):
    index = ct.bid(0) * 2
    _add_chunk(x, y, out, index)

    if N % 16384 == 0:
        _add_chunk(x, y, out, index + 1)
    else:
        if (index + 1) * 8192 < N:
            _add_chunk(x, y, out, index + 1)


def run(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    if n > 0:
        ct.launch(
            torch.cuda.current_stream(x.device),
            (ct.cdiv(n, 16384),),
            _vector_add,
            (x, y, out, n),
        )
    return out


def get_last_config() -> dict:
    return {
        "tile_size": 8192,
        "tiles_per_block": 2,
        "elements_per_block": 16384,
        "num_ctas": 1,
        "occupancy": 4,
        "opt_level": 3,
        "allow_tma": False,
        "static_shape_dims": (0,),
    }
