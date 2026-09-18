import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _dequantize_rowwise_kernel(
    x,
    state_x,
    output,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    USE_PAD: ConstBool,
):
    bid_m = ct.bid(0)
    num_n_tiles = ct.cdiv(x.shape[1], TILE_N)

    if USE_PAD:
        scale = ct.load(
            state_x,
            index=(bid_m,),
            shape=(TILE_M,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
    else:
        scale = ct.load(
            state_x,
            index=(bid_m,),
            shape=(TILE_M,),
            latency=1,
            allow_tma=False,
        )

    scale_f32 = ct.astype(scale, np.float32) * 0.007874015748031496

    for bid_n in range(0, num_n_tiles):
        if USE_PAD:
            x_tile = ct.load(
                x,
                index=(bid_m, bid_n),
                shape=(TILE_M, TILE_N),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            )
        else:
            x_tile = ct.load(
                x,
                index=(bid_m, bid_n),
                shape=(TILE_M, TILE_N),
                latency=1,
                allow_tma=False,
            )

        x_f32 = ct.astype(x_tile, np.float32)
        y = x_f32 * scale_f32[:, None]

        ct.store(
            output,
            index=(bid_m, bid_n),
            tile=ct.astype(y, output.dtype),
            latency=1,
            allow_tma=False,
        )


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    rows = x.shape[0]
    cols = x.shape[1]
    output = torch.empty((rows, cols), device=x.device, dtype=torch.float16)
    stream = torch.cuda.current_stream()

    TILE_M = 4
    TILE_N = 1024
    occupancy = 4
    USE_PAD = (rows % TILE_M) != 0 or (cols % TILE_N) != 0

    grid = (ct.cdiv(rows, TILE_M), 1, 1)
    ct.launch(
        stream,
        grid,
        _dequantize_rowwise_kernel,
        (x, state_x, output, TILE_M, TILE_N, USE_PAD),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE_M": TILE_M,
            "TILE_N": TILE_N,
            "occupancy": occupancy,
            "USE_PAD": USE_PAD,
            "LOOP_N": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
