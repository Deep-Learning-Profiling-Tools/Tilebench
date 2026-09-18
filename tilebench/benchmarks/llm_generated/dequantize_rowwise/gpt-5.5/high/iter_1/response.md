```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _dequantize_rowwise_kernel(
    x_ptr,
    state_ptr,
    out_ptr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = offs_m[:, None] * cols + offs_n[None, :]

    if USE_MASK:
        mask = (offs_m[:, None] < rows) & (offs_n[None, :] < cols)
        x_i8 = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
        scale = tl.load(state_ptr + offs_m, mask=offs_m < rows, other=0.0).to(tl.float32)
        scale = scale * 0.007874015748031496
        y = x_i8 * scale[:, None]
        tl.store(out_ptr + offsets, y, mask=mask)
    else:
        x_i8 = tl.load(x_ptr + offsets).to(tl.float32)
        scale = tl.load(state_ptr + offs_m).to(tl.float32)
        scale = scale * 0.007874015748031496
        y = x_i8 * scale[:, None]
        tl.store(out_ptr + offsets, y)


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    rows = x.shape[0]
    cols = x.shape[1]
    output = torch.empty((rows, cols), device=x.device, dtype=torch.float16)

    BLOCK_M = 4
    BLOCK_N = 1024
    num_warps = 8
    num_stages = 2
    USE_MASK = (rows % BLOCK_M) != 0 or (cols % BLOCK_N) != 0

    grid = (triton.cdiv(rows, BLOCK_M), triton.cdiv(cols, BLOCK_N))
    _dequantize_rowwise_kernel[grid](
        x,
        state_x,
        output,
        rows=rows,
        cols=cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        USE_MASK=USE_MASK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "USE_MASK": USE_MASK,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
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
    bid_n = ct.bid(1)

    if USE_PAD:
        x_tile = ct.load(
            x,
            index=(bid_m, bid_n),
            shape=(TILE_M, TILE_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        scale = ct.load(
            state_x,
            index=(bid_m,),
            shape=(TILE_M,),
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
        scale = ct.load(
            state_x,
            index=(bid_m,),
            shape=(TILE_M,),
            latency=1,
            allow_tma=False,
        )

    x_f32 = ct.astype(x_tile, np.float32)
    scale_f32 = ct.astype(scale, np.float32) * 0.007874015748031496
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

    grid = (ct.cdiv(rows, TILE_M), ct.cdiv(cols, TILE_N), 1)
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
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
