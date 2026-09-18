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
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    row = pid_m * BLOCK_M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = row * cols + offs_n

    if USE_MASK:
        mask = (row < rows) & (offs_n < cols)
        x_i8 = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
        scale = tl.load(state_ptr + row, mask=row < rows, other=0.0).to(tl.float32)
        scale = scale * 0.007874015748031496
        y = x_i8 * scale
        tl.store(out_ptr + offsets, y.to(tl.float16), mask=mask)
    else:
        x_i8 = tl.load(x_ptr + offsets).to(tl.float32)
        scale = tl.load(state_ptr + row).to(tl.float32)
        scale = scale * 0.007874015748031496
        y = x_i8 * scale
        tl.store(out_ptr + offsets, y.to(tl.float16))


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    rows = x.shape[0]
    cols = x.shape[1]
    output = torch.empty((rows, cols), device=x.device, dtype=torch.float16)

    BLOCK_M = 1
    BLOCK_N = 4096
    num_warps = 8
    num_stages = 2
    USE_MASK = (rows % BLOCK_M) != 0 or (cols % BLOCK_N) != 0

    grid = (triton.cdiv(cols, BLOCK_N), triton.cdiv(rows, BLOCK_M))
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
            "GRID_N_FAST": True,
            "ROW_CONTIGUOUS": True,
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


@ct.kernel(occupancy=8)
def _dequantize_rowwise_kernel(
    x,
    state_x,
    output,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    USE_PAD: ConstBool,
):
    bid_n = ct.bid(0)
    bid_m = ct.bid(1)

    scale = ct.load(
        state_x,
        index=(bid_m,),
        shape=(TILE_M,),
        latency=1,
        allow_tma=False,
    )

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

    x_h = ct.astype(x_tile, np.float16)
    scale_h = ct.astype(ct.astype(scale, np.float32) * 0.007874015748031496, np.float16)
    y = x_h * scale_h[:, None]

    ct.store(
        output,
        index=(bid_m, bid_n),
        tile=y,
        latency=1,
        allow_tma=False,
    )


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    rows = x.shape[0]
    cols = x.shape[1]
    output = torch.empty((rows, cols), device=x.device, dtype=torch.float16)
    stream = torch.cuda.current_stream()

    TILE_M = 1
    TILE_N = 4096
    occupancy = 8
    USE_PAD = (rows % TILE_M) != 0 or (cols % TILE_N) != 0

    grid = (ct.cdiv(cols, TILE_N), ct.cdiv(rows, TILE_M), 1)
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
            "GRID_N_FAST": True,
            "HALF_COMPUTE": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
