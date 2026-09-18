```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _mean_reduction_pair_flat_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
    PAIR_N: tl.constexpr,
    EVEN_TILE: tl.constexpr,
    EVEN_PAIR: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    base = row * N

    tl.assume(N > 0)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start in tl.range(0, N, PAIR_N, num_stages=LOOP_STAGES):
        offs0 = start + cols
        ptrs0 = x_ptr + base + offs0
        if EVEN_TILE:
            vals0 = tl.load(
                ptrs0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            vals0 = tl.load(
                ptrs0,
                mask=offs0 < N,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)

        offs1 = start + BLOCK_N + cols
        ptrs1 = x_ptr + base + offs1
        if EVEN_PAIR:
            vals1 = tl.load(
                ptrs1,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            vals1 = tl.load(
                ptrs1,
                mask=offs1 < N,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)

        acc += vals0 + vals1

    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, total * (1.0 / N))


@triton.jit
def _mean_reduction_pair_strided_kernel(
    x_ptr,
    out_ptr,
    stride_m,
    stride_n,
    N,
    BLOCK_N: tl.constexpr,
    PAIR_N: tl.constexpr,
    EVEN_TILE: tl.constexpr,
    EVEN_PAIR: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)

    tl.assume(stride_m > 0)
    tl.assume(stride_n > 0)
    tl.assume(N > 0)

    row_ptr = x_ptr + row * stride_m
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start in tl.range(0, N, PAIR_N, num_stages=LOOP_STAGES):
        offs0 = start + cols
        ptrs0 = row_ptr + offs0 * stride_n
        if EVEN_TILE:
            vals0 = tl.load(
                ptrs0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            vals0 = tl.load(
                ptrs0,
                mask=offs0 < N,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)

        offs1 = start + BLOCK_N + cols
        ptrs1 = row_ptr + offs1 * stride_n
        if EVEN_PAIR:
            vals1 = tl.load(
                ptrs1,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            vals1 = tl.load(
                ptrs1,
                mask=offs1 < N,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)

        acc += vals0 + vals1

    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, total * (1.0 / N))


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim < 0:
        dim = dim + x.dim()
    if dim != 1:
        raise NotImplementedError("mean_reduction only supports row-wise dim=1")

    M = x.shape[0]
    N = x.shape[1]
    output = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_N = 1024
    PAIR_N = 2048
    num_warps = 4
    num_stages = 3
    LOOP_STAGES = 3

    EVEN_TILE = (N % BLOCK_N) == 0
    EVEN_PAIR = (N % PAIR_N) == 0
    USE_FLAT = x.is_contiguous()

    if USE_FLAT:
        _mean_reduction_pair_flat_kernel[(M,)](
            x,
            output,
            N,
            BLOCK_N=BLOCK_N,
            PAIR_N=PAIR_N,
            EVEN_TILE=EVEN_TILE,
            EVEN_PAIR=EVEN_PAIR,
            LOOP_STAGES=LOOP_STAGES,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _mean_reduction_pair_strided_kernel[(M,)](
            x,
            output,
            x.stride(0),
            x.stride(1),
            N,
            BLOCK_N=BLOCK_N,
            PAIR_N=PAIR_N,
            EVEN_TILE=EVEN_TILE,
            EVEN_PAIR=EVEN_PAIR,
            LOOP_STAGES=LOOP_STAGES,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "ALGO": "flat_pair_unroll2" if USE_FLAT else "strided_pair_unroll2",
            "BLOCK_N": BLOCK_N,
            "PAIR_N": PAIR_N,
            "PAIR_UNROLL": 2,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "LOOP_STAGES": LOOP_STAGES,
            "EVEN_TILE": EVEN_TILE,
            "EVEN_PAIR": EVEN_PAIR,
            "USE_FLAT": USE_FLAT,
            "cache_modifier": ".cg",
            "eviction_policy": "evict_first",
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
def _mean_reduction_pair_flat_kernel(x_flat, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    full_pairs = num_tiles // 2

    acc = ct.full((TILE,), 0.0, dtype=np.float32)

    for p in range(0, full_pairs):
        tile0 = 2 * p
        tile1 = tile0 + 1

        x0 = ct.load(
            x_flat,
            index=(row * num_tiles + tile0,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.UNDETERMINED,
            latency=1,
            allow_tma=False,
        )
        x1 = ct.load(
            x_flat,
            index=(row * num_tiles + tile1,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.UNDETERMINED,
            latency=1,
            allow_tma=False,
        )
        acc = acc + ct.astype(x0, np.float32) + ct.astype(x1, np.float32)

    if num_tiles != full_pairs * 2:
        last_tile = full_pairs * 2
        x_last = ct.load(
            x_flat,
            index=(row * num_tiles + last_tile,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.UNDETERMINED,
            latency=1,
            allow_tma=False,
        )
        acc = acc + ct.astype(x_last, np.float32)

    total = ct.sum(acc)
    mean = total * (1.0 / N)
    ct.store(output, index=(row,), tile=mean, latency=1, allow_tma=False)


@ct.kernel(occupancy=4)
def _mean_reduction_pair_2d_kernel(
    x,
    output,
    N: ConstInt,
    TILE: ConstInt,
    EVEN_N: ConstBool,
):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    full_pairs = num_tiles // 2

    acc = ct.full((1, TILE), 0.0, dtype=np.float32)

    for p in range(0, full_pairs):
        tile0 = 2 * p
        tile1 = tile0 + 1

        if EVEN_N:
            x0 = ct.load(
                x,
                index=(row, tile0),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.UNDETERMINED,
                latency=1,
                allow_tma=False,
            )
            x1 = ct.load(
                x,
                index=(row, tile1),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.UNDETERMINED,
                latency=1,
                allow_tma=False,
            )
        else:
            x0 = ct.load(
                x,
                index=(row, tile0),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            )
            x1 = ct.load(
                x,
                index=(row, tile1),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            )

        acc = acc + ct.astype(x0, np.float32) + ct.astype(x1, np.float32)

    if num_tiles != full_pairs * 2:
        last_tile = full_pairs * 2
        if EVEN_N:
            x_last = ct.load(
                x,
                index=(row, last_tile),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.UNDETERMINED,
                latency=1,
                allow_tma=False,
            )
        else:
            x_last = ct.load(
                x,
                index=(row, last_tile),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            )
        acc = acc + ct.astype(x_last, np.float32)

    total = ct.sum(acc)
    mean = total * (1.0 / N)
    ct.store(output, index=(row,), tile=mean, latency=1, allow_tma=False)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim < 0:
        dim = dim + x.dim()
    if dim != 1:
        raise NotImplementedError("mean_reduction only supports row-wise dim=1")

    M = x.shape[0]
    N = x.shape[1]
    output = torch.empty((M,), device=x.device, dtype=torch.float32)

    TILE = 2048
    occupancy = 4
    EVEN_N = (N % TILE) == 0
    USE_FLAT = EVEN_N and x.is_contiguous()

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)

    if USE_FLAT:
        x_flat = x.reshape((M * N,))
        ct.launch(
            stream,
            grid,
            _mean_reduction_pair_flat_kernel,
            (x_flat, output, N, TILE),
        )
    else:
        ct.launch(
            stream,
            grid,
            _mean_reduction_pair_2d_kernel,
            (x, output, N, TILE, EVEN_N),
        )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "ALGO": "flat_pair_unroll2" if USE_FLAT else "2d_pair_unroll2",
            "TILE": TILE,
            "PAIR_UNROLL": 2,
            "occupancy": occupancy,
            "EVEN_N": EVEN_N,
            "USE_FLAT": USE_FLAT,
            "allow_tma": False,
            "latency": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
