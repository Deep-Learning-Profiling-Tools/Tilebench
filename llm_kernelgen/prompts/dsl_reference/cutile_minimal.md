# cuTile (cuda.tile) DSL Minimal Reference

This reference covers the cuTile 1.3 API patterns used in TileBench kernels.
The canonical import is `import cuda.tile as ct`.

---

## 1. Kernel definition and type aliases

```python
import cuda.tile as ct

ConstInt = ct.Constant[int]   # compile-time integer constant

@ct.kernel
def my_kernel(
    X,                   # input tensor (passed as raw pointer-like handle)
    Out,                 # output tensor
    N: ConstInt,         # compile-time constant
):
    ...
```

---

## 2. Block IDs

cuTile uses a block-tiled programming model. Threads-within-a-block are
implicit; you only specify the tile per block.

```python
bid = ct.bid(0)   # block index along axis 0
bid_x = ct.bid(0)
bid_y = ct.bid(1)
bid_z = ct.bid(2)
```

---

## 3. Load / Store

**Load** returns a tile (a local tensor fragment).

```python
tile = ct.load(array, index=(bid, 0), shape=(TILE_SIZE,))
```

- `index`: tuple of tile-space indices (one per dimension of the array).
  Think of each index as pointing to a tile-aligned position.
- `shape`: the tile shape loaded per block.
- `padding_mode`: optional; pads out-of-bounds lanes.
  - `ct.PaddingMode.ZERO`    – fill with 0 (neutral for sum reductions).
  - `ct.PaddingMode.NEG_INF` – fill with −∞ (neutral for max reductions).

```python
# Load row `bid` of a 2-D array, BLOCK_N elements wide.
row = ct.load(X, index=(bid, 0), shape=(1, BLOCK_N),
              padding_mode=ct.PaddingMode.ZERO)
```

**Store** writes a tile back.  Out-of-bound lanes are automatically ignored.

```python
ct.store(Out, index=(bid, 0), tile=result)
```

---

## 4. Elementwise operations

Arithmetic operators work element-wise on tiles:

```python
a = ct.load(X, index=(bid,), shape=(TILE,))
b = ct.load(Y, index=(bid,), shape=(TILE,))
c = a + b          # add
c = a * b          # multiply
c = ct.exp(a)      # exp
c = ct.sqrt(a)     # sqrt
c = ct.maximum(a, b)  # element-wise max
c = ct.where(cond, a, b)  # conditional select
```

---

## 5. Reductions

```python
total = ct.sum(tile, dim=1)   # reduce along dim 1
mx    = ct.max(tile, dim=1)   # max along dim 1
```

For a 1-D tile simply use `dim=0`.

---

## 6. Full (constant) tile

```python
zeros = ct.full((TILE,), 0.0, dtype=ct.float32)
neginf = ct.full((TILE,), -float("inf"), dtype=ct.float32)
```

---

## 7. Launching the kernel

```python
import torch

stream = torch.cuda.current_stream().cuda_stream

grid = (num_blocks_x, num_blocks_y)   # tuple of ints
ct.launch(stream, grid, my_kernel, (X, Out, N))
```

`ct.launch(stream, grid, kernel, args_tuple)` – positional-only.

---

## 8. CutileAutotuner (optional)

TileBench provides `core.cutile_autotune.CutileAutotuner` to cache
`replace_hints` results and autotune outcomes (mirrors `@triton.autotune`).

```python
from types import SimpleNamespace
import cuda.tile as ct
from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

@ct.kernel
def _kernel(X, Out, N: ConstInt):
    ...

_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [1, 2, 4, 8]]
_tuner = CutileAutotuner(_kernel)
_last_config: dict | None = None


def run(x, block_size: int = 1024, autotune: bool = False, **kwargs):
    global _last_config
    n = x.numel()
    out = torch.empty_like(x)
    stream = torch.cuda.current_stream().cuda_stream

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda c: ((n + block_size - 1) // block_size,),
            args_fn=lambda c: (x, out, n),
            hints_fn=lambda c: {"occupancy": c.occupancy},
        )
        _last_config = {"occupancy": cfg.occupancy}
    else:
        cfg = SimpleNamespace(occupancy=1)
        _last_config = None

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    grid = ((n + block_size - 1) // block_size,)
    ct.launch(stream, grid, kernel, (x, out, n))
    return out


def get_last_config() -> dict | None:
    return _last_config
```

---

## 9. Row-wise reduction example (softmax)

```python
@ct.kernel
def _softmax_kernel(X, Out, N_COLS: ConstInt, BLOCK_N: ConstInt):
    bid = ct.bid(0)  # one block per row
    row = ct.load(X, index=(bid, 0), shape=(1, BLOCK_N),
                  padding_mode=ct.PaddingMode.NEG_INF)
    row_max = ct.max(row, dim=1)                 # shape (1,)
    shifted = row - row_max                      # broadcast
    exp_row = ct.exp(shifted)
    row_sum = ct.sum(exp_row, dim=1)             # shape (1,)
    softmax = exp_row / row_sum
    ct.store(Out, index=(bid, 0), tile=softmax)
```

---

## 10. Key semantics summary

| Concept | cuTile behaviour |
|---------|-----------------|
| Tile index | tile-space (not element-space) — each unit = TILE elements |
| Out-of-bound store | silently ignored — no explicit masking needed for stores |
| Out-of-bound load | requires `padding_mode` for safety |
| Thread layout | implicit — you do not manage threads inside a block |
| Compile-time constants | must be `ct.Constant[int]` in kernel signature |
| Autotune cache | use `CutileAutotuner` from `core.cutile_autotune` |
