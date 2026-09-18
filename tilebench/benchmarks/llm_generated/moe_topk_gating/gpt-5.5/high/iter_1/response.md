```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _moe_topk2_kernel(
    logits_ptr,
    weights_ptr,
    indices_ptr,
    M,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)

    offs_m = tl.arange(0, BLOCK_M)
    offs_e = tl.arange(0, BLOCK_E)

    n_row_blocks = tl.cdiv(M, BLOCK_M)

    for rb in tl.range(pid, n_row_blocks, step, num_stages=LOOP_STAGES):
        rows = rb * BLOCK_M + offs_m
        row_mask = rows < M

        vals = tl.load(
            logits_ptr + rows[:, None] * E + offs_e[None, :],
            mask=row_mask[:, None] & (offs_e[None, :] < E),
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)

        # First iterative max. Reference writes this max into output column 1.
        val1, idx1 = tl.max(
            vals,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )

        # Mask exactly the selected first-occurrence max, matching scatter_.
        vals2 = tl.where(offs_e[None, :] == idx1[:, None], -float("inf"), vals)

        # Second iterative max. Reference writes this into output column 0.
        val0, idx0 = tl.max(
            vals2,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )

        e = tl.exp(val0 - val1)
        den = e + 1.0
        w0 = e / den
        w1 = 1.0 / den

        tl.store(weights_ptr + rows * K + 0, w0, mask=row_mask)
        tl.store(weights_ptr + rows * K + 1, w1, mask=row_mask)

        tl.store(indices_ptr + rows * K + 0, idx0, mask=row_mask)
        tl.store(indices_ptr + rows * K + 1, idx1, mask=row_mask)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    topk_weights = torch.empty((M, k), device=logits.device, dtype=logits.dtype)
    topk_indices = torch.empty((M, k), device=logits.device, dtype=torch.int32)

    BLOCK_M = 8
    BLOCK_E = 128
    ROW_BLOCKS = 1024
    LOOP_STAGES = 3
    num_warps = 8
    num_stages = 3

    num_row_blocks = triton.cdiv(M, BLOCK_M)
    grid = (min(num_row_blocks, ROW_BLOCKS),)

    _moe_topk2_kernel[grid](
        logits,
        topk_weights,
        topk_indices,
        M,
        E=E,
        K=k,
        BLOCK_M=BLOCK_M,
        BLOCK_E=BLOCK_E,
        LOOP_STAGES=LOOP_STAGES,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_E": BLOCK_E,
            "ROW_BLOCKS": ROW_BLOCKS,
            "LOOP_STAGES": LOOP_STAGES,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "topk_supported": 2,
        }
    )
    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _argmax_first_combine(a_val, a_idx, b_val, b_idx):
    take_a = (a_val > b_val) | ((a_val == b_val) & (a_idx <= b_idx))
    return ct.where(take_a, a_val, b_val), ct.where(take_a, a_idx, b_idx)


@ct.kernel(occupancy=8)
def _moe_topk2_kernel(logits, weights, indices, ROWS: ConstInt, TILE_E: ConstInt):
    bid = ct.bid(0)

    vals = ct.astype(
        ct.load(
            logits,
            index=(bid, 0),
            shape=(ROWS, TILE_E),
            padding_mode=ct.PaddingMode.NEG_INF,
            allow_tma=False,
            latency=1,
        ),
        np.float32,
    )

    offs = ct.arange(TILE_E, dtype=np.int32)[None, :]
    idxs = ct.broadcast_to(offs, (ROWS, TILE_E))

    # First max, first-occurrence tie break. Stored in output column 1.
    val1, idx1 = ct.reduce(
        (vals, idxs),
        axis=1,
        func=_argmax_first_combine,
        identity=(-np.inf, 2147483647),
        keepdims=True,
    )

    vals2 = ct.where(idxs == idx1, -np.inf, vals)

    # Second max after masking first selected expert. Stored in output column 0.
    val0, idx0 = ct.reduce(
        (vals2, idxs),
        axis=1,
        func=_argmax_first_combine,
        identity=(-np.inf, 2147483647),
        keepdims=True,
    )

    e = ct.exp(val0 - val1)
    den = e + 1.0
    w0 = e / den
    w1 = 1.0 / den

    ct.store(weights, index=(bid, 0), tile=ct.astype(w0, logits.dtype), allow_tma=False)
    ct.store(weights, index=(bid, 1), tile=ct.astype(w1, logits.dtype), allow_tma=False)

    ct.store(indices, index=(bid, 0), tile=ct.astype(idx0, np.int32), allow_tma=False)
    ct.store(indices, index=(bid, 1), tile=ct.astype(idx1, np.int32), allow_tma=False)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    topk_weights = torch.empty((M, k), device=logits.device, dtype=logits.dtype)
    topk_indices = torch.empty((M, k), device=logits.device, dtype=torch.int32)

    ROWS = 4
    TILE_E = 128
    occupancy = 8

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, ROWS), 1, 1)
    ct.launch(stream, grid, _moe_topk2_kernel, (logits, topk_weights, topk_indices, ROWS, TILE_E))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "ROWS": ROWS,
            "TILE_E": TILE_E,
            "occupancy": occupancy,
            "allow_tma": False,
            "topk_supported": 2,
        }
    )
    return (topk_weights, topk_indices)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
