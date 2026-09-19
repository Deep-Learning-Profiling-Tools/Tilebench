```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    row_offsets_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_bases = row_offsets_m * BLOCK_N

    target = tl.load(
        targets_ptr + row_offsets_m,
        eviction_policy="evict_first",
    ).to(tl.int32)

    vals = tl.load(
        logits_ptr + row_bases[:, None] + cols[None, :],
        eviction_policy="evict_first",
    ).to(tl.float32)

    target_logit = tl.load(
        logits_ptr + row_bases + target,
        eviction_policy="evict_first",
    ).to(tl.float32)

    max_val = tl.max(vals, axis=1)

    shifted = (vals - max_val[:, None]) * 1.4426950408889634
    exp_vals = tl.exp2(shifted)
    sum_exp = tl.sum(exp_vals, axis=1)

    loss = (max_val - target_logit) + tl.log2(sum_exp) * 0.6931471805599453
    tl.store(output_ptr + row_offsets_m, loss)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size = logits.shape[0]
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)

    BLOCK_M = 4
    BLOCK_N = 512
    num_warps = 4
    num_stages = 2

    grid = (batch_size // BLOCK_M,)
    _cross_entropy_kernel[grid](
        logits,
        targets,
        output,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
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

_LAST_CFG: dict = {}


@ct.kernel
def _cross_entropy_kernel(logits, targets, output, TILE: ConstInt, ROWS: ConstInt):
    bid = ct.bid(0)

    target = ct.astype(
        ct.load(
            targets,
            index=(bid,),
            shape=(ROWS,),
            latency=1,
            allow_tma=False,
        ),
        np.int32,
    )

    vals = ct.load(
        logits,
        index=(bid, 0),
        shape=(ROWS, TILE),
        latency=1,
        allow_tma=False,
    )
    vals_f = ct.astype(vals, np.float32)

    row_ids = bid * ROWS + ct.arange(ROWS, dtype=np.int32)
    target_logit = ct.astype(
        ct.gather(
            logits,
            (row_ids, target),
            padding_value=0.0,
            check_bounds=False,
            latency=1,
        ),
        np.float32,
    )

    max_val = ct.max(vals_f, axis=1, keepdims=False)

    shifted = (vals_f - max_val[:, None]) * 1.4426950408889634
    exp_vals = ct.exp2(shifted, flush_to_zero=True)
    sum_exp = ct.sum(exp_vals, axis=1, keepdims=False)

    loss = (max_val - target_logit) + ct.log2(sum_exp) * 0.6931471805599453
    ct.store(
        output,
        index=(bid,),
        tile=ct.astype(loss, output.dtype),
        allow_tma=False,
    )


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size = logits.shape[0]
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)
    stream = torch.cuda.current_stream()

    TILE = 512
    ROWS = 4
    occupancy = 4

    grid = (batch_size // ROWS, 1, 1)
    kernel = _cross_entropy_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (logits, targets, output, TILE, ROWS))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "ROWS": ROWS, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
