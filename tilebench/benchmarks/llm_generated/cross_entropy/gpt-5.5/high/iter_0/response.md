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
    n_cols,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    row_base = row * n_cols
    vals = tl.load(
        logits_ptr + row_base + cols,
        mask=mask,
        other=-float("inf"),
        eviction_policy="evict_first",
    ).to(tl.float32)

    max_val = tl.max(vals, axis=0)

    # exp(x) = exp2(x * log2(e)); log(x) = log2(x) * ln(2)
    shifted = (vals - max_val) * 1.4426950408889634
    exp_vals = tl.exp2(shifted)
    sum_exp = tl.sum(exp_vals, axis=0)

    target = tl.load(targets_ptr + row)
    target_logit = tl.load(logits_ptr + row_base + target).to(tl.float32)

    loss = (max_val - target_logit) + tl.log2(sum_exp) * 0.6931471805599453
    tl.store(output_ptr + row, loss)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size = logits.shape[0]
    n_cols = logits.shape[1]
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)

    BLOCK_N = 512
    num_warps = 4
    num_stages = 2

    grid = (batch_size,)
    _cross_entropy_kernel[grid](
        logits,
        targets,
        output,
        n_cols,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
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
def _cross_entropy_kernel(logits, targets, output, TILE: ConstInt):
    row = ct.bid(0)

    vals = ct.load(
        logits,
        index=(row, 0),
        shape=(1, TILE),
        padding_mode=ct.PaddingMode.NEG_INF,
        latency=1,
        allow_tma=False,
    )
    vals_f = ct.astype(vals, np.float32)

    max_val = ct.max(vals_f)

    shifted = (vals_f - max_val) * 1.4426950408889634
    exp_vals = ct.exp2(shifted, flush_to_zero=True)
    sum_exp = ct.sum(exp_vals)

    target = ct.astype(ct.load(targets, index=(row,), shape=()), np.int32)
    target_logit = ct.astype(
        ct.gather(logits, (row, target), padding_value=0.0, latency=1),
        np.float32,
    )

    loss = (max_val - target_logit) + ct.log2(sum_exp) * 0.6931471805599453
    ct.store(
        output,
        index=(row,),
        tile=ct.astype(loss, output.dtype),
        allow_tma=False,
    )


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size = logits.shape[0]
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)
    stream = torch.cuda.current_stream()

    TILE = 512
    occupancy = 8

    grid = (batch_size, 1, 1)
    kernel = _cross_entropy_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (logits, targets, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
