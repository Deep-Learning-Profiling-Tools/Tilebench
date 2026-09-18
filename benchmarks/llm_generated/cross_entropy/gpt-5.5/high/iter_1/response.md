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
    n_rows,
    BLOCK_N: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    row_step = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N)

    for row in tl.range(pid, n_rows, row_step, num_stages=LOOP_STAGES):
        row_base = row * BLOCK_N

        vals = tl.load(
            logits_ptr + row_base + cols,
            eviction_policy="evict_first",
        ).to(tl.float32)

        max_val = tl.max(vals, axis=0)

        shifted = (vals - max_val) * 1.4426950408889634
        exp_vals = tl.exp2(shifted)
        sum_exp = tl.sum(exp_vals, axis=0)

        target = tl.load(targets_ptr + row)
        target_logit = tl.load(logits_ptr + row_base + target).to(tl.float32)

        loss = (max_val - target_logit) + tl.log2(sum_exp) * 0.6931471805599453
        tl.store(output_ptr + row, loss)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size = logits.shape[0]
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)

    BLOCK_N = 512
    MAX_CTAS = 2048
    LOOP_STAGES = 2
    num_warps = 4
    num_stages = 2

    grid = (min(batch_size, MAX_CTAS),)
    _cross_entropy_kernel[grid](
        logits,
        targets,
        output,
        batch_size,
        BLOCK_N=BLOCK_N,
        LOOP_STAGES=LOOP_STAGES,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_N": BLOCK_N,
            "MAX_CTAS": MAX_CTAS,
            "LOOP_STAGES": LOOP_STAGES,
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


@ct.kernel(occupancy=8)
def _cross_entropy_kernel(logits, targets, output, TILE: ConstInt):
    row = ct.bid(0)

    vals = ct.load(
        logits,
        index=(row, 0),
        shape=(1, TILE),
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
        ct.gather(
            logits,
            (row, target),
            padding_value=0.0,
            check_bounds=False,
            latency=1,
        ),
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
    ct.launch(stream, grid, _cross_entropy_kernel, (logits, targets, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
