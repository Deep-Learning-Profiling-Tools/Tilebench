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
    batch_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CLASSES: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    base_row = pid * BLOCK_M
    cols = tl.arange(0, BLOCK_N)

    for r in tl.static_range(0, BLOCK_M):
        row = base_row + r
        valid = row < batch_size
        row_base = row * NUM_CLASSES

        target = tl.load(
            targets_ptr + row,
            mask=valid,
            other=0,
            eviction_policy="evict_first",
        ).to(tl.int32)

        target_logit = tl.load(
            logits_ptr + row_base + target,
            mask=valid,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        vals0 = tl.load(
            logits_ptr + row_base + cols,
            mask=valid & (cols < NUM_CLASSES),
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)
        exp0 = tl.exp2(vals0 * 1.4426950408889634)
        sum_exp = tl.sum(exp0, axis=0)

        for c in tl.static_range(1, NUM_CHUNKS):
            offs = c * BLOCK_N + cols
            vals = tl.load(
                logits_ptr + row_base + offs,
                mask=valid & (offs < NUM_CLASSES),
                other=-float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            exp_vals = tl.exp2(vals * 1.4426950408889634)
            sum_exp += tl.sum(exp_vals, axis=0)

        loss = tl.log2(sum_exp) * 0.6931471805599453 - target_logit
        tl.store(output_ptr + row, loss, mask=valid)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)

    BLOCK_M = 2
    BLOCK_N = 256
    NUM_CLASSES = num_classes
    NUM_CHUNKS = triton.cdiv(num_classes, BLOCK_N)
    num_warps = 1
    num_stages = 2

    grid = (triton.cdiv(batch_size, BLOCK_M),)
    _cross_entropy_kernel[grid](
        logits,
        targets,
        output,
        batch_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        NUM_CLASSES=NUM_CLASSES,
        NUM_CHUNKS=NUM_CHUNKS,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "NUM_CLASSES": NUM_CLASSES,
            "NUM_CHUNKS": NUM_CHUNKS,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "serial_rows": True,
            "direct_logsumexp": True,
            "chunked_logsumexp": True,
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
def _cross_entropy_kernel(
    logits,
    targets,
    output,
    TILE: ConstInt,
    ROWS: ConstInt,
    NUM_CHUNKS: ConstInt,
):
    bid = ct.bid(0)

    for r in range(0, ROWS):
        row = bid * ROWS + r

        vals0 = ct.load(
            logits,
            index=(row, 0),
            shape=(1, TILE),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
            allow_tma=False,
        )
        vals0_f = ct.astype(vals0, np.float32)
        exp0 = ct.exp2(vals0_f * 1.4426950408889634, flush_to_zero=True)
        sum_exp = ct.sum(exp0)

        for c in range(1, NUM_CHUNKS):
            vals = ct.load(
                logits,
                index=(row, c),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=1,
                allow_tma=False,
            )
            vals_f = ct.astype(vals, np.float32)
            exp_vals = ct.exp2(vals_f * 1.4426950408889634, flush_to_zero=True)
            sum_exp = sum_exp + ct.sum(exp_vals)

        target = ct.astype(
            ct.load(
                targets,
                index=(row,),
                shape=(),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            ),
            np.int32,
        )
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

        loss = ct.log2(sum_exp) * 0.6931471805599453 - target_logit
        ct.store(
            output,
            index=(row,),
            tile=ct.astype(loss, output.dtype),
            allow_tma=False,
        )


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size = logits.shape[0]
    num_classes = logits.shape[1]
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)
    stream = torch.cuda.current_stream()

    TILE = 256
    ROWS = 4
    NUM_CHUNKS = ct.cdiv(num_classes, TILE)
    occupancy = 8

    grid = (ct.cdiv(batch_size, ROWS), 1, 1)
    kernel = _cross_entropy_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (logits, targets, output, TILE, ROWS, NUM_CHUNKS))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "ROWS": ROWS,
            "NUM_CHUNKS": NUM_CHUNKS,
            "occupancy": occupancy,
            "serial_rows": True,
            "direct_logsumexp": True,
            "chunked_logsumexp": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
