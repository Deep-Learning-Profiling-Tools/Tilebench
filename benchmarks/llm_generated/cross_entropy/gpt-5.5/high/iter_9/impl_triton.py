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
