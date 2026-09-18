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
    base_row = pid * BLOCK_M
    cols = tl.arange(0, BLOCK_N)

    for r in tl.static_range(0, BLOCK_M):
        row = base_row + r
        row_base = row * BLOCK_N

        vals = tl.load(
            logits_ptr + row_base + cols,
            eviction_policy="evict_first",
        ).to(tl.float32)

        target = tl.load(
            targets_ptr + row,
            eviction_policy="evict_first",
        ).to(tl.int32)

        target_logit = tl.load(
            logits_ptr + row_base + target,
            eviction_policy="evict_first",
        ).to(tl.float32)

        max_val = tl.max(vals, axis=0)

        shifted = (vals - max_val) * 1.4426950408889634
        exp_vals = tl.exp2(shifted)
        sum_exp = tl.sum(exp_vals, axis=0)

        loss = (max_val - target_logit) + tl.log2(sum_exp) * 0.6931471805599453
        tl.store(output_ptr + row, loss)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size = logits.shape[0]
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)

    BLOCK_M = 2
    BLOCK_N = 512
    num_warps = 1
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
            "serial_rows": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
