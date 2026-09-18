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

    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_bases = rows * BLOCK_N

    target = tl.load(
        targets_ptr + rows,
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
    tl.store(output_ptr + rows, loss)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size = logits.shape[0]
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)

    BLOCK_M = 8
    BLOCK_N = 512
    num_warps = 8
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
