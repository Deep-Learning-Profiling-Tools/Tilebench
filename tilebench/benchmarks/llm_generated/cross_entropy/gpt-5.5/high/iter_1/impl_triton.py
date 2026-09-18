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
