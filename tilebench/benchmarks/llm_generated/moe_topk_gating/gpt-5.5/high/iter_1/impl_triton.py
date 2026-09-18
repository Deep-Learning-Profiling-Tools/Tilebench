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
