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
    E,
    stride_lm,
    stride_le,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
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

        vals = tl.load(
            logits_ptr + rows[:, None] * stride_lm + offs_e[None, :] * stride_le,
            mask=(rows[:, None] < M) & (offs_e[None, :] < E),
            other=-float("inf"),
            eviction_policy="evict_first",
        ).to(tl.float32)

        # First max: reference stores it in output column 1 for k=2.
        val1, idx1 = tl.max(
            vals,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )

        # Mask only the chosen first-occurrence max, matching torch scatter_ behavior.
        vals2 = tl.where(offs_e[None, :] == idx1[:, None], -float("inf"), vals)

        # Second max: reference stores it in output column 0 for k=2.
        val0, idx0 = tl.max(
            vals2,
            axis=1,
            return_indices=True,
            return_indices_tie_break_left=True,
        )

        # Softmax([second_max, max]) using one exp because val1 >= val0 for finite inputs.
        e = tl.exp(val0 - val1)
        den = e + 1.0
        w0 = e / den
        w1 = 1.0 / den

        row_mask = rows < M

        tl.store(weights_ptr + rows * stride_wm + 0 * stride_wk, w0, mask=row_mask)
        tl.store(weights_ptr + rows * stride_wm + 1 * stride_wk, w1, mask=row_mask)

        tl.store(indices_ptr + rows * stride_im + 0 * stride_ik, idx0, mask=row_mask)
        tl.store(indices_ptr + rows * stride_im + 1 * stride_ik, idx1, mask=row_mask)


def run(logits: torch.Tensor, M: int, E: int, k: int, **kwargs):
    topk_weights = torch.empty((M, k), device=logits.device, dtype=logits.dtype)
    topk_indices = torch.empty((M, k), device=logits.device, dtype=torch.int32)

    BLOCK_M = 2
    BLOCK_E = 256
    ROW_BLOCKS = 1024
    LOOP_STAGES = 2
    num_warps = 4
    num_stages = 2

    num_row_blocks = triton.cdiv(M, BLOCK_M)
    grid = (min(num_row_blocks, ROW_BLOCKS),)

    _moe_topk2_kernel[grid](
        logits,
        topk_weights,
        topk_indices,
        M,
        E,
        logits.stride(0),
        logits.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
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
