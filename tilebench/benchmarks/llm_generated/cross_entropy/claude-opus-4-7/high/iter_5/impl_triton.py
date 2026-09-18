import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _ce_kernel(logits_ptr, targets_ptr, out_ptr,
               batch_size, n_cols, stride_row,
               ROWS_PER_BLOCK: tl.constexpr,
               BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = rows < batch_size

    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < n_cols

    offs = rows[:, None] * stride_row + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    x = tl.load(logits_ptr + offs, mask=mask, other=-float('inf'),
                eviction_policy="evict_first").to(tl.float32)

    m = tl.max(x, axis=1)
    z = tl.exp(x - m[:, None])
    s = tl.sum(z, axis=1)
    log_s = tl.log(s)

    targets = tl.load(targets_ptr + rows, mask=row_mask, other=0,
                      eviction_policy="evict_first").to(tl.int32)

    # Extract target logits from already-loaded x — eliminates a second
    # (non-coalesced) gather load. target is guaranteed to be in-range.
    target_mask = cols[None, :] == targets[:, None]
    target_logit = tl.sum(tl.where(target_mask, x, 0.0), axis=1)

    loss = -(target_logit - m) + log_s
    tl.store(out_ptr + rows, loss.to(out_ptr.dtype.element_ty), mask=row_mask)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size, num_classes = logits.shape
    output = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)

    BLOCK_N = triton.next_power_of_2(num_classes)
    if BLOCK_N < 256:
        BLOCK_N = 256

    ROWS_PER_BLOCK = 4
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(batch_size, ROWS_PER_BLOCK),)
    _ce_kernel[grid](
        logits, targets, output,
        batch_size, num_classes, logits.stride(0),
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"ROWS_PER_BLOCK": ROWS_PER_BLOCK,
                      "BLOCK_N": BLOCK_N,
                      "num_warps": num_warps,
                      "num_stages": num_stages,
                      "opt": "mask_extract_target"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
