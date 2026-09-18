import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _ce_kernel(logits_ptr, targets_ptr, out_ptr,
               n_cols, stride_row,
               BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    row_ptr = logits_ptr + row * stride_row
    x = tl.load(row_ptr + cols, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    z = tl.exp(x - m)
    s = tl.sum(z, axis=0)
    log_s = tl.log(s)

    target = tl.load(targets_ptr + row)
    target_logit = tl.load(row_ptr + target).to(tl.float32)
    loss = -(target_logit - m) + log_s
    tl.store(out_ptr + row, loss.to(out_ptr.dtype.element_ty))


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size, num_classes = logits.shape
    output = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)

    BLOCK_N = triton.next_power_of_2(num_classes)
    if BLOCK_N < 256:
        BLOCK_N = 256
    num_warps = 4
    num_stages = 1

    grid = (batch_size,)
    _ce_kernel[grid](
        logits, targets, output,
        num_classes, logits.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N,
                      "num_warps": num_warps,
                      "num_stages": num_stages})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
