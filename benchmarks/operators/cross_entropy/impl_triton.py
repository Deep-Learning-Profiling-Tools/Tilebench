import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"num_warps": 8, "num_stages": 2}


@triton.jit
def _cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    output_ptr,
    num_classes,
    stride_bn,
    stride_bc,
    BLOCK_CLASSES: tl.constexpr,
):
    pid = tl.program_id(0)
    cls_offsets = tl.arange(0, BLOCK_CLASSES)
    cls_mask = cls_offsets < num_classes

    row_ptr = logits_ptr + pid * stride_bn + cls_offsets * stride_bc
    logits = tl.load(row_ptr, mask=cls_mask, other=-float("inf")).to(tl.float32)

    row_max = tl.max(logits, axis=0)
    exp_shifted = tl.exp(logits - row_max)
    row_sum = tl.sum(exp_shifted, axis=0)

    target_cls = tl.load(targets_ptr + pid).to(tl.int32)
    target_ok = (target_cls >= 0) & (target_cls < num_classes)
    target_ptr = logits_ptr + pid * stride_bn + target_cls * stride_bc
    target_logit = tl.load(target_ptr, mask=target_ok, other=-float("inf")).to(
        tl.float32
    )

    loss = -(target_logit - row_max - tl.log(row_sum))
    tl.store(output_ptr + pid, loss)


def run(
    logits: torch.Tensor,
    targets: torch.Tensor,
    block_size: int = 1024,
    autotune: bool = False,
) -> torch.Tensor:
    batch_size, num_classes = logits.shape
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)
    grid = (batch_size,)
    block_classes = triton.next_power_of_2(num_classes)
    cfg = _DEFAULT_CONFIG
    _cross_entropy_kernel[grid](
        logits,
        targets,
        output,
        num_classes,
        logits.stride(0),
        logits.stride(1),
        BLOCK_CLASSES=block_classes,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return output


def get_last_config() -> dict | None:
    return None
