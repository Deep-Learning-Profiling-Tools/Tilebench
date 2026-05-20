import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

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

    logits_desc = tl.make_tensor_descriptor(
        logits_ptr + pid * stride_bn,
        shape=[num_classes, 1],
        strides=[stride_bc, 1],
        block_shape=[BLOCK_CLASSES, 1],
    )
    logits = logits_desc.load([0, 0])[:, 0].to(tl.float32)
    logits = tl.where(cls_mask, logits, -float("inf"))

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


_cross_entropy_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw)
        for nw in [1, 2, 4, 8]
    ],
    key=["num_classes"],
)(_cross_entropy_kernel)


def run(
    logits: torch.Tensor,
    targets: torch.Tensor,
    block_size: int = 1024,
    autotune: bool = False,
) -> torch.Tensor:
    ensure_tma_available()
    logits = logits.contiguous()
    batch_size, num_classes = logits.shape
    output = torch.empty((batch_size,), device=logits.device, dtype=logits.dtype)
    grid = (batch_size,)
    block_classes = triton.next_power_of_2(num_classes)
    if autotune:
        _cross_entropy_kernel_autotuned[grid](
            logits, targets, output,
            num_classes,
            logits.stride(0), logits.stride(1),
            BLOCK_CLASSES=block_classes,
        )
    else:
        cfg = _DEFAULT_CONFIG
        _cross_entropy_kernel[grid](
            logits, targets, output,
            num_classes,
            logits.stride(0), logits.stride(1),
            BLOCK_CLASSES=block_classes,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return output


def get_last_config() -> dict | None:
    cfg = getattr(_cross_entropy_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"num_warps": cfg.num_warps}
