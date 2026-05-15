"""Triton KL divergence forward (TritonBench / Liger-Kernel style).

Convention (PyTorch F.kl_div with log_target=False):
    log_y_pred  is log-probabilities (log_softmax output)
    y_true      is plain probabilities (softmax output)
    loss[b] = sum_s y_true[b,s] * (log(y_true[b,s]) - log_y_pred[b,s])

One program per row. The inner tile loop iterates BLOCK_SIZE-sized chunks
across the cols axis -- this decouples BLOCK_SIZE from cols, so cuTile-
style "load whole row in one tile" register pressure is avoided for
cols=16384 and BLOCK_SIZE becomes a real autotune knob.
"""
import torch
import triton
import triton.language as tl


_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 3}


@triton.jit
def _kl_divergence_kernel(
    log_y_pred_ptr, log_y_pred_stride,
    y_true_ptr, y_true_stride,
    loss_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    log_y_pred_ptr += pid * log_y_pred_stride
    y_true_ptr += pid * y_true_stride

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_start in range(0, n_cols, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        log_y_pred = tl.load(log_y_pred_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y_true = tl.load(y_true_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        # Where y_true == 0 (true zero or OOB padding), the KL term is 0
        # by convention. Guard the log explicitly so 0 * log(0) -> 0
        # cleanly without relying on NaN-suppression in tl.where.
        safe_log = tl.where(y_true > 0.0, tl.log(y_true), 0.0)
        loss_chunk = y_true * (safe_log - log_y_pred)
        acc += tl.where(mask, loss_chunk, 0.0)

    row_sum = tl.sum(acc, axis=0)
    tl.store(loss_ptr + pid, row_sum)


_kl_divergence_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [512, 1024, 2048]
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
        if bs >= nw * 32
    ],
    key=["n_cols"],
    warmup=3,
    rep=10,
)(_kl_divergence_kernel)


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    rows, cols = log_y_pred.shape
    loss = torch.empty(rows, device=log_y_pred.device, dtype=torch.float32)
    grid = (rows,)

    if autotune:
        _kl_divergence_kernel_autotuned[grid](
            log_y_pred, log_y_pred.stride(0),
            y_true, y_true.stride(0),
            loss,
            cols,
        )
    else:
        cfg = _DEFAULT_CONFIG
        _kl_divergence_kernel[grid](
            log_y_pred, log_y_pred.stride(0),
            y_true, y_true.stride(0),
            loss,
            cols,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return loss


def get_last_config() -> dict | None:
    cfg = getattr(_kl_divergence_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps":  cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
