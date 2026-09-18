import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _kl_kernel(log_y_pred_ptr, y_true_ptr, out_ptr,
               n_cols, stride_row,
               BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * stride_row
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        lp = tl.load(log_y_pred_ptr + base + cols, mask=mask, other=0.0)
        yt = tl.load(y_true_ptr + base + cols, mask=mask, other=0.0)
        pos = yt > 0.0
        log_yt = tl.log(tl.where(pos, yt, 1.0))
        term = tl.where(pos, yt * (log_yt - lp), 0.0)
        acc += term
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, total)


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor):
    assert log_y_pred.shape == y_true.shape
    assert log_y_pred.is_contiguous() and y_true.is_contiguous()
    rows, cols = log_y_pred.shape
    output = torch.empty((rows,), device=log_y_pred.device, dtype=torch.float32)

    BLOCK_N = 2048
    num_warps = 8
    num_stages = 4

    grid = (rows,)
    _kl_kernel[grid](
        log_y_pred, y_true, output,
        cols, log_y_pred.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N, "num_warps": num_warps, "num_stages": num_stages})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
