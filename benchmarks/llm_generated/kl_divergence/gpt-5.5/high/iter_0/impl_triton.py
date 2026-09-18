import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _kl_divergence_kernel(
    log_y_pred_ptr,
    y_true_ptr,
    output_ptr,
    COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * COLS
    offs = tl.arange(0, BLOCK_N)
    acc = tl.full((), 0.0, dtype=tl.float32)

    for start in range(0, COLS, BLOCK_N):
        cols = start + offs

        if start + BLOCK_N <= COLS:
            y = tl.load(
                y_true_ptr + base + cols,
                eviction_policy="evict_first",
            ).to(tl.float32)
            log_pred = tl.load(
                log_y_pred_ptr + base + cols,
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            mask = cols < COLS
            y = tl.load(
                y_true_ptr + base + cols,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            log_pred = tl.load(
                log_y_pred_ptr + base + cols,
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)

        y_log_arg = tl.where(y > 0.0, y, 1.0)
        term = y * (tl.log(y_log_arg) - log_pred)
        acc += tl.sum(term, axis=0)

    tl.store(output_ptr + row, acc)


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    log_y_pred = log_y_pred.contiguous()
    y_true = y_true.contiguous()

    rows = log_y_pred.shape[0]
    cols = log_y_pred.shape[1]
    output = torch.empty((rows,), device=log_y_pred.device, dtype=torch.float32)

    BLOCK_N = 2048
    num_warps = 8
    num_stages = 2

    grid = (rows,)
    _kl_divergence_kernel[grid](
        log_y_pred,
        y_true,
        output,
        COLS=cols,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_N": BLOCK_N,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
