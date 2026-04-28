"""Triton dequantize_rowwise (matches bitsandbytes' kernel of the same name).

One program per row. BLOCK_SIZE = cols (compile-time constant), so the
kernel is shape-specialised — recompiled per cols. P2 = next_pow2(cols)
gives the tl.arange size; cols < P2 is masked off. We restrict cols to
powers of 2 in case_grid so P2 == BLOCK_SIZE and the mask is a no-op.

Autotune knob: num_warps. BLOCK_SIZE / P2 are dictated by cols and not
sweepable.
"""
import torch
import triton
import triton.language as tl


_DEFAULT_CONFIG = {"num_warps": 4}


@triton.jit
def _dequantize_rowwise_kernel(
    x_ptr, state_x, output_ptr,
    inv_127, n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)
    max_val = tl.load(state_x + pid)
    output = max_val * x * inv_127
    tl.store(output_ptr + offsets, output, mask=row_mask)


_dequantize_rowwise_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw)
        for nw in [2, 4, 8, 16]
    ],
    key=["BLOCK_SIZE"],
)(_dequantize_rowwise_kernel)


def run(x: torch.Tensor, state_x: torch.Tensor,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    rows, cols = x.shape
    output = torch.empty(rows, cols, device=x.device, dtype=torch.float16)
    n_elements = output.numel()
    P2 = triton.next_power_of_2(cols)
    grid = (rows,)

    if autotune:
        _dequantize_rowwise_kernel_autotuned[grid](
            x, state_x, output, 1.0 / 127.0, n_elements,
            BLOCK_SIZE=cols, P2=P2,
        )
    else:
        _dequantize_rowwise_kernel[grid](
            x, state_x, output, 1.0 / 127.0, n_elements,
            BLOCK_SIZE=cols, P2=P2,
            num_warps=_DEFAULT_CONFIG["num_warps"],
        )
    return output


def get_last_config() -> dict | None:
    cfg = getattr(_dequantize_rowwise_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"num_warps": cfg.num_warps}
