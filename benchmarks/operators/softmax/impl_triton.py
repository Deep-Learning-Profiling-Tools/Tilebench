import torch
import triton
import triton.language as tl
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID
    row_idx = tl.program_id(axis=0)

    # Compute the memory offsets for this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Load the row into SRAM
    row = tl.load(row_start_ptr + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_cols, other=-float('inf'))

    # Compute max for numerical stability
    row_max = tl.max(row, axis=0)

    # Subtract max from row and exponentiate
    numerator = tl.exp(row - row_max)

    # Compute sum for normalization
    denominator = tl.sum(numerator, axis=0)

    # Normalize
    softmax_output = numerator / denominator

    # Store the output
    tl.store(out_row_start_ptr + tl.arange(0, BLOCK_SIZE), softmax_output, mask=tl.arange(0, BLOCK_SIZE) < n_cols)


_softmax_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [4, 8, 16]
        for ns in [2, 3, 4]
    ],
    key=["n_cols"],
)(softmax_kernel)


def run(x: torch.Tensor, block_size: int, autotune: bool = False):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    if block_size < n_cols:
        block_size = n_cols

    block_size = triton.next_power_of_2(block_size)

    grid = (n_rows,)

    if autotune:
        _softmax_kernel_autotuned[grid](
            output, x,
            x.stride(0), output.stride(0),
            n_cols,
            BLOCK_SIZE=block_size,
        )
    else:
        softmax_kernel[grid](
            output, x,
            x.stride(0), output.stride(0),
            n_cols,
            BLOCK_SIZE=block_size,
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_softmax_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
