import torch
import torch.nn.functional as F


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_rows: int, input_cols: int,
        kernel_rows: int, kernel_cols: int, **kwargs):
    """
    Reference single-channel VALID 2D correlation (no padding, no channels).

    output[r, c] = sum_{i, j} kernel[i, j] * input[r + i, c + j]

    input:  [input_rows, input_cols]
    kernel: [kernel_rows, kernel_cols]
    output: [input_rows - kernel_rows + 1, input_cols - kernel_cols + 1]

    F.conv2d is cross-correlation (no kernel flip), matching the Triton stencil.
    """
    x = input.float().reshape(1, 1, input_rows, input_cols)
    w = kernel.float().reshape(1, 1, kernel_rows, kernel_cols)
    result = F.conv2d(x, w).reshape(
        input_rows - kernel_rows + 1, input_cols - kernel_cols + 1
    )
    return result.to(input.dtype)
