import torch


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    """
    2D 5-point Jacobi stencil reference. Interior cells receive the average
    of their four cardinal neighbors; boundary cells are copied from input.

    Implemented out-of-place to match the problem spec ("read exclusively
    from input, write exclusively to output") and to match the Triton /
    cuTile kernels — the LeetGPU torch solve mutates ``input`` in place,
    which would make the engine's warmup / repeat loop produce wrong
    results on the second call.
    """
    # Native-dtype arithmetic to match the Triton / cuTile kernels, which
    # load and average in the input dtype — the old input.float() round-trip
    # materialised an fp32 copy of the whole grid (plus the cast back) and
    # made torch's fp16 path both slower and semantically different.
    output = input.clone()  # boundaries remain the same as input
    output[1:rows - 1, 1:cols - 1] = 0.25 * (
        input[0:rows - 2, 1:cols - 1]
        + input[2:rows,   1:cols - 1]
        + input[1:rows - 1, 0:cols - 2]
        + input[1:rows - 1, 2:cols]
    )
    return output
