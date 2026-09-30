import torch


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):


    output = input.clone()
    output[1:rows - 1, 1:cols - 1] = 0.25 * (
        input[0:rows - 2, 1:cols - 1]
        + input[2:rows,   1:cols - 1]
        + input[1:rows - 1, 0:cols - 2]
        + input[1:rows - 1, 2:cols]
    )
    return output
