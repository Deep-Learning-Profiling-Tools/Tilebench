import torch


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    if input.device.type == "xla":
        up, down = input[0:rows - 2, 1:cols - 1], input[2:rows, 1:cols - 1]
        left, right = input[1:rows - 1, 0:cols - 2], input[1:rows - 1, 2:cols]
        if input.dtype in (torch.float16, torch.bfloat16):
            s = (up.float() + down.float()).to(input.dtype)
            s = (s.float() + left.float()).to(input.dtype)
            s = (s.float() + right.float()).to(input.dtype)
        else:
            s = up + down + left + right
        inner = 0.25 * s
        mid = torch.cat([input[1:rows - 1, 0:1], inner, input[1:rows - 1, cols - 1:cols]], dim=1)
        return torch.cat([input[0:1], mid, input[rows - 1:rows]], dim=0)

    output = input.clone()
    output[1:rows - 1, 1:cols - 1] = 0.25 * (
        input[0:rows - 2, 1:cols - 1]
        + input[2:rows,   1:cols - 1]
        + input[1:rows - 1, 0:cols - 2]
        + input[1:rows - 1, 2:cols]
    )
    return output
