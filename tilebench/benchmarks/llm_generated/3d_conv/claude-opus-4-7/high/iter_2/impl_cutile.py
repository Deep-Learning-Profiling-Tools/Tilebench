import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _conv3d_kernel(x_flat, k_3d, out_flat,
                   IR: ConstInt, IC: ConstInt,
                   OR: ConstInt, OC: ConstInt,
                   KD: ConstInt, KR: ConstInt, KC: ConstInt,
                   BLOCK_R: ConstInt, BLOCK_C: ConstInt):
    pid_drb = ct.bid(0)
    pid_c = ct.bid(1)
    num_row_blocks = ct.cdiv(OR, BLOCK_R)
    out_d = pid_drb // num_row_blocks
    rb = pid_drb % num_row_blocks

    r_offs = rb * BLOCK_R + ct.arange(BLOCK_R, dtype=np.int32)   # [BLOCK_R]
    c_offs = pid_c * BLOCK_C + ct.arange(BLOCK_C, dtype=np.int32)  # [BLOCK_C]
    r_valid = r_offs < OR
    c_valid = c_offs < OC

    acc = ct.full((BLOCK_R, BLOCK_C), 0.0, dtype=np.float32)

    for kd in range(KD):
        d_base = (out_d + kd) * IR * IC
        for kr in range(KR):
            in_r = r_offs + kr                                   # [BLOCK_R]
            row_base = d_base + in_r[:, None] * IC               # [BLOCK_R, 1]
            for kc in range(KC):
                k_val_t = ct.load(k_3d, index=(kd, kr, kc), shape=())
                k_val = ct.astype(k_val_t, np.float32)
                x_idx = row_base + (c_offs[None, :] + kc)        # [BLOCK_R, BLOCK_C]
                x_vals = ct.gather(x_flat, x_idx, padding_value=0.0)
                acc = acc + k_val * ct.astype(x_vals, np.float32)

    out_idx = out_d * OR * OC + r_offs[:, None] * OC + c_offs[None, :]
    valid = r_valid[:, None] & c_valid[None, :]
    safe_idx = ct.where(valid, out_idx, -1)
    ct.scatter(out_flat, safe_idx, ct.astype(acc, out_flat.dtype))


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, **kwargs):
    OD = input_depth - kernel_depth + 1
    OR = input_rows - kernel_rows + 1
    OC = input_cols - kernel_cols + 1

    output = torch.empty(OD * OR * OC, dtype=input.dtype, device=input.device)
    k_3d = kernel.view(kernel_depth, kernel_rows, kernel_cols)

    BLOCK_R = 4
    BLOCK_C = 128

    stream = torch.cuda.current_stream()
    grid = (OD * ct.cdiv(OR, BLOCK_R), ct.cdiv(OC, BLOCK_C), 1)

    ct.launch(stream, grid, _conv3d_kernel, (
        input, k_3d, output,
        input_rows, input_cols, OR, OC,
        kernel_depth, kernel_rows, kernel_cols,
        BLOCK_R, BLOCK_C,
    ))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_R": BLOCK_R, "BLOCK_C": BLOCK_C, "occupancy": 2})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
