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
                   BLOCK_C: ConstInt):
    pid_dr = ct.bid(0)
    pid_c = ct.bid(1)
    out_d = pid_dr // OR
    out_r = pid_dr % OR

    c_offs = pid_c * BLOCK_C + ct.arange(BLOCK_C, dtype=np.int32)
    valid = c_offs < OC

    acc = ct.full((BLOCK_C,), 0.0, dtype=np.float32)

    for kd in range(KD):
        d_base = (out_d + kd) * IR * IC
        for kr in range(KR):
            row_base = d_base + (out_r + kr) * IC
            for kc in range(KC):
                k_val_t = ct.load(k_3d, index=(kd, kr, kc), shape=())
                k_val = ct.astype(k_val_t, np.float32)
                x_idx = row_base + c_offs + kc
                x_vals = ct.gather(x_flat, x_idx, padding_value=0.0)
                acc = acc + k_val * ct.astype(x_vals, np.float32)

    out_idx = out_d * OR * OC + out_r * OC + c_offs
    safe_idx = ct.where(valid, out_idx, -1)
    ct.scatter(out_flat, safe_idx, ct.astype(acc, out_flat.dtype))


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, **kwargs):
    OD = input_depth - kernel_depth + 1
    OR = input_rows - kernel_rows + 1
    OC = input_cols - kernel_cols + 1

    output = torch.empty(OD * OR * OC, dtype=input.dtype, device=input.device)
    k_3d = kernel.view(kernel_depth, kernel_rows, kernel_cols)

    BLOCK_C = 512

    stream = torch.cuda.current_stream()
    grid = (OD * OR, ct.cdiv(OC, BLOCK_C), 1)

    ct.launch(stream, grid, _conv3d_kernel, (
        input, k_3d, output,
        input_rows, input_cols, OR, OC,
        kernel_depth, kernel_rows, kernel_cols,
        BLOCK_C,
    ))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_C": BLOCK_C, "occupancy": 2})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
