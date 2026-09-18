import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv3d_kernel(x_ptr, k_ptr, out_ptr,
                   IR, IC, OR, OC,
                   KD: tl.constexpr, KR: tl.constexpr, KC: tl.constexpr,
                   BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_drb = tl.program_id(0)
    pid_c = tl.program_id(1)
    num_row_blocks = tl.cdiv(OR, BLOCK_R)
    out_d = pid_drb // num_row_blocks
    rb = pid_drb % num_row_blocks

    r_offs = rb * BLOCK_R + tl.arange(0, BLOCK_R)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r_mask = r_offs < OR
    c_mask = c_offs < OC

    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)

    for kd in tl.static_range(KD):
        d_base = (out_d + kd) * IR * IC
        for kr in tl.static_range(KR):
            row_base = d_base + (r_offs + kr)[:, None] * IC
            for kc in tl.static_range(KC):
                k_val = tl.load(k_ptr + (kd * KR * KC + kr * KC + kc)).to(tl.float32)
                x_offs = row_base + (c_offs + kc)[None, :]
                x_val = tl.load(x_ptr + x_offs, mask=c_mask[None, :], other=0.0).to(tl.float32)
                acc += k_val * x_val

    out_offs = out_d * OR * OC + r_offs[:, None] * OC + c_offs[None, :]
    valid = r_mask[:, None] & c_mask[None, :]
    tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=valid)


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, **kwargs):
    OD = input_depth - kernel_depth + 1
    OR = input_rows - kernel_rows + 1
    OC = input_cols - kernel_cols + 1

    output = torch.empty(OD * OR * OC, dtype=input.dtype, device=input.device)

    BLOCK_R = 4
    BLOCK_C = 128
    num_warps = 4
    num_stages = 3

    grid = (OD * triton.cdiv(OR, BLOCK_R), triton.cdiv(OC, BLOCK_C))

    _conv3d_kernel[grid](
        input, kernel, output,
        input_rows, input_cols, OR, OC,
        kernel_depth, kernel_rows, kernel_cols,
        BLOCK_R, BLOCK_C,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_R": BLOCK_R,
        "BLOCK_C": BLOCK_C,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
