import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv3d_kernel(x_ptr, k_ptr, out_ptr,
                   IR, IC, OR, OC,
                   KD: tl.constexpr, KR: tl.constexpr, KC: tl.constexpr,
                   BLOCK_C: tl.constexpr):
    pid_dr = tl.program_id(0)
    pid_c = tl.program_id(1)
    out_d = pid_dr // OR
    out_r = pid_dr % OR

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < OC

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for kd in tl.static_range(KD):
        for kr in tl.static_range(KR):
            row_base = (out_d + kd) * IR * IC + (out_r + kr) * IC
            for kc in tl.static_range(KC):
                k_val = tl.load(k_ptr + (kd * KR * KC + kr * KC + kc)).to(tl.float32)
                x_offs = row_base + c_offs + kc
                x_val = tl.load(x_ptr + x_offs, mask=c_mask, other=0.0).to(tl.float32)
                acc += k_val * x_val

    out_offs = out_d * OR * OC + out_r * OC + c_offs
    tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=c_mask)


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, **kwargs):
    OD = input_depth - kernel_depth + 1
    OR = input_rows - kernel_rows + 1
    OC = input_cols - kernel_cols + 1

    output = torch.empty(OD * OR * OC, dtype=input.dtype, device=input.device)

    BLOCK_C = 128
    num_warps = 4
    num_stages = 3

    grid = (OD * OR, triton.cdiv(OC, BLOCK_C))

    _conv3d_kernel[grid](
        input, kernel, output,
        input_rows, input_cols, OR, OC,
        kernel_depth, kernel_rows, kernel_cols,
        BLOCK_C,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_C": BLOCK_C,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
