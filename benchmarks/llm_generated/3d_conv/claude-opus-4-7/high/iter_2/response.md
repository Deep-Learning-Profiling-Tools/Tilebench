Looking at the regression in iter 1, BLOCK_C=512 was worse than BLOCK_C=128. The bottleneck is bandwidth — each output requires 27 input reads with heavy overlap. Let me try processing **multiple output rows per program** (2D tile of shape `BLOCK_R × BLOCK_C`). Adjacent rows share input rows across `kr`, giving better cache reuse, and fewer total programs reduce launch overhead. I'll also add eviction-policy hints to keep reused input in L2.

```python title="impl_triton.py"
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

    r_offs = rb * BLOCK_R + tl.arange(0, BLOCK_R)        # [BLOCK_R]
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)     # [BLOCK_C]
    r_mask = r_offs < OR
    c_mask = c_offs < OC

    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)

    for kd in tl.static_range(KD):
        d_base = (out_d + kd) * IR * IC
        for kr in tl.static_range(KR):
            in_r = r_offs + kr                            # [BLOCK_R]
            row_base = d_base + in_r[:, None] * IC        # [BLOCK_R, 1]
            for kc in tl.static_range(KC):
                k_val = tl.load(k_ptr + (kd * KR * KC + kr * KC + kc)).to(tl.float32)
                x_addr = row_base + (c_offs[None, :] + kc)
                mask2d = r_mask[:, None] & c_mask[None, :]
                x_val = tl.load(x_ptr + x_addr, mask=mask2d, other=0.0,
                                eviction_policy="evict_last").to(tl.float32)
                acc += k_val * x_val

    out_addr = out_d * OR * OC + r_offs[:, None] * OC + c_offs[None, :]
    out_mask = r_mask[:, None] & c_mask[None, :]
    tl.store(out_ptr + out_addr, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, **kwargs):
    OD = input_depth - kernel_depth + 1
    OR = input_rows - kernel_rows + 1
    OC = input_cols - kernel_cols + 1

    output = torch.empty(OD * OR * OC, dtype=input.dtype, device=input.device)

    BLOCK_R = 4
    BLOCK_C = 128
    num_warps = 4
    num_stages = 2

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
```

```python title="impl_cutile.py"
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
```

Approach: 2D output tile `BLOCK_R × BLOCK_C` (4×128) per program. Adjacent output rows share input rows across the `kr` loop, improving cache reuse vs single-row programs. `evict_last` keeps reused input warm in L2.
