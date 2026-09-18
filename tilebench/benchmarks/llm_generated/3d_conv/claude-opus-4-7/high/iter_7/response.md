Looking at the trajectory: best Triton (10.8%) uses BLOCK_R=1, best cuTile (13.9%) uses BLOCK_R=8. The kernel is bandwidth-bound and benefits from row-tiling for input reuse. Let me apply the row-tiling that worked for cuTile to Triton, and try a wider tile for cuTile.

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
    num_r_blocks = tl.cdiv(OR, BLOCK_R)
    out_d = pid_drb // num_r_blocks
    rb = pid_drb % num_r_blocks

    r_offs = rb * BLOCK_R + tl.arange(0, BLOCK_R)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    r_mask = r_offs < OR
    c_mask = c_offs < OC

    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)

    for kd in tl.static_range(KD):
        d_base = (out_d + kd) * IR * IC
        for kr in tl.static_range(KR):
            row_base = d_base + (r_offs[:, None] + kr) * IC
            for kc in tl.static_range(KC):
                k_val = tl.load(k_ptr + (kd * KR * KC + kr * KC + kc)).to(tl.float32)
                x_offs = row_base + (c_offs[None, :] + kc)
                x_val = tl.load(x_ptr + x_offs, mask=c_mask[None, :], other=0.0).to(tl.float32)
                acc += k_val * x_val

    out_offs = out_d * OR * OC + r_offs[:, None] * OC + c_offs[None, :]
    out_mask = r_mask[:, None] & c_mask[None, :]
    tl.store(out_ptr + out_offs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, **kwargs):
    OD = input_depth - kernel_depth + 1
    OR = input_rows - kernel_rows + 1
    OC = input_cols - kernel_cols + 1

    output = torch.empty(OD * OR * OC, dtype=input.dtype, device=input.device)

    BLOCK_R = 8
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

    r_offs = rb * BLOCK_R + ct.arange(BLOCK_R, dtype=np.int32)
    c_offs = pid_c * BLOCK_C + ct.arange(BLOCK_C, dtype=np.int32)
    r_valid = r_offs < OR
    c_valid = c_offs < OC

    acc = ct.full((BLOCK_R, BLOCK_C), 0.0, dtype=np.float32)

    for kd in range(KD):
        d_base = (out_d + kd) * IR * IC
        for kr in range(KR):
            in_r = r_offs + kr
            row_base = d_base + in_r[:, None] * IC
            for kc in range(KC):
                k_val_t = ct.load(k_3d, index=(kd, kr, kc), shape=())
                k_val = ct.astype(k_val_t, np.float32)
                x_idx = row_base + (c_offs[None, :] + kc)
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

    BLOCK_R = 8
    BLOCK_C = 256
    occupancy = 2

    stream = torch.cuda.current_stream()
    grid = (OD * ct.cdiv(OR, BLOCK_R), ct.cdiv(OC, BLOCK_C), 1)

    ct.launch(stream, grid, _conv3d_kernel, (
        input, k_3d, output,
        input_rows, input_cols, OR, OC,
        kernel_depth, kernel_rows, kernel_cols,
        BLOCK_R, BLOCK_C,
    ))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_R": BLOCK_R, "BLOCK_C": BLOCK_C, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Triton: switched to 2D row+col output tiling (BLOCK_R=8, BLOCK_C=128) matching the cuTile winner — better input reuse across rows. cuTile: widened BLOCK_C to 256 to amortize launch overhead and improve L1/coalescing per tile.
