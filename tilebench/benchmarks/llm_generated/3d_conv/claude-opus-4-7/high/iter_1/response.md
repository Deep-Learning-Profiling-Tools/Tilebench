Looking at the iter 0 results: Triton at 10.8% (verify-clean) but only used BLOCK_C=128; cuTile failed because `with_hints` doesn't exist on `@ct.kernel` objects — occupancy must be baked into the decorator.

**Plan:**
- **Triton**: bump `BLOCK_C` to 512 with `num_warps=8`, `num_stages=3` to better saturate HBM and amortize the 3×3×3 weight broadcast loop.
- **cuTile**: keep the simple per-(out_d,out_r) tile, but set `occupancy=2` on the `@ct.kernel` decorator instead of using the non-existent `with_hints` API. Use larger `BLOCK_C=512`.

```python title="impl_triton.py"
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
        d_base = (out_d + kd) * IR * IC
        for kr in tl.static_range(KR):
            row_base = d_base + (out_r + kr) * IC
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

    BLOCK_C = 512
    num_warps = 8
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
```
