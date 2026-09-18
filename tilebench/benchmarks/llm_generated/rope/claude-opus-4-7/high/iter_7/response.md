Looking at the trajectory:
- Iter 3 (best verify-clean, 77.6%): explicit fp32 promotion + per-product rounding through output dtype
- Iter 6 (faster, 100% but failed fp16 verify): fp32 promotion but no intermediate rounding → mismatches of 0.0078 (>atol 0.005)

The fix: compute **natively in the input dtype** (no fp32 promotion at all). This means each multiply produces a fp16/fp32 tile in its natural dtype, matching PyTorch's per-op rounding exactly without the overhead of explicit double-cast rounding.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _rope_kernel(q, cos, sin, output,
                 S: ConstInt, H: ConstInt, HALF: ConstInt):
    pid = ct.bid(0)
    b = pid // S
    s = pid - b * S

    # Two contiguous half-tile loads of the head dim. Each load is
    # H * HALF * dtype contiguous bytes in q (B,S,H,D) layout.
    q1 = ct.load(q, index=(b, s, 0, 0),
                 shape=(1, 1, H, HALF)).reshape((H, HALF))
    q2 = ct.load(q, index=(b, s, 0, 1),
                 shape=(1, 1, H, HALF)).reshape((H, HALF))
    c = ct.load(cos, index=(s, 0), shape=(1, HALF))
    sn = ct.load(sin, index=(s, 0), shape=(1, HALF))

    # Compute natively in the input dtype. For fp16 this means each
    # multiply rounds to fp16 (matching torch's per-op rounding), and
    # for fp32 this is just normal fp32 arithmetic. This is far cheaper
    # than fp32-promote-then-round-back, and avoids the FMA-fusion
    # accuracy regression seen in iter 6 (which promoted to fp32 and
    # skipped intermediate rounding).
    t1 = q1 * c
    t2 = q2 * sn
    t3 = q2 * c
    t4 = q1 * sn

    out1 = t1 - t2
    out2 = t3 + t4

    ct.store(output, index=(b, s, 0, 0),
             tile=out1.reshape((1, 1, H, HALF)))
    ct.store(output, index=(b, s, 0, 1),
             tile=out2.reshape((1, 1, H, HALF)))


def run(q, cos, sin):
    B, S, H, D = q.shape
    half = D // 2
    out_dtype = torch.result_type(q, cos)
    output = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    stream = torch.cuda.current_stream()

    occupancy = 4

    grid = (B * S, 1, 1)
    ct.launch(stream, grid, _rope_kernel,
              (q, cos, sin, output, S, H, half))

    _LAST_CFG.clear()
    _LAST_CFG.update({"H": H, "HALF": half, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: compute entirely in the input's native dtype (no fp32 promotion) — this matches torch's per-op rounding semantics for fp16 without the overhead of explicit cast-round-cast that capped iter 3 at 77.6%, while avoiding the FMA-fusion accuracy loss that broke iter 6.
