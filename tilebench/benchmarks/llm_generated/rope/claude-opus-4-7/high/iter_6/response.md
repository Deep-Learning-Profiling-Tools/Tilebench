Looking at the trajectory:
- Iter 3 (best clean: 77.6%) used 1 position/CTA with per-op fp16 rounding
- Iter 5 (SPB=4, 34.4%) regressed badly — likely register pressure from larger tiles

Going back to iter-3's geometry but removing the unnecessary per-op rounding through output dtype. The verify tolerance is loose (5e-3), so single-cast-at-end is fine and reduces work / register pressure.

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

    # Promote to fp32 for compute, cast back at the very end.
    # Verify tolerance (atol=5e-3, rtol=5e-3) is loose enough that
    # we do NOT need per-product rounding through the output dtype.
    q1f = ct.astype(q1, np.float32)
    q2f = ct.astype(q2, np.float32)
    cf = ct.astype(c, np.float32)
    snf = ct.astype(sn, np.float32)

    od = output.dtype
    out1 = ct.astype(q1f * cf - q2f * snf, od)
    out2 = ct.astype(q2f * cf + q1f * snf, od)

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

Approach: revert to iter-3's geometry (1 sequence position per CTA, occupancy=4) since SPB=4 caused register pressure / occupancy collapse. Strip the redundant per-product roundtrips through `od` — compute in fp32 and round once at store time. Verify tolerance (5e-3) tolerates the single end-cast.
