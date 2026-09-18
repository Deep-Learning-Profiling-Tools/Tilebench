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

    # Two contiguous half-tile loads (B,S,H,D) layout.
    q1 = ct.load(q, index=(b, s, 0, 0),
                 shape=(1, 1, H, HALF)).reshape((H, HALF))
    q2 = ct.load(q, index=(b, s, 0, 1),
                 shape=(1, 1, H, HALF)).reshape((H, HALF))
    c = ct.load(cos, index=(s, 0), shape=(1, HALF))
    sn = ct.load(sin, index=(s, 0), shape=(1, HALF))

    # Compute natively in input dtype. To match torch's per-op rounding
    # (no FMA fusion across the multiply -> subtract/add boundary), insert
    # optimization barriers between the multiplies and the final
    # subtract/add. This preserves fp16 cancellation semantics, which is
    # what triggered the iter-7 verify failure (inf relative diff at
    # locations where torch's non-fused fp16 result was exactly 0).
    t1 = q1 * c
    t2 = q2 * sn
    t3 = q2 * c
    t4 = q1 * sn

    t1 = ct.optimization_barrier(t1)
    t2 = ct.optimization_barrier(t2)
    t3 = ct.optimization_barrier(t3)
    t4 = ct.optimization_barrier(t4)

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
