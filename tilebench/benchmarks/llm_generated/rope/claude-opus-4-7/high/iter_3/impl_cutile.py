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

    q1 = ct.load(q, index=(b, s, 0, 0), shape=(1, 1, H, HALF)).reshape((H, HALF))
    q2 = ct.load(q, index=(b, s, 0, 1), shape=(1, 1, H, HALF)).reshape((H, HALF))
    c = ct.load(cos, index=(s, 0), shape=(1, HALF))
    sn = ct.load(sin, index=(s, 0), shape=(1, HALF))

    q1f = ct.astype(q1, np.float32)
    q2f = ct.astype(q2, np.float32)
    cf = ct.astype(c, np.float32)
    snf = ct.astype(sn, np.float32)

    t1 = q1f * cf
    t2 = q2f * snf
    t3 = q2f * cf
    t4 = q1f * snf

    # Round through output dtype to match torch's per-op rounding semantics
    # (no-op when output dtype is fp32; fp16-quantizes when output is fp16).
    od = output.dtype
    t1r = ct.astype(ct.astype(t1, od), np.float32)
    t2r = ct.astype(ct.astype(t2, od), np.float32)
    t3r = ct.astype(ct.astype(t3, od), np.float32)
    t4r = ct.astype(ct.astype(t4, od), np.float32)

    out1 = ct.astype(t1r - t2r, od)
    out2 = ct.astype(t3r + t4r, od)

    ct.store(output, index=(b, s, 0, 0), tile=out1.reshape((1, 1, H, HALF)))
    ct.store(output, index=(b, s, 0, 1), tile=out2.reshape((1, 1, H, HALF)))


def run(q, cos, sin):
    B, S, H, D = q.shape
    half = D // 2
    out_dtype = torch.result_type(q, cos)
    output = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    stream = torch.cuda.current_stream()

    occupancy = 4

    grid = (B * S, 1, 1)
    ct.launch(stream, grid, _rope_kernel, (q, cos, sin, output, S, H, half))

    _LAST_CFG.clear()
    _LAST_CFG.update({"H": H, "HALF": half, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
