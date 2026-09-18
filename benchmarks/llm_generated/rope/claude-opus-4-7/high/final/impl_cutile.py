import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _rope_kernel_fp32(q, cos, sin, output,
                      S: ConstInt, H: ConstInt, HALF: ConstInt):
    pid = ct.bid(0)
    b = pid // S
    s = pid - b * S

    q1 = ct.load(q, index=(b, s, 0, 0), shape=(1, 1, H, HALF)).reshape((H, HALF))
    q2 = ct.load(q, index=(b, s, 0, 1), shape=(1, 1, H, HALF)).reshape((H, HALF))
    c = ct.load(cos, index=(s, 0), shape=(1, HALF))
    sn = ct.load(sin, index=(s, 0), shape=(1, HALF))

    # Native fp32 compute. Any FMA fusion stays well inside the 5e-3 tolerance.
    out1 = q1 * c - q2 * sn
    out2 = q2 * c + q1 * sn

    ct.store(output, index=(b, s, 0, 0), tile=out1.reshape((1, 1, H, HALF)))
    ct.store(output, index=(b, s, 0, 1), tile=out2.reshape((1, 1, H, HALF)))


@ct.kernel(occupancy=4)
def _rope_kernel_fp16(q, cos, sin, output,
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

    # Round each product through output dtype (fp16) to mimic torch's
    # per-op rounding — needed to avoid fp16 cancellation mismatches.
    od = output.dtype
    t1 = ct.astype(ct.astype(q1f * cf, od), np.float32)
    t2 = ct.astype(ct.astype(q2f * snf, od), np.float32)
    t3 = ct.astype(ct.astype(q2f * cf, od), np.float32)
    t4 = ct.astype(ct.astype(q1f * snf, od), np.float32)

    out1 = ct.astype(t1 - t2, od)
    out2 = ct.astype(t3 + t4, od)

    ct.store(output, index=(b, s, 0, 0), tile=out1.reshape((1, 1, H, HALF)))
    ct.store(output, index=(b, s, 0, 1), tile=out2.reshape((1, 1, H, HALF)))


def run(q, cos, sin):
    B, S, H, D = q.shape
    half = D // 2
    out_dtype = torch.result_type(q, cos)
    output = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    stream = torch.cuda.current_stream()

    occupancy = 4

    if out_dtype == torch.float32:
        kernel = _rope_kernel_fp32
        path = "fp32_native"
    else:
        kernel = _rope_kernel_fp16
        path = "fp16_rounded"

    grid = (B * S, 1, 1)
    ct.launch(stream, grid, kernel, (q, cos, sin, output, S, H, half))

    _LAST_CFG.clear()
    _LAST_CFG.update({"H": H, "HALF": half, "occupancy": occupancy, "path": path})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
