import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _rope_kernel(q, cos, sin, output,
                 S: ConstInt, H: ConstInt, HALF: ConstInt):
    pid = ct.bid(0)
    b = pid // S
    s = pid - b * S

    q1 = ct.astype(ct.load(q, index=(b, s, 0, 0), shape=(1, 1, H, HALF)),
                   np.float32).reshape((H, HALF))
    q2 = ct.astype(ct.load(q, index=(b, s, 0, 1), shape=(1, 1, H, HALF)),
                   np.float32).reshape((H, HALF))

    c = ct.astype(ct.load(cos, index=(s, 0), shape=(1, HALF)), np.float32)
    sn = ct.astype(ct.load(sin, index=(s, 0), shape=(1, HALF)), np.float32)

    out1 = q1 * c - q2 * sn
    out2 = q2 * c + q1 * sn

    ct.store(output, index=(b, s, 0, 0),
             tile=ct.astype(out1.reshape((1, 1, H, HALF)), q.dtype))
    ct.store(output, index=(b, s, 0, 1),
             tile=ct.astype(out2.reshape((1, 1, H, HALF)), q.dtype))


def run(q, cos, sin):
    B, S, H, D = q.shape
    half = D // 2
    output = torch.empty_like(q)
    stream = torch.cuda.current_stream()

    occupancy = 4

    grid = (B * S, 1, 1)
    kernel = _rope_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (q, cos, sin, output, S, H, half))

    _LAST_CFG.clear()
    _LAST_CFG.update({"H": H, "HALF": half, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
