import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _rope_kernel(q, cos, sin, output,
                 S: ConstInt, H: ConstInt, HALF: ConstInt, SPB: ConstInt):
    pid = ct.bid(0)
    num_s_blocks = S // SPB
    b = pid // num_s_blocks
    s_tile = pid - b * num_s_blocks  # tile-space coord on seq axis

    # Tile shape (1, SPB, H, HALF): axis-1 tile-space index `s_tile`
    # maps to element start `s_tile * SPB`, covering SPB seq positions.
    q1 = ct.load(q, index=(b, s_tile, 0, 0),
                 shape=(1, SPB, H, HALF)).reshape((SPB, H, HALF))
    q2 = ct.load(q, index=(b, s_tile, 0, 1),
                 shape=(1, SPB, H, HALF)).reshape((SPB, H, HALF))
    c = ct.load(cos, index=(s_tile, 0), shape=(SPB, HALF))
    sn = ct.load(sin, index=(s_tile, 0), shape=(SPB, HALF))

    cb = ct.broadcast_to(c.reshape((SPB, 1, HALF)), (SPB, H, HALF))
    snb = ct.broadcast_to(sn.reshape((SPB, 1, HALF)), (SPB, H, HALF))

    q1f = ct.astype(q1, np.float32)
    q2f = ct.astype(q2, np.float32)
    cf = ct.astype(cb, np.float32)
    snf = ct.astype(snb, np.float32)

    t1 = q1f * cf
    t2 = q2f * snf
    t3 = q2f * cf
    t4 = q1f * snf

    # Round each product through the output dtype to mirror torch's
    # per-op rounding semantics (no-op when od == fp32).
    od = output.dtype
    t1r = ct.astype(ct.astype(t1, od), np.float32)
    t2r = ct.astype(ct.astype(t2, od), np.float32)
    t3r = ct.astype(ct.astype(t3, od), np.float32)
    t4r = ct.astype(ct.astype(t4, od), np.float32)

    out1 = ct.astype(t1r - t2r, od)
    out2 = ct.astype(t3r + t4r, od)

    ct.store(output, index=(b, s_tile, 0, 0),
             tile=out1.reshape((1, SPB, H, HALF)))
    ct.store(output, index=(b, s_tile, 0, 1),
             tile=out2.reshape((1, SPB, H, HALF)))


def run(q, cos, sin):
    B, S, H, D = q.shape
    half = D // 2
    out_dtype = torch.result_type(q, cos)
    output = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    stream = torch.cuda.current_stream()

    # Sequence positions per CTA. Cases use S = 1024*i (always divisible
    # by 4); fall back to smaller SPB if S happens not to be divisible.
    if S % 4 == 0:
        SPB = 4
    elif S % 2 == 0:
        SPB = 2
    else:
        SPB = 1

    occupancy = 4

    grid = ((B * S) // SPB, 1, 1)
    ct.launch(stream, grid, _rope_kernel,
              (q, cos, sin, output, S, H, half, SPB))

    _LAST_CFG.clear()
    _LAST_CFG.update({"H": H, "HALF": half, "SPB": SPB,
                      "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
