import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _jacobi_kernel(input, output, rows, cols,
                   BM: ConstInt, BN: ConstInt):
    bm = ct.bid(0)
    bn = ct.bid(1)

    rs = ct.arange(BM, dtype=np.int32) + bm * BM
    cs = ct.arange(BN, dtype=np.int32) + bn * BN

    r_2d = ct.broadcast_to(rs[:, None], (BM, BN))
    c_2d = ct.broadcast_to(cs[None, :], (BM, BN))

    # Clamp so every gather index is guaranteed in-bounds → skip bounds-check.
    r_c = ct.minimum(r_2d, rows - 1)
    c_c = ct.minimum(c_2d, cols - 1)
    r_up = ct.maximum(r_c - 1, 0)
    r_dn = ct.minimum(r_c + 1, rows - 1)
    c_lf = ct.maximum(c_c - 1, 0)
    c_rt = ct.minimum(c_c + 1, cols - 1)

    center = ct.gather(input, (r_c, c_c), check_bounds=False)
    up    = ct.gather(input, (r_up, c_c), check_bounds=False)
    down  = ct.gather(input, (r_dn, c_c), check_bounds=False)
    left  = ct.gather(input, (r_c, c_lf), check_bounds=False)
    right = ct.gather(input, (r_c, c_rt), check_bounds=False)

    center_f = ct.astype(center, np.float32)
    up_f     = ct.astype(up,     np.float32)
    down_f   = ct.astype(down,   np.float32)
    left_f   = ct.astype(left,   np.float32)
    right_f  = ct.astype(right,  np.float32)

    avg = 0.25 * (up_f + down_f + left_f + right_f)

    is_interior = (r_2d > 0) & (r_2d < rows - 1) & (c_2d > 0) & (c_2d < cols - 1)
    result_f = ct.where(is_interior, avg, center_f)

    result = ct.astype(result_f, input.dtype)
    ct.store(output, index=(bm, bn), tile=result)


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    BM = 16
    BN = 128

    grid = (ct.cdiv(rows, BM), ct.cdiv(cols, BN), 1)
    ct.launch(stream, grid, _jacobi_kernel,
              (input, output, rows, cols, BM, BN))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BM": BM, "BN": BN, "occupancy": 2})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
