import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _conv1d_kernel(x, w, out, input_size, output_size,
                   KERNEL_SIZE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    base = bid * TILE

    acc = ct.full((TILE,), 0.0, dtype=np.float32)

    for j in range(KERNEL_SIZE):
        wj = ct.load(w, index=(j,), shape=())
        wj_f = ct.astype(wj, np.float32)
        x_sub = x.slice(0, base + j, input_size)
        xj = ct.load(x_sub, index=(0,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
        xj_f = ct.astype(xj, np.float32)
        acc = acc + xj_f * wj_f

    out_sub = out.slice(0, base, output_size)
    ct.store(out_sub, index=(0,), tile=ct.astype(acc, out.dtype))


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 2  # encoded in decorator

    grid = (ct.cdiv(output_size, TILE), 1, 1)
    ct.launch(stream, grid, _conv1d_kernel,
              (input, kernel, output, input_size, output_size, kernel_size, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
