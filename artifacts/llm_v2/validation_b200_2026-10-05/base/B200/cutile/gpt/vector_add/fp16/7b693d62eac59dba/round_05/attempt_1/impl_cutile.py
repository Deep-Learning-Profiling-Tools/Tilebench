import torch
import cuda.tile as ct


@ct.kernel(num_ctas=1, occupancy=4, opt_level=3)
def _vector_add(x, y, out, N: ct.Constant[int]):
    offsets = ct.bid(0) * 8192 + ct.arange(8192, dtype=ct.int32)
    xm = x.get_raw_memory()
    ym = y.get_raw_memory()
    om = out.get_raw_memory()

    if N % 8192 == 0:
        a = xm.load_offset(offsets)
        b = ym.load_offset(offsets)
        om.store_offset(offsets, a + b)
    else:
        mask = offsets < N
        a = xm.load_offset(offsets, mask=mask, padding_value=0)
        b = ym.load_offset(offsets, mask=mask, padding_value=0)
        om.store_offset(offsets, a + b, mask=mask)


def run(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    if n > 0:
        ct.launch(
            torch.cuda.current_stream(x.device),
            (ct.cdiv(n, 8192),),
            _vector_add,
            (x, y, out, n),
        )
    return out


def get_last_config() -> dict:
    return {
        "tile_size": 8192,
        "num_ctas": 1,
        "occupancy": 4,
        "opt_level": 3,
        "addressing": "raw",
        "unmasked_when_divisible": True,
    }
