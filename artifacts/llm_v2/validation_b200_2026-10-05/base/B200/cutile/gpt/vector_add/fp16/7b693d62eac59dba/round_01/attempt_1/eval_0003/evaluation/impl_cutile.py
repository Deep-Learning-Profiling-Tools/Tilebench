import torch
import cuda.tile as ct


@ct.kernel(num_ctas=1, occupancy=4, opt_level=3)
def _vector_add(x, y, out):
    bid = ct.bid(0)
    a = ct.load(
        x, (bid,), (8192,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    b = ct.load(
        y, (bid,), (8192,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    ct.store(out, (bid,), a + b, allow_tma=False)


def run(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    if n > 0:
        ct.launch(
            torch.cuda.current_stream(x.device),
            (ct.cdiv(n, 8192),),
            _vector_add,
            (x, y, out),
        )
    return out


def get_last_config() -> dict:
    return {
        "tile_size": 8192,
        "num_ctas": 1,
        "occupancy": 4,
        "opt_level": 3,
        "allow_tma": False,
    }
