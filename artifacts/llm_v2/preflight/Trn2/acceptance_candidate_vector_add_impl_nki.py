"""Evaluator acceptance candidate (vector_add, fp16): written for the Trn2 evaluator validation only.
x and y are viewed as [128, n/128]; the columns are split between the two physical cores of the
logical NeuronCore (program_id), and each core streams fixed [128, 2048] tiles through SBUF."""
import nki
import nki.isa as nisa
import nki.language as nl

P = 128
F = 2048
LNC = 2


@nki.jit
def _add_kernel(x, y):
    M = x.shape[1]
    out = nl.ndarray((P, M), dtype=x.dtype, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    per = M // nl.num_programs()
    base = pid * per
    for t in range(per // F):
        c0 = base + t * F
        xt = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        yt = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        zt = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x[:, c0:c0 + F])
        nisa.dma_copy(dst=yt, src=y[:, c0:c0 + F])
        nisa.tensor_tensor(dst=zt, data1=xt, data2=yt, op=nl.add)
        nisa.dma_copy(dst=out[:, c0:c0 + F], src=zt)
    return out


_LAST = {}


def run(x, y):
    n = x.numel()
    if n % (P * F * LNC):
        raise NotImplementedError("acceptance candidate needs n % (128 * 2048 * 2) == 0")
    out = _add_kernel[LNC](x.view(P, n // P), y.view(P, n // P))
    _LAST.clear()
    _LAST.update({"P": P, "F": F, "lnc": LNC})
    return out.view(n)


def get_last_config():
    return dict(_LAST)
