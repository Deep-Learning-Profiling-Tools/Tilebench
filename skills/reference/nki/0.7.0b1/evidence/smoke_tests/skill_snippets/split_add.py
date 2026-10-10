# example: split_add
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def split_add(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    P = 128
    n_prog = nl.num_programs()            # 2 for kernel(...) / kernel[2](...), 1 for kernel[1](...)
    pid = nl.program_id(0)                # 0 or 1
    rows_per_prog = a.shape[0] // n_prog
    for i in range(rows_per_prog // P):
        r0 = pid * rows_per_prog + i * P
        a_t = nl.ndarray((P, a.shape[1]), dtype=a.dtype, buffer=nl.sbuf)
        b_t = nl.ndarray((P, a.shape[1]), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_t, src=a[r0:r0 + P, :])
        nisa.dma_copy(dst=b_t, src=b[r0:r0 + P, :])
        nisa.tensor_tensor(dst=a_t, data1=a_t, data2=b_t, op=nl.add)   # dst may alias an input
        nisa.dma_copy(dst=out[r0:r0 + P, :], src=a_t)
    return out
