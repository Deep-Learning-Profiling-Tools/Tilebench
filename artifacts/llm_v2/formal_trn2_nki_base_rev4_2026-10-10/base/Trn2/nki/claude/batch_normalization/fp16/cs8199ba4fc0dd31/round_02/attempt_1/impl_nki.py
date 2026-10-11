import torch
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
BANK = 512
_LAST_CONFIG = {}


def _choose_k(N, C):
    # rows per partition in one DMA chunk (chunk = 128 * K rows)
    return 4


def _stats_chunk(x, G, S, ones, r0, np_, k, C, NB, first):
    # rows r0 + p*k + j  (p < np_, j < k) -> partition p, free block j
    xt = nl.ndarray((P, k * C), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt[0:np_, :],
                  src=x.ap(pattern=[[k * C, np_], [1, k * C]], offset=r0 * C))
    for j in range(k):
        acc = (first + j) > 0
        for b in range(NB):
            c0 = j * C + b * P
            # Gram block: diag = per-channel sum of squares (fp32 accumulation)
            nisa.nc_matmul(dst=G[:, b * P:(b + 1) * P],
                           stationary=xt[0:np_, c0:c0 + P],
                           moving=xt[0:np_, c0:c0 + P],
                           accumulate=acc)
            # per-channel sum
            nisa.nc_matmul(dst=S[:, b:b + 1],
                           stationary=xt[0:np_, c0:c0 + P],
                           moving=ones[0:np_, :],
                           accumulate=acc)


def _apply_chunk(x, out, SBs, SBt, r0, np_, k, C):
    xa = nl.ndarray((P, k * C), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xa[0:np_, :],
                  src=x.ap(pattern=[[k * C, np_], [1, k * C]], offset=r0 * C))
    for j in range(k):
        tmp = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tmp[0:np_, :], data1=xa[0:np_, j * C:(j + 1) * C],
                           data2=SBs[0:np_, :], op=nl.multiply,
                           engine=nisa.engine.vector)
        if j % 2 == 0:
            nisa.tensor_tensor(dst=xa[0:np_, j * C:(j + 1) * C], data1=tmp[0:np_, :],
                               data2=SBt[0:np_, :], op=nl.add,
                               engine=nisa.engine.gpsimd)
        else:
            nisa.tensor_tensor(dst=xa[0:np_, j * C:(j + 1) * C], data1=tmp[0:np_, :],
                               data2=SBt[0:np_, :], op=nl.add,
                               engine=nisa.engine.vector)
    nisa.dma_copy(dst=out.ap(pattern=[[k * C, np_], [1, k * C]], offset=r0 * C),
                  src=xa[0:np_, :])


@nki.jit
def _bn_kernel(x, gamma, beta, eps, K):
    Nr = x.shape[0]
    C = x.shape[1]
    NB = C // P
    f32 = nl.float32
    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    out = nl.ndarray((Nr, C), dtype=x.dtype, buffer=nl.shared_hbm)
    part = nl.ndarray((n_prog * P, 2 * NB), dtype=f32, buffer=nl.shared_hbm)
    ssh = nl.ndarray((n_prog * 2 * NB, P), dtype=f32, buffer=nl.shared_hbm)

    r_start = (pid * Nr) // n_prog
    r_end = ((pid + 1) * Nr) // n_prog
    R = r_end - r_start
    CH = P * K

    # gamma / beta in channel-on-partition layout: [p, b] = v[b*128 + p]
    g16 = nl.ndarray((P, NB), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g16, src=gamma.reshape((1, C)).ap(pattern=[[1, P], [P, NB]], offset=0))
    b16 = nl.ndarray((P, NB), dtype=beta.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=b16, src=beta.reshape((1, C)).ap(pattern=[[1, P], [P, NB]], offset=0))
    gt = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gt, src=g16)
    bt = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=bt, src=b16)

    # identity mask replicated NB times (for Gram diagonal extraction)
    a_i = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=a_i, pattern=[[1, P]], offset=0, channel_multiplier=1)   # p + j
    b_i = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=b_i, pattern=[[2, P]], offset=0, channel_multiplier=0)   # 2j
    a_f = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=a_f, src=a_i)
    b_f = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=b_f, src=b_i)
    ident8 = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    for b in range(NB):
        nisa.tensor_tensor(dst=ident8[:, b * P:(b + 1) * P], data1=a_f, data2=b_f,
                           op=nl.equal)

    ones = nl.ndarray((P, 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)

    G = nl.ndarray((P, C), dtype=f32, buffer=nl.psum)
    S = nl.ndarray((P, NB), dtype=f32, buffer=nl.psum)

    # ---------------- stage 2: partial statistics ----------------
    nfull = R // CH
    rem = R - nfull * CH
    kr = rem // P
    rr = rem - kr * P
    for i in range(nfull):
        _stats_chunk(x, G, S, ones, r_start + i * CH, P, K, C, NB, min(i, 1))
    if kr > 0:
        _stats_chunk(x, G, S, ones, r_start + nfull * CH, P, kr, C, NB, min(nfull, 1))
    if rr > 0:
        _stats_chunk(x, G, S, ones, r_start + nfull * CH + kr * P, rr, 1, C, NB,
                     min(nfull + kr, 1))

    PT = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=PT[:, 0:NB], src=S)
    Dg = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    nh = (C + BANK - 1) // BANK
    for h in range(nh):
        c0 = h * BANK
        c1 = min(C, c0 + BANK)
        nisa.tensor_tensor(dst=Dg[:, c0:c1], data1=G[:, c0:c1], data2=ident8[:, c0:c1],
                           op=nl.multiply)
    nisa.tensor_reduce(dst=PT[:, NB:2 * NB], op=nl.add, data=Dg.reshape((P, NB, P)), axis=2)
    nisa.dma_copy(dst=part[pid * P:(pid + 1) * P, :], src=PT)

    if n_prog > 1:
        nisa.core_barrier(data=part, cores=(0, 1))

    # ---------------- stage 3: combine (fixed order) ----------------
    A0 = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=A0, src=part[0:P, :])
    tot = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    if n_prog > 1:
        A1 = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=A1, src=part[P:2 * P, :])
        nisa.tensor_tensor(dst=tot, data1=A0, data2=A1, op=nl.add)
    else:
        nisa.tensor_copy(dst=tot, src=A0)

    inv_n = 1.0 / float(Nr)
    mom = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mom, data=tot, op0=nl.multiply, operand0=inv_n)
    msq = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=msq, data1=mom[:, 0:NB], data2=mom[:, 0:NB], op=nl.multiply)
    var = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=var, data1=mom[:, NB:2 * NB], data2=msq, op=nl.subtract)
    nisa.tensor_scalar(dst=var, data=var, op0=nl.maximum, operand0=0.0,
                       op1=nl.add, operand1=eps)
    inv = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.activation(dst=inv, op=nl.rsqrt, data=var)
    SS = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=SS[:, 0:NB], data1=inv, data2=gt, op=nl.multiply)
    ms = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ms, data1=mom[:, 0:NB], data2=SS[:, 0:NB], op=nl.multiply)
    nisa.tensor_tensor(dst=SS[:, NB:2 * NB], data1=bt, data2=ms, op=nl.subtract)

    # to channel order rows, then broadcast to all partitions
    psT = nl.ndarray((2 * NB, P), dtype=f32, buffer=nl.psum)
    nisa.nc_transpose(dst=psT, data=SS, engine=nisa.engine.tensor)
    SST = nl.ndarray((2 * NB, P), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=SST, src=psT)
    nisa.dma_copy(dst=ssh[pid * 2 * NB:(pid + 1) * 2 * NB, :], src=SST)
    SBs = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=SBs, src=ssh.ap(pattern=[[0, P], [1, C]], offset=pid * 2 * C))
    SBt = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=SBt, src=ssh.ap(pattern=[[0, P], [1, C]], offset=pid * 2 * C + C))

    # ---------------- stage 4: apply ----------------
    for i in range(nfull):
        _apply_chunk(x, out, SBs, SBt, r_start + i * CH, P, K, C)
    if kr > 0:
        _apply_chunk(x, out, SBs, SBt, r_start + nfull * CH, P, kr, C)
    if rr > 0:
        _apply_chunk(x, out, SBs, SBt, r_start + nfull * CH + kr * P, rr, 1, C)

    return out


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    global _LAST_CONFIG
    x = input.contiguous()
    Ni = int(N)
    Ci = int(C)
    K = _choose_k(Ni, Ci)
    out = _bn_kernel(x, gamma, beta, float(eps), K)
    rows_per_prog = Ni // 2
    _LAST_CONFIG = {
        "programs": 2,
        "rows_per_program": rows_per_prog,
        "K_rows_per_partition": K,
        "chunk_rows": 128 * K,
        "channel_block": 128,
        "stats": "tensorE_gram_diag_plus_ones",
        "apply": "vector_mul_gpsimd_vector_add",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
