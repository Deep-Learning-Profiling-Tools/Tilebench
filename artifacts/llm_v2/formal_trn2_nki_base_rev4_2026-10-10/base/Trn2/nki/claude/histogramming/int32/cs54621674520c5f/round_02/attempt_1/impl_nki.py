import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Two-level privatised histogram on Trn2 (LNC=2):
#  stage 1: each program (physical core) zero-fills its own row of the
#           (P, num_bins) int32 scratch allocated in run().
#  stage 2: each program counts its contiguous share of the input.
#           Value v = hi*L + lo. On-chip pre-aggregation per chunk:
#             VectorE: one-hot of hi (0/1), ScalarE: step indicator of lo
#             s_b = sign(lo - b + 0.5) (+-1), TensorE: block-diagonal
#             products summed over <= 131072 elements per chunk, so every
#             PSUM value is an exactly representable small integer.
#           Each chunk result is moved losslessly into an int32 accumulator
#           (exact int32 GpSimd adds). The exact histogram row is recovered
#           with int32 first differences and written to the program's row.
#  stage 3: after a grid-wide core_barrier, each core sums the P private
#           rows for half of the bins (exact int32) into the output.
# Values outside [0, num_bins) produce hi >= H, hence no one-hot match:
# they are dropped.

_PART = 128
_NPROG = 2          # programs of one LNC=2 launch == number of partials P
_U = 2              # sub-chunks per loop iteration (overlap engines)
_SBUF_BUDGET = 160000
_last_config = {}


def _select_config(N, B):
    best = None
    for L in (8, 16, 32, 64):
        if B % L != 0:
            continue
        H = B // L
        if H > 128:
            continue
        key = (max(H + 2, L + 1), H + L, -L)
        if best is None or key < best[0]:
            best = (key, H, L)
    assert best is not None, "unsupported num_bins"
    H = best[1]
    L = best[2]
    gmax = min(128 // H, 512 // L)
    G = 1
    while G * 2 <= gmax:
        G *= 2
    E = N // (_PART * _NPROG)
    F = 512
    while F > G and (_U * F * (2 * (H + L) + 16) + 4 * _U * F > _SBUF_BUDGET
                     or E % (_U * F) != 0):
        F //= 2
    assert E % (_U * F) == 0
    sh = 0
    while (1 << sh) < L:
        sh += 1
    return {
        "P": _NPROG, "H": H, "L": L, "G": G, "F": F, "U": _U,
        "shift": sh, "iters_per_program": E // (_U * F),
    }


@nki.jit
def _hist_kernel(inp, scratch, H, L, G, F, U, sh):
    B = H * L
    npg = nl.num_programs()
    pid = nl.program_id(0)
    N = inp.shape[0]
    E = N // (128 * npg)
    CW = U * F
    I = E // CW
    GH = G * H
    GL = G * L
    T = F // G
    FH = F * H
    FL = F * L

    out = nl.ndarray((B,), dtype=nl.int32, buffer=nl.shared_hbm)
    rows = scratch.reshape((npg, H, L))

    # stage 1: owning program zero-fills its private row
    zt = nl.ndarray((H, L), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=zt, value=0)
    nisa.dma_copy(dst=rows[pid], src=zt)

    # private int32 accumulator (block matrix, diagonal blocks are used)
    acc = nl.ndarray((GH, GL), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0)

    # per-column bias for the step indicator: sign(64*lo + 32 - 64*b)
    bias = nl.ndarray((128, L), dtype=nl.float32, buffer=nl.sbuf)
    for b in range(L):
        nisa.memset(dst=bias[:, b:b + 1], value=32.0 - 64.0 * b)

    x3 = inp.reshape((npg * I, 128, CW))

    def body(j):
        xt = nl.ndarray((128, CW), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x3[j])
        ps = nl.ndarray((GH, GL), dtype=nl.float32, buffer=nl.psum)
        for u in range(U):
            xs = xt[:, u * F:(u + 1) * F]
            hi_i = nl.ndarray((128, F), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=hi_i, data=xs, op0=nl.right_shift, operand0=sh)
            lo_i = nl.ndarray((128, F), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=lo_i, data=xs, op0=nl.bitwise_and, operand0=L - 1)
            hb = nl.ndarray((128, F), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=hb, src=hi_i)
            lb = nl.ndarray((128, F), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=lb, src=lo_i)
            ohh = nl.ndarray((128, FH), dtype=nl.bfloat16, buffer=nl.sbuf)
            for a in range(H):
                nisa.tensor_scalar(dst=ohh[:, a:FH:H], data=hb, op0=nl.equal,
                                   operand0=float(a), engine=nisa.engine.vector)
            ohl = nl.ndarray((128, FL), dtype=nl.bfloat16, buffer=nl.sbuf)
            for b in range(L):
                nisa.activation(dst=ohl[:, b:FL:L], op=nl.sign, data=lb,
                                bias=bias[:, b:b + 1], scale=64.0)
            for t in range(T):
                nisa.nc_matmul(dst=ps,
                               stationary=ohh[:, t * GH:(t + 1) * GH],
                               moving=ohl[:, t * GL:(t + 1) * GL],
                               accumulate=((u + t) > 0))
        pc = nl.ndarray((GH, GL), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pc, src=ps, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=acc, data1=acc, data2=pc, op=nl.add)

    nl.fori_loop(pid * I, (pid + 1) * I, body)

    # sum the diagonal blocks (exact int32)
    R = nl.ndarray((H, GL), dtype=nl.int32, buffer=nl.sbuf)
    for g in range(G):
        nisa.dma_copy(dst=R[:, g * L:(g + 1) * L],
                      src=acc[g * H:(g + 1) * H, g * L:(g + 1) * L])
    for g in range(1, G):
        nisa.tensor_tensor(dst=R[:, 0:L], data1=R[:, 0:L],
                           data2=R[:, g * L:(g + 1) * L], op=nl.add)
    # hist[a,b] = (S[a,b] - S[a,b+1]) / 2 ; hist[a,L-1] = (S[a,L-1] + S[a,0]) / 2
    hh = nl.ndarray((H, L), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=hh[:, 0:L - 1], data1=R[:, 0:L - 1], data2=R[:, 1:L],
                       op=nl.subtract)
    nisa.tensor_tensor(dst=hh[:, L - 1:L], data1=R[:, L - 1:L], data2=R[:, 0:1],
                       op=nl.add)
    nisa.tensor_scalar(dst=hh, data=hh, op0=nl.right_shift, operand0=1)
    # row was zero-filled; row = 0 + private partial
    nisa.dma_copy(dst=rows[pid], src=hh)

    # stage 3: grid-wide barrier, then sum the P private rows
    if npg > 1:
        nisa.core_barrier(data=scratch, cores=(0, 1))
    half = B // npg
    RR = half // 32
    sv = scratch.reshape((npg, npg, 32, RR))
    ov = out.reshape((npg, 32, RR))
    tot = nl.ndarray((32, RR), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tot, src=sv[0, pid])
    for r in range(1, npg):
        oth = nl.ndarray((32, RR), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=oth, src=sv[r, pid])
        nisa.tensor_tensor(dst=tot, data1=tot, data2=oth, op=nl.add)
    nisa.dma_copy(dst=ov[pid], src=tot)
    return out


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    N = int(N)
    num_bins = int(num_bins)
    cfg = _select_config(N, num_bins)
    _last_config.clear()
    _last_config.update(cfg)
    scratch = torch.empty((cfg["P"], num_bins), dtype=torch.int32, device=input.device)
    out = _hist_kernel(input, scratch, cfg["H"], cfg["L"], cfg["G"], cfg["F"],
                       cfg["U"], cfg["shift"])
    return out


def get_last_config() -> dict:
    return dict(_last_config)
