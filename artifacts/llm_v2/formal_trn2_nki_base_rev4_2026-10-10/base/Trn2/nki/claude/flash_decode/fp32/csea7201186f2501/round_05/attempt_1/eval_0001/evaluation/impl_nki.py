import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Configuration literals
# ---------------------------------------------------------------------------
_PEN = 1.0e30       # penalty used to exclude invalid splits from the max
_GCAP = 16          # max partitions per (batch, head) pair (explicit source lists below)

_LAST_CONFIG = {}


def _choose_g(npc, S):
    # largest power of two G <= 16 with npc * G <= 128 and S % G == 0
    G = 1
    for _ in range(4):
        if G * 2 <= 16:
            if G * 2 * npc <= 128:
                if S % (G * 2) == 0:
                    G = G * 2
    return G


def _row(t, r, w):
    return t[r:r + 1, 0:w]


def _pair_reduce(FIN, RS, pl, b0, G, W):
    # FIN[pl, :] = sum_{j < G} RS[b0 + j, :]   (fp32, DMA engines, stays on chip)
    d = FIN[pl:pl + 1, 0:W]
    if G == 1:
        nisa.dma_copy(dst=d, src=_row(RS, b0, W))
    elif G == 2:
        nisa.dma_compute(dst=d,
                         srcs=[_row(RS, b0, W), _row(RS, b0 + 1, W)],
                         reduce_op=nl.add)
    elif G == 4:
        nisa.dma_compute(dst=d,
                         srcs=[_row(RS, b0, W), _row(RS, b0 + 1, W),
                               _row(RS, b0 + 2, W), _row(RS, b0 + 3, W)],
                         reduce_op=nl.add)
    elif G == 8:
        nisa.dma_compute(dst=d,
                         srcs=[_row(RS, b0, W), _row(RS, b0 + 1, W),
                               _row(RS, b0 + 2, W), _row(RS, b0 + 3, W),
                               _row(RS, b0 + 4, W), _row(RS, b0 + 5, W),
                               _row(RS, b0 + 6, W), _row(RS, b0 + 7, W)],
                         reduce_op=nl.add)
    else:
        nisa.dma_compute(dst=d,
                         srcs=[_row(RS, b0, W), _row(RS, b0 + 1, W),
                               _row(RS, b0 + 2, W), _row(RS, b0 + 3, W),
                               _row(RS, b0 + 4, W), _row(RS, b0 + 5, W),
                               _row(RS, b0 + 6, W), _row(RS, b0 + 7, W),
                               _row(RS, b0 + 8, W), _row(RS, b0 + 9, W),
                               _row(RS, b0 + 10, W), _row(RS, b0 + 11, W),
                               _row(RS, b0 + 12, W), _row(RS, b0 + 13, W),
                               _row(RS, b0 + 14, W), _row(RS, b0 + 15, W)],
                         reduce_op=nl.add)


def _combine_group(mid_o, mid_o_lse, b_seqlen, out, bs, g0, npc):
    B = mid_o.shape[0]
    H = mid_o.shape[1]
    S = mid_o.shape[2]
    D = mid_o.shape[3]
    BH = B * H
    G = _choose_g(npc, S)
    C = S // G
    NP = npc * G
    r0 = g0 * G
    W = D + 1

    midC = mid_o.reshape((BH * G, C * D))
    lseC = mid_o_lse.reshape((BH * G, C))
    lse2 = mid_o_lse.reshape((BH, S))
    bsl2 = b_seqlen.reshape((1, B))
    out2 = out.reshape((BH, D))

    # ---- 1. small loads first (lse rows replicated over the G partitions of a pair)
    LS = nl.ndarray((NP, S), dtype=nl.float32, buffer=nl.sbuf)
    for lp in range(npc):
        nisa.dma_copy(dst=LS[lp * G:(lp + 1) * G, 0:S],
                      src=lse2.ap(pattern=[[0, G], [1, S]], offset=(g0 + lp) * S))

    Li = nl.ndarray((NP, 1), dtype=nl.int32, buffer=nl.sbuf)
    la = 0
    while la < npc:
        b = (g0 + la) // H
        lq = min(npc, (b + 1) * H - g0)
        nisa.dma_copy(dst=Li[la * G:lq * G, 0:1],
                      src=bsl2.ap(pattern=[[0, (lq - la) * G], [1, 1]], offset=b))
        la = lq

    # own-chunk lse: partition p = pair_local*G + j holds splits j*C .. j*C+C-1
    LC = nl.ndarray((NP, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=LC, src=lseC[r0:r0 + NP, 0:C])

    # ---- 2. bulk load of partial outputs (one contiguous DMA per pair)
    M = nl.ndarray((NP, C * D), dtype=nl.float32, buffer=nl.sbuf)
    for lp in range(npc):
        nisa.dma_copy(dst=M[lp * G:(lp + 1) * G, 0:C * D],
                      src=midC[r0 + lp * G:r0 + (lp + 1) * G, 0:C * D])

    # ---- 3. per-partition sequence length and chunk threshold
    Lf = nl.ndarray((NP, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Lf, src=Li)
    Lm1 = nl.ndarray((NP, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=Lm1, data=Lf, op0=nl.subtract, operand0=1.0)

    pidx = nl.ndarray((NP, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pidx, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    jint = nl.ndarray((NP, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=jint, data=pidx, op0=nl.bitwise_and, operand0=G - 1)
    jf = nl.ndarray((NP, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=jf, src=jint)
    thr = nl.ndarray((NP, 1), dtype=nl.float32, buffer=nl.sbuf)
    # thr = L - j*C*bs ; local split c valid <=> thr - c*bs >= 1
    nisa.tensor_scalar(dst=thr, data=jf, op0=nl.multiply, operand0=float(-(C * bs)),
                       op1=nl.add, operand1=Lf)

    # ramp s * block_seq (exact integers)
    ioi = nl.ndarray((NP, S), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=ioi, pattern=[[bs, S]], offset=0, channel_multiplier=0)
    iof = nl.ndarray((NP, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=iof, src=ioi)

    # ---- 4. masked max over all valid splits of the pair (split s valid <=> s*bs < L)
    tF = nl.ndarray((NP, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tF, data=iof, op0=nl.multiply, operand0=-1.0,
                       op1=nl.add, operand1=Lm1)
    nisa.tensor_scalar(dst=tF, data=tF, op0=nl.minimum, operand0=0.0,
                       op1=nl.multiply, operand1=_PEN)
    nisa.tensor_tensor(dst=tF, data1=tF, data2=LS, op=nl.add)
    mx = nl.ndarray((NP, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=tF, axis=1)

    # ---- 5. own-chunk weights w = exp(lse - m) for valid splits, 0 otherwise
    tC = nl.ndarray((NP, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tC, data=iof[0:NP, 0:C], op0=nl.multiply, operand0=-1.0,
                       op1=nl.add, operand1=thr)
    nisa.tensor_scalar(dst=tC, data=tC, op0=nl.maximum, operand0=0.0,
                       op1=nl.minimum, operand1=1.0)
    xC = nl.ndarray((NP, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=xC, data=LC, op0=nl.subtract, operand0=mx,
                       op1=nl.minimum, operand1=0.0)
    wC = nl.ndarray((NP, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=wC, op=nl.exp, data=xC)
    nisa.tensor_tensor(dst=wC, data1=wC, data2=tC, op=nl.multiply)

    # RS[:, 0:D] = partial numerator, RS[:, D] = partial denominator (fp32)
    RS = nl.ndarray((NP, W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=RS[0:NP, D:D + 1], op=nl.add, data=wC, axis=1)

    # ---- 6. weighted partial numerator per partition
    if C == 1:
        nisa.tensor_scalar(dst=RS[0:NP, 0:D], data=M[0:NP, 0:D],
                           op0=nl.multiply, operand0=wC[0:NP, 0:1])
    else:
        for c in range(C):
            nisa.tensor_scalar(dst=M[0:NP, c * D:(c + 1) * D], data=M[0:NP, c * D:(c + 1) * D],
                               op0=nl.multiply, operand0=wC[0:NP, c:c + 1])
        w = C
        while w > 2:
            h = w // 2
            nisa.tensor_tensor(dst=M[0:NP, 0:h * D], data1=M[0:NP, 0:h * D],
                               data2=M[0:NP, (w - h) * D:w * D], op=nl.add)
            w = w - h
        nisa.tensor_tensor(dst=RS[0:NP, 0:D], data1=M[0:NP, 0:D],
                           data2=M[0:NP, D:2 * D], op=nl.add)

    # ---- 7. combine the G partials of each pair on chip (fp32 DMA add)
    FIN = nl.ndarray((npc, W), dtype=nl.float32, buffer=nl.sbuf)
    for lp in range(npc):
        _pair_reduce(FIN, RS, lp, lp * G, G, W)

    # ---- 8. single final division and store
    ones = nl.ndarray((npc, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    denB = nl.ndarray((npc, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=denB, data=ones, op0=nl.multiply, operand0=FIN[0:npc, D:D + 1])
    o_sb = nl.ndarray((npc, D), dtype=mid_o.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=o_sb, data1=FIN[0:npc, 0:D], data2=denB, op=nl.divide)
    nisa.dma_copy(dst=out2[g0:g0 + npc, 0:D], src=o_sb)


@nki.jit
def _fd_kernel(mid_o, mid_o_lse, b_seqlen, bs):
    B = mid_o.shape[0]
    H = mid_o.shape[1]
    D = mid_o.shape[3]
    BH = B * H
    out = nl.ndarray((B, H, D), dtype=mid_o.dtype, buffer=nl.shared_hbm)
    nprog = nl.num_programs()
    pid = nl.program_id(0)
    ppp = (BH + nprog - 1) // nprog
    p_lo = pid * ppp
    p_hi = min(BH, p_lo + ppp)
    npg = min(ppp, 128)
    if p_hi > p_lo:
        ngroups = (p_hi - p_lo + npg - 1) // npg
        for gi in range(ngroups):
            g0 = p_lo + gi * npg
            npc = min(npg, p_hi - g0)
            _combine_group(mid_o, mid_o_lse, b_seqlen, out, bs, g0, npc)
    return out


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    global _LAST_CONFIG
    if isinstance(block_seq, torch.Tensor):
        bs = int(block_seq.item())
    else:
        bs = int(block_seq)
    B, H, S, D = mid_o.shape
    out = _fd_kernel(mid_o, mid_o_lse, b_seqlen, bs)

    BH = int(B) * int(H)
    ppp = (BH + 1) // 2
    npg = min(ppp, 128)
    G = _choose_g(npg, int(S))
    C = int(S) // G
    _LAST_CONFIG = {
        "lnc_programs": 2,
        "pairs_per_program": int(ppp),
        "pairs_per_group": int(npg),
        "partitions_per_pair": int(G),
        "splits_per_partition": int(C),
        "block_seq": int(bs),
        "max_penalty": _PEN,
        "cross_partition_combine": "dma_compute_fp32",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
