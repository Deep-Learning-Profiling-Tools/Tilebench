import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Configuration literals
# ---------------------------------------------------------------------------
_NKB = 4            # number of DMA blocks used to load the partial outputs of a group
_PEN = 1.0e30       # penalty used to exclude invalid splits from the max

_LAST_CONFIG = {}


def _choose_g(npc, S):
    # largest power of two G with npc * G <= 128 and S % G == 0
    G = 1
    for _ in range(7):
        if G * 2 * npc <= 128:
            if S % (G * 2) == 0:
                G = G * 2
    return G


def _combine_group(mid_o, mid_o_lse, b_seqlen, out, bs, nkb, g0, npc):
    B = mid_o.shape[0]
    H = mid_o.shape[1]
    S = mid_o.shape[2]
    D = mid_o.shape[3]
    BH = B * H
    G = _choose_g(npc, S)
    C = S // G
    NP = npc * G
    r0 = g0 * G

    midC = mid_o.reshape((BH * G, C * D))
    lseC = mid_o_lse.reshape((BH * G, C))
    lse2 = mid_o_lse.reshape((BH, S))
    bsl2 = b_seqlen.reshape((1, B))
    out2 = out.reshape((BH, D))

    # ---- 1. bulk load of partial outputs: partition p = pair_local*G + j holds
    #         splits j*C .. j*C+C-1 of its pair (contiguous in HBM)
    M = nl.ndarray((NP, C * D), dtype=nl.float32, buffer=nl.sbuf)
    kb = min(nkb, C)
    for k in range(kb):
        c0 = (k * C) // kb
        c1 = ((k + 1) * C) // kb
        nisa.dma_copy(dst=M[0:NP, c0 * D:c1 * D], src=midC[r0:r0 + NP, c0 * D:c1 * D])

    # ---- 2. chunk lse and full-row lse (replicated over the G partitions of a pair)
    LC = nl.ndarray((NP, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=LC, src=lseC[r0:r0 + NP, 0:C])
    LS = nl.ndarray((NP, S), dtype=nl.float32, buffer=nl.sbuf)
    for lp in range(npc):
        nisa.dma_copy(dst=LS[lp * G:(lp + 1) * G, 0:S],
                      src=lse2.ap(pattern=[[0, G], [1, S]], offset=(g0 + lp) * S))

    # ---- 3. per-partition sequence length L = b_seqlen[batch of its pair]
    Li = nl.ndarray((NP, 1), dtype=nl.int32, buffer=nl.sbuf)
    la = 0
    while la < npc:
        b = (g0 + la) // H
        lq = min(npc, (b + 1) * H - g0)
        nisa.dma_copy(dst=Li[la * G:lq * G, 0:1],
                      src=bsl2.ap(pattern=[[0, (lq - la) * G], [1, 1]], offset=b))
        la = lq
    Lf = nl.ndarray((NP, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Lf, src=Li)
    Lm1 = nl.ndarray((NP, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=Lm1, data=Lf, op0=nl.subtract, operand0=1.0)

    # j = p mod G (G power of two), thr = L - j*C*bs
    pidx = nl.ndarray((NP, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=pidx, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    jint = nl.ndarray((NP, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=jint, data=pidx, op0=nl.bitwise_and, operand0=G - 1)
    jf = nl.ndarray((NP, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=jf, src=jint)
    thr = nl.ndarray((NP, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=thr, data=jf, op0=nl.multiply, operand0=float(-(C * bs)),
                       op1=nl.add, operand1=Lf)

    # ramp s * block_seq (exact integers in fp32)
    iof = nl.ndarray((NP, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=iof, pattern=[[bs, S]], offset=0, channel_multiplier=0)

    # ---- 4. masked max over the valid splits (split s valid <=> s*bs < L)
    tF = nl.ndarray((NP, S), dtype=nl.float32, buffer=nl.sbuf)
    # tF = L - 1 - s*bs  (>= 0 valid, <= -1 invalid)
    nisa.tensor_scalar(dst=tF, data=iof, op0=nl.multiply, operand0=-1.0,
                       op1=nl.add, operand1=Lm1)
    # penalty: 0 for valid, <= -1e30 for invalid
    nisa.tensor_scalar(dst=tF, data=tF, op0=nl.minimum, operand0=0.0,
                       op1=nl.multiply, operand1=_PEN)
    nisa.tensor_tensor(dst=tF, data1=tF, data2=LS, op=nl.add)
    mx = nl.ndarray((NP, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx, op=nl.maximum, data=tF, axis=1)

    # ---- 5. chunk weights w = exp(lse - m) for valid splits, 0 otherwise
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
    dP = nl.ndarray((NP, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=dP, op=nl.add, data=wC, axis=1)

    # ---- 6. weighted partial numerator per partition (fp32)
    PR = nl.ndarray((NP, C * D), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(C):
        nisa.tensor_scalar(dst=PR[0:NP, c * D:(c + 1) * D], data=M[0:NP, c * D:(c + 1) * D],
                           op0=nl.multiply, operand0=wC[0:NP, c:c + 1])
    w = C
    while w > 1:
        h = w // 2
        nisa.tensor_tensor(dst=PR[0:NP, 0:h * D], data1=PR[0:NP, 0:h * D],
                           data2=PR[0:NP, (w - h) * D:w * D], op=nl.add)
        w = w - h

    # ---- 7. sum the G partials of each pair (bit-exact transposes + free-axis reduce)
    A_ps = nl.ndarray((D, NP), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=A_ps, data=PR[0:NP, 0:D], engine=nisa.engine.tensor)
    A_sb = nl.ndarray((D, NP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=A_sb, src=A_ps)
    numT = nl.ndarray((D, npc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=numT, op=nl.add, data=A_sb.reshape((D, npc, G)), axis=2)

    ones = nl.ndarray((NP, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    Dw = nl.ndarray((NP, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=Dw, data=ones, op0=nl.multiply, operand0=dP)
    D_ps = nl.ndarray((D, NP), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=D_ps, data=Dw, engine=nisa.engine.tensor)
    D_sb = nl.ndarray((D, NP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=D_sb, src=D_ps)
    denT = nl.ndarray((D, npc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=denT, op=nl.add, data=D_sb.reshape((D, npc, G)), axis=2)

    # ---- 8. single final division, transpose back, store
    oT = nl.ndarray((D, npc), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=oT, data1=numT, data2=denT, op=nl.divide)
    o_ps = nl.ndarray((npc, D), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=o_ps, data=oT, engine=nisa.engine.tensor)
    o_sb = nl.ndarray((npc, D), dtype=mid_o.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o_sb, src=o_ps)
    nisa.dma_copy(dst=out2[g0:g0 + npc, 0:D], src=o_sb)


@nki.jit
def _fd_kernel(mid_o, mid_o_lse, b_seqlen, bs, nkb):
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
            _combine_group(mid_o, mid_o_lse, b_seqlen, out, bs, nkb, g0, npc)
    return out


def run(mid_o, mid_o_lse, b_seqlen, block_seq, block_size=None):
    global _LAST_CONFIG
    if isinstance(block_seq, torch.Tensor):
        bs = int(block_seq.item())
    else:
        bs = int(block_seq)
    B, H, S, D = mid_o.shape
    out = _fd_kernel(mid_o, mid_o_lse, b_seqlen, bs, _NKB)

    BH = B * H
    ppp = (BH + 1) // 2
    npg = min(ppp, 128)
    G = _choose_g(npg, S)
    C = S // G
    _LAST_CONFIG = {
        "lnc_programs": 2,
        "pairs_per_program": int(ppp),
        "pairs_per_group": int(npg),
        "partitions_per_pair": int(G),
        "splits_per_partition": int(C),
        "mid_dma_blocks": int(min(_NKB, C)),
        "block_seq": int(bs),
        "max_penalty": _PEN,
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
