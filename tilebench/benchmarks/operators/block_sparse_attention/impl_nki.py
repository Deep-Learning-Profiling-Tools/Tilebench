import functools
import os
import re
import subprocess
from types import SimpleNamespace

import torch

from tilebench.core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None
    PMAX = 128

NEG_INF = -3.0e38


@functools.lru_cache(maxsize=1)
def _lnc_degree() -> int:
    explicit = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "")
    if explicit.strip().isdigit():
        return int(explicit.strip())
    match = re.search(r"--lnc[=\s]+(\d+)", os.environ.get("NEURON_CC_FLAGS", ""))
    if match:
        return int(match.group(1))
    try:
        out = subprocess.run(["neuron-ls"], capture_output=True, text=True, timeout=10).stdout
        lnc = re.search(r"logical-neuroncore-config:\s*(\d+)", out)
        if lnc:
            return int(lnc.group(1))
    except (OSError, subprocess.SubprocessError):
        pass
    return 1


def _div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


_PATTERNS: list = []

if nki is not None:
    def _attend(Kt, Vt, kv_row0, ks, buf, q_size, head_dim, dtype, bias):
        k_tile, v_tile = buf["k"], buf["v"]
        nisa.dma_copy(dst=k_tile[0:ks, 0:head_dim], src=Kt[kv_row0:kv_row0 + ks, 0:head_dim])
        nisa.dma_copy(dst=v_tile[0:ks, 0:head_dim], src=Vt[kv_row0:kv_row0 + ks, 0:head_dim])
        kt_ps = nl.ndarray((head_dim, ks), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=kt_ps, data=k_tile[0:ks, 0:head_dim])
        nisa.tensor_copy(dst=buf["kt"][0:head_dim, 0:ks], src=kt_ps)
        s_ps = nl.ndarray((q_size, ks), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=s_ps, stationary=buf["qt"][0:head_dim, 0:q_size],
                       moving=buf["kt"][0:head_dim, 0:ks])
        s_sb = buf["s"]
        if bias is not None:
            nisa.tensor_tensor(dst=s_sb[0:q_size, 0:ks], data1=s_ps, data2=bias[0:q_size, 0:ks],
                               op=nl.add)
        else:
            nisa.tensor_copy(dst=s_sb[0:q_size, 0:ks], src=s_ps)
        m_prev, m_new, neg_m, corr = buf["m_prev"], buf["m_new"], buf["neg_m"], buf["corr"]
        nisa.tensor_reduce(dst=m_new[0:q_size, 0:1], data=s_sb[0:q_size, 0:ks], op=nl.maximum,
                           axis=(1,))
        nisa.tensor_tensor(dst=m_new[0:q_size, 0:1], data1=m_new[0:q_size, 0:1],
                           data2=m_prev[0:q_size, 0:1], op=nl.maximum)
        nisa.tensor_scalar(dst=neg_m[0:q_size, 0:1], data=m_new[0:q_size, 0:1],
                           op0=nl.multiply, operand0=-1.0)
        nisa.activation(dst=corr[0:q_size, 0:1], data=m_prev[0:q_size, 0:1], op=nl.exp,
                        bias=neg_m[0:q_size, 0:1])
        nisa.tensor_copy(dst=m_prev[0:q_size, 0:1], src=m_new[0:q_size, 0:1])
        p = buf["p"]
        nisa.activation(dst=p[0:q_size, 0:ks], data=s_sb[0:q_size, 0:ks], op=nl.exp,
                        bias=neg_m[0:q_size, 0:1])
        nisa.tensor_reduce(dst=buf["rsum"][0:q_size, 0:1], data=p[0:q_size, 0:ks], op=nl.add,
                           axis=(1,))
        nisa.scalar_tensor_tensor(dst=buf["l"][0:q_size, 0:1], data=buf["l"][0:q_size, 0:1],
                                  op0=nl.multiply, operand0=corr[0:q_size, 0:1],
                                  op1=nl.add, operand1=buf["rsum"][0:q_size, 0:1])
        pt_ps = nl.ndarray((ks, q_size), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=pt_ps, data=p[0:q_size, 0:ks])
        nisa.tensor_copy(dst=buf["pt"][0:ks, 0:q_size], src=pt_ps)
        pv_ps = nl.ndarray((q_size, head_dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=pv_ps, stationary=buf["pt"][0:ks, 0:q_size],
                       moving=v_tile[0:ks, 0:head_dim])
        nisa.scalar_tensor_tensor(dst=buf["o"][0:q_size, 0:head_dim],
                                  data=buf["o"][0:q_size, 0:head_dim],
                                  op0=nl.multiply, operand0=corr[0:q_size, 0:1],
                                  op1=nl.add, operand1=pv_ps)

    @nki.jit
    def block_sparse_kernel(Q, K, V, n_heads, n_kv_heads, seq_q, seq_kv, scale, pattern_key, BM, BN):
        pattern = _PATTERNS[pattern_key]
        rows_q, head_dim = Q.shape
        dtype = Q.dtype
        out = nl.ndarray((rows_q, head_dim), dtype=dtype, buffer=nl.shared_hbm)
        n_bh = rows_q // seq_q
        groups = n_heads // n_kv_heads
        num_layout = len(pattern)
        n_qb = _div_ceil(seq_q, BM)

        buf = {
            "q": nl.ndarray((BM, head_dim), dtype=dtype, buffer=nl.sbuf),
            "qt": nl.ndarray((head_dim, BM), dtype=dtype, buffer=nl.sbuf),
            "k": nl.ndarray((BN, head_dim), dtype=dtype, buffer=nl.sbuf),
            "kt": nl.ndarray((head_dim, BN), dtype=dtype, buffer=nl.sbuf),
            "v": nl.ndarray((BN, head_dim), dtype=dtype, buffer=nl.sbuf),
            "s": nl.ndarray((BM, BN), dtype=nl.float32, buffer=nl.sbuf),
            "p": nl.ndarray((BM, BN), dtype=dtype, buffer=nl.sbuf),
            "pt": nl.ndarray((BN, BM), dtype=dtype, buffer=nl.sbuf),
            "m_prev": nl.ndarray((BM, 1), dtype=nl.float32, buffer=nl.sbuf),
            "m_new": nl.ndarray((BM, 1), dtype=nl.float32, buffer=nl.sbuf),
            "neg_m": nl.ndarray((BM, 1), dtype=nl.float32, buffer=nl.sbuf),
            "corr": nl.ndarray((BM, 1), dtype=nl.float32, buffer=nl.sbuf),
            "rsum": nl.ndarray((BM, 1), dtype=nl.float32, buffer=nl.sbuf),
            "l": nl.ndarray((BM, 1), dtype=nl.float32, buffer=nl.sbuf),
            "o": nl.ndarray((BM, head_dim), dtype=nl.float32, buffer=nl.sbuf),
            "res": nl.ndarray((BM, head_dim), dtype=dtype, buffer=nl.sbuf),
        }

        q_idx = nl.ndarray((BM, BN), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=q_idx, pattern=[[0, BN]], offset=0, channel_multiplier=1)
        k_idx = nl.ndarray((BM, BN), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=k_idx, pattern=[[1, BN]], offset=0, channel_multiplier=0)
        ok = nl.greater_equal(q_idx, k_idx)
        diag_bias = nl.ndarray((BM, BN), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=diag_bias, src=ok)
        nisa.tensor_scalar(dst=diag_bias, data=diag_bias, op0=nl.subtract, operand0=1.0,
                           op1=nl.multiply, operand1=-NEG_INF)

        n_prog = nl.num_programs()
        pid = nl.program_id(0)
        per_core = _div_ceil(n_bh, n_prog)
        p_lo = min(n_bh, pid * per_core)
        p_hi = min(n_bh, p_lo + per_core)

        for p in range(p_lo, p_hi):
            b = p // n_heads
            h = p % n_heads
            kv_p = b * n_kv_heads + h // groups
            q_base = p * seq_q
            kv_base = kv_p * seq_kv
            rows = pattern[h % num_layout]
            for r in range(n_qb):
                q0 = r * BM
                qs = min(BM, seq_q - q0)
                nisa.dma_copy(dst=buf["q"][0:qs, 0:head_dim],
                              src=Q[q_base + q0:q_base + q0 + qs, 0:head_dim])
                qt_ps = nl.ndarray((head_dim, qs), dtype=dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=qt_ps, data=buf["q"][0:qs, 0:head_dim])
                nisa.tensor_scalar(dst=buf["qt"][0:head_dim, 0:qs], data=qt_ps,
                                   op0=nl.multiply, operand0=scale)
                nisa.memset(dst=buf["m_prev"][0:qs, 0:1], value=NEG_INF)
                nisa.memset(dst=buf["l"][0:qs, 0:1], value=0.0)
                nisa.memset(dst=buf["o"][0:qs, 0:head_dim], value=0.0)

                cols = rows[r]
                for c in cols:
                    k0 = c * BN
                    if k0 < seq_kv and k0 <= q0 + qs - 1:
                        ks = min(BN, seq_kv - k0)
                        if k0 + ks - 1 > q0:
                            _attend(K, V, kv_base + k0, ks, buf, qs, head_dim, dtype, diag_bias)
                        else:
                            _attend(K, V, kv_base + k0, ks, buf, qs, head_dim, dtype, None)

                nisa.tensor_scalar(dst=buf["l"][0:qs, 0:1], data=buf["l"][0:qs, 0:1],
                                   op0=nl.maximum, operand0=1.0e-30)
                nisa.reciprocal(dst=buf["corr"][0:qs, 0:1], data=buf["l"][0:qs, 0:1])
                nisa.tensor_scalar(dst=buf["res"][0:qs, 0:head_dim], data=buf["o"][0:qs, 0:head_dim],
                                   op0=nl.multiply, operand0=buf["corr"][0:qs, 0:1])
                nisa.dma_copy(dst=out[q_base + q0:q_base + q0 + qs, 0:head_dim],
                              src=buf["res"][0:qs, 0:head_dim])
        return out


_pattern_cache = torch.utils.weak.WeakTensorKeyDictionary()
_kernel_cache: dict = {}
_tuner_cache: dict = {}
_last_autotune_config: dict = {}
_DEFAULT_CONFIG = SimpleNamespace(block_m=64, block_n=64)


def _pattern(row_idx, col_idx, row_stride_h, col_stride_h, num_layout, num_rows):
    cached = _pattern_cache.get(col_idx)
    if cached is not None:
        return cached
    r = row_idx.cpu().tolist()
    c = col_idx.cpu().tolist()
    layouts = []
    for lay in range(num_layout):
        rows = []
        for i in range(num_rows):
            s = r[lay * row_stride_h + i]
            e = r[lay * row_stride_h + i + 1]
            rows.append(tuple(c[lay * col_stride_h + l] for l in range(s, e)))
        layouts.append(tuple(rows))
    pat = tuple(layouts)
    _pattern_cache[col_idx] = pat
    return pat


def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads,
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS,
        block_size: int = None, autotune: bool = False, **kwargs):
    B, H, M, D = Q.shape
    _, H_kv, N, _ = K.shape
    if BLOCK_M > PMAX or BLOCK_N > PMAX or D > PMAX:
        raise NotImplementedError("block_sparse_attention NKI: BLOCK_M, BLOCK_N and D must be <= 128")
    pat = _pattern(layout_csr_row_indices, layout_csr_col_indices, layout_csr_row_stride_h,
                   layout_csr_col_stride_h, num_layout, _div_ceil(M, BLOCK_M))
    if BLOCK_M != BLOCK_N:
        raise NotImplementedError("block_sparse_attention NKI: BLOCK_M must equal BLOCK_N")
    pat_key = _kernel_cache.get(pat)
    if pat_key is None:
        pat_key = _kernel_cache[pat] = len(_PATTERNS)
        _PATTERNS.append(pat)
    kernel = block_sparse_kernel[_lnc_degree()]
    if autotune:
        _last_autotune_config.clear()
        _last_autotune_config.update({"block_m": BLOCK_M, "block_n": BLOCK_N})
    out = kernel(Q.reshape(B * H * M, D), K.reshape(B * H_kv * N, D), V.reshape(B * H_kv * N, D),
                 H, H_kv, M, N, float(softmax_scale), pat_key, BLOCK_M, BLOCK_N)
    return out.reshape(B, H, M, D)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
