import os
import re

import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None
    PMAX = 128

# Deliberately finite, not -inf: a row masked everywhere then yields
# exp(NEG_INF - NEG_INF) = 1 instead of NaN. Never happens for causal attention (every
# row attends at least to itself) but kept for the non-causal empty-tail edge case.
NEG_INF = -3.0e38

# Fewer key blocks than this and the on-device loop is not worth its overhead, so
# they are unrolled at compile time instead.
MIN_DYNAMIC_ITERS = 3


def _lnc_degree() -> int:
    """Logical-NeuronCore degree the kernel must be launched with.

    The kernel contains on-device control flow (``nl.dynamic_range``), which the
    backend only lowers correctly when the NKI launch degree matches the LNC the XLA
    module is compiled for -- launching an LNC=1 kernel into an LNC=2 module fails with
    ``[NCC_IXGM002] ... core 1 has 1 basic blocks``. trn2/trn3 default to LNC=2 unless
    the compiler/runtime env says otherwise.
    """
    explicit = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "")
    if explicit.strip().isdigit():
        return int(explicit.strip())
    match = re.search(r"--lnc[=\s]+(\d+)", os.environ.get("NEURON_CC_FLAGS", ""))
    if match:
        return int(match.group(1))
    target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", "").strip().lower()
    return 2 if target in ("trn2", "gen3", "trn3", "gen4") else 1


def div_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


if nki is not None:
    def _attend_block(K, V, buf, cfg, q_offset, q_size, k_offset, k_size,
                      apply_causal_mask, k_off_sb=None):
        """Fold one key/value block into the running softmax state.

        ``apply_causal_mask`` is a compile-time Python bool: True only for the diagonal
        block, where a genuine triangular mask is needed. Every other visited block is
        fully valid by construction (see module docstring), so the mask machinery
        block_sparse_attention needs (a whole HBM mask tensor) doesn't exist here --
        just an on-device iota comparison, and only for the one block that needs it.
        """
        head_dim = cfg["head_dim"]
        k_tile, v_tile = buf["k_tile"], buf["v_tile"]

        if k_off_sb is not None:
            nisa.dma_copy(
                dst=k_tile[0:k_size, 0:head_dim],
                src=K.ap(pattern=[[head_dim, k_size], [1, head_dim]], scalar_offset=k_off_sb, indirect_dim=0),
            )
            nisa.dma_copy(
                dst=v_tile[0:k_size, 0:head_dim],
                src=V.ap(pattern=[[head_dim, k_size], [1, head_dim]], scalar_offset=k_off_sb, indirect_dim=0),
            )
        else:
            nisa.dma_copy(dst=k_tile[0:k_size, 0:head_dim], src=K[k_offset:k_offset + k_size, 0:head_dim])
            nisa.dma_copy(dst=v_tile[0:k_size, 0:head_dim], src=V[k_offset:k_offset + k_size, 0:head_dim])

        kt_psum = nl.ndarray((head_dim, k_size), dtype=cfg["dtype"], buffer=nl.psum)
        nisa.nc_transpose(dst=kt_psum, data=k_tile[0:k_size, 0:head_dim])
        kt_sb = buf["kt_sb"]
        nisa.tensor_copy(dst=kt_sb[0:head_dim, 0:k_size], src=kt_psum)

        s_psum = nl.ndarray((q_size, k_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=s_psum, stationary=buf["qt_sb"][0:head_dim, 0:q_size], moving=kt_sb[0:head_dim, 0:k_size])
        s_sb = buf["s_sb"]

        if apply_causal_mask:
            nisa.tensor_tensor(dst=s_sb[0:q_size, 0:k_size], data1=s_psum,
                               data2=buf["causal_bias"][0:q_size, 0:k_size], op=nl.add)
        else:
            nisa.tensor_copy(dst=s_sb[0:q_size, 0:k_size], src=s_psum)

        # ---- online softmax update -------------------------------------
        m_prev, m_new = buf["m_prev"], buf["m_new"]
        neg_m, corr = buf["neg_m"], buf["corr"]
        row_sum, l_acc, o_acc = buf["row_sum"], buf["l_acc"], buf["o_acc"]

        nisa.tensor_reduce(dst=m_new[0:q_size, 0:1], data=s_sb[0:q_size, 0:k_size], op=nl.maximum, axis=(1,))
        nisa.tensor_tensor(dst=m_new[0:q_size, 0:1], data1=m_new[0:q_size, 0:1], data2=m_prev[0:q_size, 0:1], op=nl.maximum)
        nisa.tensor_scalar(dst=neg_m[0:q_size, 0:1], data=m_new[0:q_size, 0:1], op0=nl.multiply, operand0=-1.0)
        
        nisa.activation(dst=corr[0:q_size, 0:1], data=m_prev[0:q_size, 0:1], op=nl.exp, bias=neg_m[0:q_size, 0:1])
        nisa.tensor_copy(dst=m_prev[0:q_size, 0:1], src=m_new[0:q_size, 0:1])

        p = buf["p"]
        nisa.activation(dst=p[0:q_size, 0:k_size], data=s_sb[0:q_size, 0:k_size], op=nl.exp, bias=neg_m[0:q_size, 0:1])
        nisa.tensor_reduce(dst=row_sum[0:q_size, 0:1], data=p[0:q_size, 0:k_size], op=nl.add, axis=(1,))
        nisa.scalar_tensor_tensor(dst=l_acc[0:q_size, 0:1], data=l_acc[0:q_size, 0:1], op0=nl.multiply, operand0=corr[0:q_size, 0:1], op1=nl.add, operand1=row_sum[0:q_size, 0:1])

        pt_psum = nl.ndarray((k_size, q_size), dtype=cfg["dtype"], buffer=nl.psum)
        nisa.nc_transpose(dst=pt_psum, data=p[0:q_size, 0:k_size])
        pt_sb = buf["pt_sb"]
        nisa.tensor_copy(dst=pt_sb[0:k_size, 0:q_size], src=pt_psum)

        pv_psum = nl.ndarray((q_size, head_dim), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=pv_psum, stationary=pt_sb[0:k_size, 0:q_size], moving=v_tile[0:k_size, 0:head_dim])
        nisa.scalar_tensor_tensor(dst=o_acc[0:q_size, 0:head_dim], data=o_acc[0:q_size, 0:head_dim], op0=nl.multiply, operand0=corr[0:q_size, 0:1], op1=nl.add, operand1=pv_psum)

    @nki.jit
    def flash_attention_kernel(Q, K, V, causal, scale):
        seq_len, head_dim = Q.shape
        assert head_dim <= PMAX

        num_q_blocks = div_ceil(seq_len, PMAX)
        num_kv_blocks = num_q_blocks

        cfg = {"head_dim": head_dim, "dtype": Q.dtype}

        out = nl.ndarray((seq_len, head_dim), dtype=Q.dtype, buffer=nl.shared_hbm)

        buf = {
            "q_tile": nl.ndarray((PMAX, head_dim), dtype=Q.dtype, buffer=nl.sbuf),
            "qt_sb": nl.ndarray((head_dim, PMAX), dtype=Q.dtype, buffer=nl.sbuf),
            "k_tile": nl.ndarray((PMAX, head_dim), dtype=Q.dtype, buffer=nl.sbuf),
            "kt_sb": nl.ndarray((head_dim, PMAX), dtype=Q.dtype, buffer=nl.sbuf),
            "v_tile": nl.ndarray((PMAX, head_dim), dtype=Q.dtype, buffer=nl.sbuf),
            "s_sb": nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf),
            "p": nl.ndarray((PMAX, PMAX), dtype=Q.dtype, buffer=nl.sbuf),
            "pt_sb": nl.ndarray((PMAX, PMAX), dtype=Q.dtype, buffer=nl.sbuf),
            "m_prev": nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf),
            "m_new": nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf),
            "neg_m": nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf),
            "corr": nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf),
            "row_sum": nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf),
            "l_acc": nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf),
            "o_acc": nl.ndarray((PMAX, head_dim), dtype=nl.float32, buffer=nl.sbuf),
            "res": nl.ndarray((PMAX, head_dim), dtype=Q.dtype, buffer=nl.sbuf),
        }
        if causal:
            buf["causal_bias"] = nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf)

        # The diagonal-block causal mask -- and the additive {0, NEG_INF} bias derived
        # from it -- is the same (PMAX, PMAX) local row >= col triangle for every qi
        # (see `_attend_block`), so it's built once here instead of once per unrolled
        # qi iteration. `causal` is a compile-time bool, so this whole block (and the
        # buffer above) are simply absent from the non-causal trace.
        if causal:
            q_local = nl.ndarray((PMAX, PMAX), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=q_local, pattern=[[0, PMAX]], offset=0, channel_multiplier=1)
            k_local = nl.ndarray((PMAX, PMAX), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=k_local, pattern=[[1, PMAX]], offset=0, channel_multiplier=0)
            local_ok = nl.greater_equal(q_local, k_local)
            # {0, 1} -> {0, NEG_INF}, the same additive-bias technique
            # block_sparse_attention uses for its mask.
            nisa.tensor_copy(dst=buf["causal_bias"], src=local_ok)
            nisa.tensor_scalar(dst=buf["causal_bias"], data=buf["causal_bias"], op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=-NEG_INF)

        for qi in range(num_q_blocks):
            q_offset = qi * PMAX
            q_size = min(PMAX, seq_len - q_offset)

            nisa.dma_copy(dst=buf["q_tile"][0:q_size, 0:head_dim], src=Q[q_offset:q_offset + q_size, 0:head_dim])
            qt_psum = nl.ndarray((head_dim, q_size), dtype=Q.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=qt_psum, data=buf["q_tile"][0:q_size, 0:head_dim])
            nisa.tensor_scalar(dst=buf["qt_sb"][0:head_dim, 0:q_size], data=qt_psum, op0=nl.multiply, operand0=scale)

            nisa.memset(dst=buf["m_prev"][0:q_size, 0:1], value=NEG_INF)
            nisa.memset(dst=buf["l_acc"][0:q_size, 0:1], value=0.0)
            nisa.memset(dst=buf["o_acc"][0:q_size, 0:head_dim], value=0.0)

            # Every block up to and including qi (causal) or every block (non-causal).
            # The diagonal block for causal, or the last (possibly partial) kv block
            # for non-causal, is kept out of the dynamic loop -- see module docstring.
            n_kj = (qi + 1) if causal else num_kv_blocks
            n_full = n_kj if (not causal and seq_len % PMAX == 0) or causal else n_kj - 1
            # Under causal, the diagonal block always needs the triangular mask, so it
            # never joins the "fully valid" dynamic loop regardless of size.
            if causal:
                n_full -= 1
            n_dynamic = n_full if n_full >= MIN_DYNAMIC_ITERS else 0

            if n_dynamic > 0:
                k_off_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.memset(dst=k_off_sb, value=0)
                for _ in nl.dynamic_range(n_dynamic):
                    _attend_block(K, V, buf, cfg, q_offset, q_size, 0, PMAX, apply_causal_mask=False, k_off_sb=k_off_sb)
                    nisa.tensor_scalar(dst=k_off_sb, data=k_off_sb, op0=nl.add, operand0=PMAX)

            for kj in range(n_dynamic, n_kj):
                k_offset = kj * PMAX
                k_size = min(PMAX, seq_len - k_offset)
                is_diag = causal and kj == qi
                _attend_block(K, V, buf, cfg, q_offset, q_size, k_offset, k_size, apply_causal_mask=is_diag)

            nisa.reciprocal(dst=buf["corr"][0:q_size, 0:1], data=buf["l_acc"][0:q_size, 0:1])
            nisa.tensor_scalar(dst=buf["res"][0:q_size, 0:head_dim], data=buf["o_acc"][0:q_size, 0:head_dim], op0=nl.multiply, operand0=buf["corr"][0:q_size, 0:1])
            nisa.dma_copy(dst=out[q_offset:q_offset + q_size, 0:head_dim], src=buf["res"][0:q_size, 0:head_dim])

        return out


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    
    batch, n_heads, seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim ** 0.5)
    out = torch.empty_like(q)
    
    for b in range(batch):
        for h in range(n_heads):
            res = flash_attention_kernel[_lnc_degree()](
                q[b, h].contiguous(), k[b, h].contiguous(),
                v[b, h].contiguous(), causal, scale)
            out[b, h] = res
    return out


def get_last_config() -> dict | None:
    return None
