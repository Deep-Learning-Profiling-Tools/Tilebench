import math
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

# Additive "not attended" score. Deliberately finite (not -inf): a row that is
# masked everywhere then yields exp(NEG_INF - NEG_INF) = 1 rather than NaN,
NEG_INF = -3.0e38

# Fewer key blocks than this and the on-device loop is not worth its overhead,
# so they are unrolled at compile time instead.
MIN_DYNAMIC_ITERS = 3


def _lnc_degree() -> int:
    """Logical-NeuronCore degree the kernel must be launched with.

    The kernel contains on-device control flow (``nl.dynamic_range``), which the
    backend only lowers correctly when the NKI launch degree matches the LNC the
    XLA module is compiled for -- launching an LNC=1 kernel into an LNC=2 module
    fails with ``[NCC_IXGM002] ... core 1 has 1 basic blocks``.  trn2/trn3
    default to LNC=2 unless the compiler/runtime env says otherwise.
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


def kernel_assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"[block_sparse_attention NKI] {message}")


if nki is not None:

    def _attend_block(K, V, mask, buf, cfg, q_offset, q_size, k_offset, k_size,
                      k_off_sb=None):
        """Fold one key/value block into the running softmax state.

        ``k_off_sb`` (a 1x1 int32 SBUF scalar holding the first key row of the
        block) selects the dynamic path used inside ``nl.dynamic_range``; on the
        static path ``k_offset`` is a plain Python int.  Every other quantity is
        a compile-time constant, so both paths emit the same instructions.
        """
        head_dim = cfg["head_dim"]
        mask_cols = cfg["mask_cols"]

        k_tile, v_tile, m_tile = buf["k_tile"], buf["v_tile"], buf["m_tile"]

        if k_off_sb != None:
            # A dynamic_range register supports no arithmetic, so the key-block
            # position is carried in an SBUF scalar (in key *rows*) and applied
            # with .ap(scalar_offset=..., indirect_dim=...).
            nisa.dma_copy(
                dst=k_tile[0:k_size, 0:head_dim],
                src=K.ap(pattern=[[head_dim, k_size], [1, head_dim]], scalar_offset=k_off_sb, indirect_dim=0),
            )
            nisa.dma_copy(
                dst=v_tile[0:k_size, 0:head_dim],
                src=V.ap(pattern=[[head_dim, k_size], [1, head_dim]], scalar_offset=k_off_sb, indirect_dim=0),
            )
            nisa.dma_copy(
                dst=m_tile[0:q_size, 0:k_size],
                src=mask.ap(pattern=[[mask_cols, q_size], [1, k_size]], offset=q_offset * mask_cols, scalar_offset=k_off_sb, indirect_dim=1),
            )
        else:
            nisa.dma_copy(dst=k_tile[0:k_size, 0:head_dim], src=K[k_offset:k_offset + k_size, 0:head_dim])
            nisa.dma_copy(dst=v_tile[0:k_size, 0:head_dim], src=V[k_offset:k_offset + k_size, 0:head_dim])
            nisa.dma_copy(dst=m_tile[0:q_size, 0:k_size], src=mask[q_offset:q_offset + q_size, k_offset:k_offset + k_size])

        # scores = (Q * scale) @ K^T.  nc_matmul contracts along the partition
        # axis, so both operands need head_dim on partitions: Q^T is hoisted out
        # of this loop, K^T is transposed on-chip per block.
        kt_psum = nl.ndarray((head_dim, k_size), dtype=cfg["dtype"], buffer=nl.psum)
        nisa.nc_transpose(dst=kt_psum, data=k_tile[0:k_size, 0:head_dim])
        kt_sb = buf["kt_sb"]
        nisa.tensor_copy(dst=kt_sb[0:head_dim, 0:k_size], src=kt_psum)

        s_psum = nl.ndarray((q_size, k_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=s_psum, stationary=buf["qt_sb"][0:head_dim, 0:q_size], moving=kt_sb[0:head_dim, 0:k_size])

        # Turn the {0, 1} mask into an additive {0, NEG_INF} bias (in place),
        # then apply it -- same shape as the reference's mask addition.
        nisa.tensor_scalar(dst=m_tile[0:q_size, 0:k_size], data=m_tile[0:q_size, 0:k_size], op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=-NEG_INF)
        s_sb = buf["s_sb"]
        nisa.tensor_tensor(dst=s_sb[0:q_size, 0:k_size], data1=s_psum, data2=m_tile[0:q_size, 0:k_size], op=nl.add)

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
    def block_sparse_kernel(Q, K, V, mask, scale):
        seq_len, head_dim = Q.shape
        kv_len = K.shape[0]
        mask_cols = mask.shape[1]

        assert head_dim <= PMAX
        assert V.shape[0] == kv_len

        num_q_blocks = div_ceil(seq_len, PMAX)
        num_kv_blocks = div_ceil(kv_len, PMAX)

        cfg = {"head_dim": head_dim, "mask_cols": mask_cols, "dtype": Q.dtype}

        out = nl.ndarray((seq_len, head_dim), dtype=Q.dtype, buffer=nl.shared_hbm)

        # One set of buffers reused by every (qi, kj) pair: allocating inside the
        # loops would multiply both SBUF pressure and compile time.
        buf = {
            "q_tile": nl.ndarray((PMAX, head_dim), dtype=Q.dtype, buffer=nl.sbuf),
            "qt_sb": nl.ndarray((head_dim, PMAX), dtype=Q.dtype, buffer=nl.sbuf),
            "k_tile": nl.ndarray((PMAX, head_dim), dtype=Q.dtype, buffer=nl.sbuf),
            "kt_sb": nl.ndarray((head_dim, PMAX), dtype=Q.dtype, buffer=nl.sbuf),
            "v_tile": nl.ndarray((PMAX, head_dim), dtype=Q.dtype, buffer=nl.sbuf),
            "m_tile": nl.ndarray((PMAX, PMAX), dtype=nl.float32, buffer=nl.sbuf),
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

        for qi in range(num_q_blocks):
            q_offset = qi * PMAX
            q_size = min(PMAX, seq_len - q_offset)

            nisa.dma_copy(dst=buf["q_tile"][0:q_size, 0:head_dim], src=Q[q_offset:q_offset + q_size, 0:head_dim])
            qt_psum = nl.ndarray((head_dim, q_size), dtype=Q.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=qt_psum, data=buf["q_tile"][0:q_size, 0:head_dim])
            # Fold the softmax scale into Q^T once instead of scaling every
            # 128x128 score tile.
            nisa.tensor_scalar(dst=buf["qt_sb"][0:head_dim, 0:q_size], data=qt_psum, op0=nl.multiply, operand0=scale)

            nisa.memset(dst=buf["m_prev"][0:q_size, 0:1], value=NEG_INF)
            nisa.memset(dst=buf["l_acc"][0:q_size, 0:1], value=0.0)
            nisa.memset(dst=buf["o_acc"][0:q_size, 0:head_dim], value=0.0)

            # Causality (always present in the mask) bounds the key blocks: the
            # last query row of this tile is q_offset + q_size - 1.
            n_kj = min(div_ceil(q_offset + q_size, PMAX), num_kv_blocks)
            # Only the very last key block can be partial; keep it off the
            # on-device loop so every dynamic iteration is identical.
            n_full = n_kj if kv_len % PMAX == 0 or n_kj < num_kv_blocks else n_kj - 1
            n_dynamic = n_full if n_full >= MIN_DYNAMIC_ITERS else 0

            if n_dynamic > 0:
                k_off_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.memset(dst=k_off_sb, value=0)
                for _ in nl.dynamic_range(n_dynamic):
                    _attend_block(K, V, mask, buf, cfg, q_offset, q_size, 0, PMAX, k_off_sb=k_off_sb)
                    nisa.tensor_scalar(dst=k_off_sb, data=k_off_sb, op0=nl.add, operand0=PMAX)

            for kj in range(n_dynamic, n_kj):
                k_offset = kj * PMAX
                _attend_block(K, V, mask, buf, cfg, q_offset, q_size, k_offset, min(PMAX, kv_len - k_offset))

            nisa.reciprocal(dst=buf["corr"][0:q_size, 0:1], data=buf["l_acc"][0:q_size, 0:1])
            nisa.tensor_scalar(dst=buf["res"][0:q_size, 0:head_dim], data=buf["o_acc"][0:q_size, 0:head_dim], op0=nl.multiply, operand0=buf["corr"][0:q_size, 0:1])
            nisa.dma_copy(dst=out[q_offset:q_offset + q_size, 0:head_dim], src=buf["res"][0:q_size, 0:head_dim])

        return out


def _build_dense_mask(H, M, N, layout_csr_row_indices, layout_csr_col_indices, layout_csr_row_stride_h, layout_csr_col_stride_h, num_layout, BLOCK_M, BLOCK_N, device):
    """Mirrors impl_torch.py's CSR decode + causal-triangle combination."""
    num_rows = math.ceil(M / BLOCK_M)
    num_cols = math.ceil(N / BLOCK_N)

    sparse_mask = torch.zeros((H, M, N), dtype=torch.bool, device=device)
    for h in range(H):
        layout_h = h % num_layout
        for r in range(num_rows):
            start_l = layout_csr_row_indices[layout_h * layout_csr_row_stride_h + r].item()
            end_l = layout_csr_row_indices[layout_h * layout_csr_row_stride_h + r + 1].item()
            for l in range(start_l, end_l):
                c = layout_csr_col_indices[layout_h * layout_csr_col_stride_h + l].item()
                r_start, r_end = r * BLOCK_M, min((r + 1) * BLOCK_M, M)
                c_start, c_end = c * BLOCK_N, min((c + 1) * BLOCK_N, N)
                sparse_mask[h, r_start:r_end, c_start:c_end] = True

    causal_mask = torch.tril(torch.ones(M, N, dtype=torch.bool, device=device))
    return sparse_mask & causal_mask.unsqueeze(0)


def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads,
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS,
        block_size: int = None, autotune: bool = False, **kwargs):
    
    B, H, M, D = Q.shape
    _, H_kv, N, _ = K.shape
    head_groups = H // H_kv

    # The mask is built on the *host*, then transferred once.  Building it on the
    # XLA device instead puts one dynamic-update-slice per CSR block (thousands
    # of them, each over a multi-hundred-MB tensor) plus a full-size bool->fp32
    # cast into the same XLA module as the kernel, which is what actually blows
    # past the backend's instruction budget ([NCC_EXTP004]) at long sequences.
    # The decode logic itself is unchanged -- only where it executes.
    row_idx = layout_csr_row_indices.cpu()
    col_idx = layout_csr_col_indices.cpu()
    dense_mask = _build_dense_mask(H, M, N, row_idx, col_idx, layout_csr_row_stride_h, layout_csr_col_stride_h, num_layout, BLOCK_M, BLOCK_N, "cpu")

    Mp = ((M + PMAX - 1) // PMAX) * PMAX
    Np = ((N + PMAX - 1) // PMAX) * PMAX
    mask_padded = torch.zeros(H, Mp, Np, dtype=torch.float32, device="cpu")
    mask_padded[:, :M, :N] = dense_mask.to(torch.float32)
    del dense_mask
    mask_padded = mask_padded.to(Q.device)

    out = torch.empty(B, H, M, D, dtype=Q.dtype, device=Q.device)
    
    for b in range(B):
        for h in range(H):
            kv_h = h // head_groups
            res = block_sparse_kernel[_lnc_degree()](Q[b, h].contiguous(), K[b, kv_h].contiguous(), V[b, kv_h].contiguous(), mask_padded[h], softmax_scale)
            out[b, h] = res
    return out


def get_last_config() -> dict | None:
    return None
