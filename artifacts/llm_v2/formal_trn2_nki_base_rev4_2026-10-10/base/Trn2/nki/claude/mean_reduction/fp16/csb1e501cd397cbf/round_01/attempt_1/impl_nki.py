import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ----------------------------------------------------------------------------
# Static configuration (deterministic functions of the shape only)
# ----------------------------------------------------------------------------
P = 128            # partitions per tile
MM_W = 512         # matmul moving free size (one PSUM bank of fp32)
TARGET_W = 4096    # target columns per DMA unit (8 KiB/partition, 1 MiB per DMA)
T_MIN_N = 4096     # use the TensorE (identity-matmul accumulation) path when N >= this

_LAST_CONFIG = {}


@nki.jit
def _row_mean_kernel(x, chunk, n_chunks, use_t, inv_n):
    # x: (M, N) in HBM. Each program (physical core) handles M / n_prog rows.
    # Partition p of program pid holds rows pid*P*RP + p*RP + r, r in [0, RP).
    M = x.shape[0]
    N = x.shape[1]
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    RP = M // (n_prog * P)

    out = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    xv = x.reshape((n_prog, P, RP, N))
    ov = out.reshape((n_prog, P, RP))

    if use_t:
        # Build an identity matrix (stationary operand) in the input dtype.
        iota_j = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=iota_j, pattern=[[1, P]], offset=0, channel_multiplier=0)
        iota_p = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=iota_p, pattern=[[1, 1]], offset=0, channel_multiplier=1)
        f_j = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=f_j, src=iota_j)
        f_p = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=f_p, src=iota_p)
        ident_f = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ident_f, data=f_j, op0=nl.equal, operand0=f_p)
        ident = nl.ndarray((P, P), dtype=x.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ident, src=ident_f)

    # write-only scratch for the ScalarE copy (only its fused fp32 sum is used)
    junk = nl.ndarray((P, chunk), dtype=x.dtype, buffer=nl.sbuf)
    rowsum = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)

    for r in range(RP):
        # VectorE-owned partials: per-unit V sums, T (PSUM) sum, S total
        pv = nl.ndarray((P, n_chunks + 2), dtype=nl.float32, buffer=nl.sbuf)
        # ScalarE-owned partials: per-unit S sums
        ps = nl.ndarray((P, n_chunks), dtype=nl.float32, buffer=nl.sbuf)
        if use_t:
            acc = nl.ndarray((P, MM_W), dtype=nl.float32, buffer=nl.psum)
        n_mm = 0
        n_v = 0
        n_s = 0
        for ci in range(n_chunks):
            c0 = ci * chunk
            w = min(chunk, N - c0)
            t = nl.ndarray((P, w), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=xv[pid, :, r, c0:c0 + w])

            # split the unit's columns between TensorE / VectorE / ScalarE
            w_t = 0
            if use_t:
                w_t = ((w // 2) // MM_W) * MM_W
            rem = w - w_t
            w_v = ((rem // 2) // 2) * 2
            s0 = w_t + w_v
            w_s = w - s0

            if w_t > 0:
                n_j = w_t // MM_W
                for j in range(n_j):
                    nisa.nc_matmul(dst=acc, stationary=ident,
                                   moving=t[:, j * MM_W:(j + 1) * MM_W],
                                   accumulate=(n_mm > 0))
                    n_mm = n_mm + 1
            if w_v > 0:
                nisa.tensor_reduce(dst=pv[:, n_v:n_v + 1], op=nl.add,
                                   data=t[:, w_t:s0], axis=1)
                n_v = n_v + 1
            if w_s > 0:
                nisa.activation(dst=junk[:, 0:w_s], op=nl.copy, data=t[:, s0:w],
                                reduce_op=nl.add, reduce_res=ps[:, n_s:n_s + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                n_s = n_s + 1

        if n_mm > 0:
            nisa.tensor_reduce(dst=pv[:, n_v:n_v + 1], op=nl.add, data=acc, axis=1)
            n_v = n_v + 1
        if n_s > 0:
            nisa.tensor_reduce(dst=pv[:, n_v:n_v + 1], op=nl.add,
                               data=ps[:, 0:n_s], axis=1)
            n_v = n_v + 1
        nisa.tensor_reduce(dst=rowsum[:, r:r + 1], op=nl.add,
                           data=pv[:, 0:n_v], axis=1)

    res = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=res, data=rowsum, op0=nl.multiply, operand0=inv_n)
    nisa.dma_copy(dst=ov[pid], src=res)
    return out


def _plan(M, N, dtype):
    n_chunks = (N + TARGET_W - 1) // TARGET_W
    chunk = (N + n_chunks - 1) // n_chunks
    chunk = ((chunk + MM_W - 1) // MM_W) * MM_W
    n_chunks = (N + chunk - 1) // chunk
    two_byte = dtype in (torch.float16, torch.bfloat16)
    use_t = bool(two_byte and N >= T_MIN_N)
    if M % (2 * P) == 0:
        lnc = 2
    elif M % P == 0:
        lnc = 1
    else:
        raise ValueError("mean_reduction kernel requires M to be a multiple of 128")
    return {
        "lnc": lnc,
        "chunk": int(chunk),
        "n_chunks": int(n_chunks),
        "use_tensor_engine": use_t,
        "rows_per_partition": int(M // (lnc * P)),
        "matmul_width": MM_W,
        "split": "T:1/2(N>=4096) V:1/2 of rest S:rest",
    }


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    global _LAST_CONFIG
    if x.ndim != 2:
        raise ValueError("expected a 2-D input")
    d = dim
    if d < 0:
        d = d + 2
    if d != 1:
        raise ValueError("only dim=1 is supported")
    M = int(x.shape[0])
    N = int(x.shape[1])
    cfg = _plan(M, N, x.dtype)
    inv_n = 1.0 / float(N)
    kern = _row_mean_kernel[cfg["lnc"]]
    out = kern(x, cfg["chunk"], cfg["n_chunks"], cfg["use_tensor_engine"], inv_n)
    _LAST_CONFIG = dict(cfg)
    return out.view(M)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
