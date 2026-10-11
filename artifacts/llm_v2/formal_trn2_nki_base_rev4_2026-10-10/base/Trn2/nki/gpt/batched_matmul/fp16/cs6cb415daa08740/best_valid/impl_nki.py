import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _gemm_group(a, b, c, M, N, K, TM, TN, TK, G, slots):
    nk = (K + TK - 1) // TK
    npad = ((N + 511) // 512) * 512

    # Batch and contraction-chunk axes are free axes in SBUF.
    # Each matrix slice consequently has contiguous M/N coordinates.
    lhs = nl.ndarray((TK, nk, G, M), dtype=a.dtype, buffer=nl.sbuf)
    rhs = nl.ndarray((TK, nk, G, N), dtype=b.dtype, buffer=nl.sbuf)

    a_rows = a.reshape((1, G * M, K))

    if K % TK == 0:
        a_src = a_rows.reshape((1, G * M, nk, TK)).permute((0, 2, 1, 3))
        nisa.dma_transpose(
            dst=lhs.reshape((TK, nk, G * M, 1)),
            src=a_src,
            axes=(3, 1, 2, 0),
            dge_mode=nisa.dge_mode.hwdge,
        )

        b_src = b.reshape((G, nk, TK, N)).permute((2, 1, 0, 3))
        nisa.dma_copy(dst=rhs, src=b_src)
    else:
        full = K // TK
        remainder = K - full * TK

        # Only the final K chunk needs zero initialization.
        nisa.memset(dst=lhs[:, full, :, :], value=0.0)
        nisa.memset(dst=rhs[:, full, :, :], value=0.0)

        if full > 0:
            a_prefix = a_rows[:, :, 0:full * TK]
            a_src = a_prefix.reshape_dim(2, (full, TK)).permute((0, 2, 1, 3))
            nisa.dma_transpose(
                dst=lhs[:, 0:full, :, :].reshape((TK, full, G * M, 1)),
                src=a_src,
                axes=(3, 1, 2, 0),
                dge_mode=nisa.dge_mode.hwdge,
            )

            b_prefix = b[:, 0:full * TK, :]
            b_src = b_prefix.reshape_dim(1, (full, TK)).permute((2, 1, 0, 3))
            nisa.dma_copy(dst=rhs[:, 0:full, :, :], src=b_src)

        nisa.dma_transpose(
            dst=lhs[0:remainder, full, :, :].reshape((remainder, G * M, 1)),
            src=a_rows[:, :, full * TK:K],
            axes=(2, 1, 0),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=rhs[0:remainder, full, :, :],
            src=b[:, full * TK:K, :].permute((1, 0, 2)),
        )

    # Bank-aligned batch slots permit several independent accumulators
    # without any partial sums leaving PSUM.
    acc = nl.ndarray((TM, slots, npad), dtype=nl.float32, buffer=nl.psum)
    result = nl.ndarray((TM, G, N), dtype=a.dtype, buffer=nl.sbuf)

    for mi in range((M + TM - 1) // TM):
        m0 = mi * TM
        mr = min(TM, M - m0)

        for b0 in range(0, G, slots):
            for ki in range(nk):
                for si in range(slots):
                    bi = b0 + si
                    for ni in range((N + TN - 1) // TN):
                        n0 = ni * TN
                        nr = min(TN, N - n0)
                        nisa.nc_matmul(
                            dst=acc[0:mr, si, n0:n0 + nr],
                            stationary=lhs[:, ki, bi, m0:m0 + mr],
                            moving=rhs[:, ki, bi, n0:n0 + nr],
                            accumulate=(ki > 0),
                        )

            for si in range(slots):
                bi = b0 + si
                nisa.tensor_copy(
                    dst=result[0:mr, bi, :],
                    src=acc[0:mr, si, 0:N],
                )

        # One grouped store for this M tile. No padding is written.
        nisa.dma_copy(
            dst=c[:, m0:m0 + mr, :].permute((1, 0, 2)),
            src=result[0:mr, :, :],
        )


@nki.jit
def _batched_gemm(A, B, BATCH, M, N, K, TM, TN, TK, G, slots):
    out = nl.ndarray(
        (BATCH * M * N,),
        dtype=A.dtype,
        buffer=nl.shared_hbm,
    )

    groups = BATCH // G
    a_groups = A.reshape((groups, G, M, K))
    b_groups = B.reshape((groups, G, K, N))
    c_groups = out.reshape((groups, G, M, N))

    programs = nl.num_programs()
    pid = nl.program_id(0)
    groups_per_program = groups // programs
    first = pid * groups_per_program
    last = first + groups_per_program

    if groups_per_program == 1:
        _gemm_group(
            a_groups[first],
            b_groups[first],
            c_groups[first],
            M, N, K, TM, TN, TK, G, slots,
        )
    else:
        def group_body(gi):
            _gemm_group(
                a_groups[gi],
                b_groups[gi],
                c_groups[gi],
                M, N, K, TM, TN, TK, G, slots,
            )

        nl.fori_loop(first, last, group_body)

    return out


def run(
    A: torch.Tensor,
    B: torch.Tensor,
    BATCH: int,
    M: int,
    N: int,
    K: int,
    **kwargs,
):
    global _LAST_CONFIG

    tm = min(128, M)
    tn = min(512, N)
    tk = min(128, K)

    npad = ((N + 511) // 512) * 512
    slots = 4096 // npad
    batch_group = 2 * slots

    _LAST_CONFIG = {
        "tile_m": tm,
        "tile_n": tn,
        "tile_k": tk,
        "psum_batch_slots": slots,
        "psum_slot_width": npad,
        "batch_group": batch_group,
        "batch_loop": "static" if BATCH // batch_group == 2 else "fori",
        "operand_load": "grouped_packed" if K % tk == 0 else "grouped_padded",
        "output_store": "grouped_m_tile",
        "transpose": "hwdge",
        "accumulator_dtype": "float32",
        "lnc": 2,
        "launches": 1,
    }

    return _batched_gemm[2](
        A, B, BATCH, M, N, K,
        tm, tn, tk, batch_group, slots,
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
