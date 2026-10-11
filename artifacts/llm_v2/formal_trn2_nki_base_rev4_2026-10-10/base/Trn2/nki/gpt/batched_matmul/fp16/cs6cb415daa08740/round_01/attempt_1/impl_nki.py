import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _gemm_one(a, b, c, M, N, K, TM, TN, TK):
    # a: [1, M, K], b: [K, N], c: [M, N].
    # Both operands remain in their original precision.
    nk = (K + TK - 1) // TK
    lhs = nl.ndarray((TK, nk, M), dtype=a.dtype, buffer=nl.sbuf)
    rhs = nl.ndarray((TK, nk, N), dtype=b.dtype, buffer=nl.sbuf)

    if K % TK == 0:
        # HWDGE transposes the first and last axes. The singleton
        # first axis allows a direct HBM-to-SBUF transpose of A.
        a_view = a.reshape((1, M, nk, TK)).permute((0, 2, 1, 3))
        nisa.dma_transpose(
            dst=lhs.reshape((TK, nk, M, 1)),
            src=a_view,
            axes=(3, 1, 2, 0),
            dge_mode=nisa.dge_mode.hwdge,
        )

        b_view = b.reshape((nk, TK, N)).permute((1, 0, 2))
        nisa.dma_copy(dst=rhs, src=b_view)
    else:
        # The final contraction tile is zero-padded on chip.
        nisa.memset(dst=lhs, value=0.0)
        nisa.memset(dst=rhs, value=0.0)
        for ki in range(nk):
            k0 = ki * TK
            kc = min(TK, K - k0)
            nisa.dma_transpose(
                dst=lhs[0:kc, ki, :].reshape((kc, M, 1)),
                src=a[:, :, k0:k0 + kc],
                axes=(2, 1, 0),
                dge_mode=nisa.dge_mode.hwdge,
            )
            nisa.dma_copy(
                dst=rhs[0:kc, ki, :],
                src=b[k0:k0 + kc, :],
            )

    # Reuse the PSUM allocation between M tiles. Each output region
    # is overwritten by its first K chunk and subsequently accumulated.
    acc = nl.ndarray((TM, N), dtype=nl.float32, buffer=nl.psum)

    for mi in range((M + TM - 1) // TM):
        m0 = mi * TM
        mr = min(TM, M - m0)

        for ki in range(nk):
            for ni in range((N + TN - 1) // TN):
                n0 = ni * TN
                nr = min(TN, N - n0)
                nisa.nc_matmul(
                    dst=acc[0:mr, n0:n0 + nr],
                    stationary=lhs[:, ki, m0:m0 + mr],
                    moving=rhs[:, ki, n0:n0 + nr],
                    accumulate=(ki > 0),
                )

        result = nl.ndarray((mr, N), dtype=a.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=acc[0:mr, :])
        nisa.dma_copy(dst=c[m0:m0 + mr, :], src=result)


@nki.jit
def _batched_gemm(A, B, BATCH, M, N, K, TM, TN, TK, batch_group):
    out = nl.ndarray(
        (BATCH * M * N,),
        dtype=A.dtype,
        buffer=nl.shared_hbm,
    )

    groups = BATCH // batch_group
    a_groups = A.reshape((groups, batch_group, 1, M, K))
    b_groups = B.reshape((groups, batch_group, K, N))
    c_groups = out.reshape((groups, batch_group, M, N))

    pid = nl.program_id(0)
    programs = nl.num_programs()
    groups_per_program = groups // programs
    first = pid * groups_per_program
    last = first + groups_per_program

    if batch_group == 1:
        # Small GEMMs are unrolled to expose independent DMA and
        # Tensor Engine work to the scheduler.
        for gi in range(first, last):
            _gemm_one(
                a_groups[gi, 0],
                b_groups[gi, 0],
                c_groups[gi, 0],
                M, N, K, TM, TN, TK,
            )
    else:
        # Larger GEMMs use a rolled outer loop with a fixed-size
        # unrolled group, bounding code size and on-chip live storage.
        def group_body(gi):
            for bi in range(batch_group):
                _gemm_one(
                    a_groups[gi, bi],
                    b_groups[gi, bi],
                    c_groups[gi, bi],
                    M, N, K, TM, TN, TK,
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
    batch_group = 1 if M <= 128 else 4

    _LAST_CONFIG = {
        "tile_m": tm,
        "tile_n": tn,
        "tile_k": tk,
        "batch_group": batch_group,
        "batch_loop": "static" if batch_group == 1 else "fori",
        "operand_load": "packed_4d" if K % tk == 0 else "padded_3d",
        "transpose": "hwdge",
        "accumulator_dtype": "float32",
        "lnc": 2,
        "launches": 1,
    }

    return _batched_gemm[2](
        A, B, BATCH, M, N, K, tm, tn, tk, batch_group
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
