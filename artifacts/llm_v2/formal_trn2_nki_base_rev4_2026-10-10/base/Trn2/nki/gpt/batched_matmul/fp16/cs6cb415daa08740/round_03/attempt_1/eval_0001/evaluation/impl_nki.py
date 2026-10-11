import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _gemm_group(a, b, c, M, N, K, TM, TN, TK, G, slots, fusion, whole_output):
    nk = (K + TK - 1) // TK
    nm = (M + TM - 1) // TM
    npad = ((N + 511) // 512) * 512

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

    result_tiles = 1
    if whole_output:
        result_tiles = nm

    result = nl.ndarray(
        (TM, result_tiles, G, N),
        dtype=a.dtype,
        buffer=nl.sbuf,
    )

    if fusion > 1:
        # Concatenating small matrices fills the stationary dimension.
        # Only diagonal subtiles are retained; each is an independent
        # full-K fp32 GEMM accumulator.
        fused_groups = G // fusion
        acc = nl.ndarray(
            (fusion * M, fused_groups, npad),
            dtype=nl.float32,
            buffer=nl.psum,
        )

        for ki in range(nk):
            kr = min(TK, K - ki * TK)
            for gi in range(fused_groups):
                b0 = gi * fusion
                nisa.nc_matmul(
                    dst=acc[:, gi, 0:fusion * N],
                    stationary=lhs[
                        0:kr, ki, b0:b0 + fusion, :
                    ].reshape((kr, fusion * M)),
                    moving=rhs[
                        0:kr, ki, b0:b0 + fusion, :
                    ].reshape((kr, fusion * N)),
                    accumulate=(ki > 0),
                )

        for gi in range(fused_groups):
            for fi in range(fusion):
                bi = gi * fusion + fi
                nisa.tensor_copy(
                    dst=result[:, 0, bi, :],
                    src=acc[
                        fi * M:(fi + 1) * M,
                        gi,
                        fi * N:(fi + 1) * N,
                    ],
                )
    else:
        acc = nl.ndarray(
            (TM, slots, npad),
            dtype=nl.float32,
            buffer=nl.psum,
        )

        for mi in range(nm):
            m0 = mi * TM
            mr = min(TM, M - m0)
            ri = 0
            if whole_output:
                ri = mi

            for b0 in range(0, G, slots):
                for ki in range(nk):
                    kr = min(TK, K - ki * TK)
                    for si in range(slots):
                        bi = b0 + si
                        for ni in range((N + TN - 1) // TN):
                            n0 = ni * TN
                            nr = min(TN, N - n0)
                            nisa.nc_matmul(
                                dst=acc[0:mr, si, n0:n0 + nr],
                                stationary=lhs[
                                    0:kr, ki, bi, m0:m0 + mr
                                ],
                                moving=rhs[
                                    0:kr, ki, bi, n0:n0 + nr
                                ],
                                accumulate=(ki > 0),
                            )

                nisa.tensor_copy(
                    dst=result[0:mr, ri, b0:b0 + slots, :],
                    src=acc[0:mr, :, 0:N],
                )

            if not whole_output:
                nisa.dma_copy(
                    dst=c[:, m0:m0 + mr, :].permute((1, 0, 2)),
                    src=result[0:mr, 0, :, :],
                )

    if whole_output:
        full_m = M // TM
        remainder_m = M - full_m * TM

        if full_m > 0:
            c_prefix = c[:, 0:full_m * TM, :]
            c_view = c_prefix.reshape_dim(1, (full_m, TM)).permute((2, 1, 0, 3))
            nisa.dma_copy(
                dst=c_view,
                src=result[:, 0:full_m, :, :],
            )

        if remainder_m > 0:
            nisa.dma_copy(
                dst=c[:, full_m * TM:M, :].permute((1, 0, 2)),
                src=result[0:remainder_m, full_m, :, :],
            )
    elif fusion > 1:
        nisa.dma_copy(
            dst=c.permute((1, 0, 2)),
            src=result[:, 0, :, :],
        )


@nki.jit
def _batched_gemm(
    A, B, BATCH, M, N, K, TM, TN, TK, G, slots, fusion, whole_output
):
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
            a_groups[first], b_groups[first], c_groups[first],
            M, N, K, TM, TN, TK, G, slots, fusion, whole_output,
        )
    else:
        def group_body(gi):
            _gemm_group(
                a_groups[gi], b_groups[gi], c_groups[gi],
                M, N, K, TM, TN, TK, G, slots, fusion, whole_output,
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
    nk = (K + tk - 1) // tk
    nm = (M + tm - 1) // tm
    npad = ((N + 511) // 512) * 512
    slots = 4096 // npad

    # Leave room below the physical per-partition SBUF limit.
    sbuf_budget = 200 * 1024
    bytes_per_batch = 2 * nk * (M + N) + 2 * N
    batch_group = min(16, BATCH // 2)
    while batch_group > 1 and batch_group * bytes_per_batch > sbuf_budget:
        batch_group //= 2

    fusion = 1
    while (
        2 * fusion * tm <= 128
        and 2 * fusion * tn <= 512
        and 2 * fusion <= batch_group
    ):
        fusion *= 2

    whole_output_bytes = (
        2 * nk * batch_group * (M + N)
        + 2 * nm * batch_group * N
    )
    whole_output = whole_output_bytes <= sbuf_budget

    _LAST_CONFIG = {
        "tile_m": tm,
        "tile_n": tn,
        "tile_k": tk,
        "batch_group": batch_group,
        "batch_fusion": fusion,
        "psum_batch_slots": batch_group // fusion if fusion > 1 else slots,
        "psum_slot_width": npad,
        "batch_loop": "static" if BATCH // batch_group == 2 else "fori",
        "operand_load": "grouped_packed" if K % tk == 0 else "grouped_padded",
        "output_store": "grouped_full_prefix" if whole_output else "grouped_m_tile",
        "output_buffer_tiles": nm if whole_output else 1,
        "transpose": "hwdge",
        "accumulator_dtype": "float32",
        "lnc": 2,
        "launches": 1,
    }

    return _batched_gemm[2](
        A, B, BATCH, M, N, K,
        tm, tn, tk, batch_group, slots, fusion, whole_output,
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
