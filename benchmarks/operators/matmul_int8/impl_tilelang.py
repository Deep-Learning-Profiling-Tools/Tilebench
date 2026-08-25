import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

from . import impl_torch


_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "threads": 128,
    "num_stages": 2,
}
_last_autotune_config: dict = {}


_HANG = frozenset({
    (64, 64, 32, 128, 3), (64, 64, 32, 128, 4), (64, 64, 32, 256, 3),
    (64, 64, 32, 256, 4), (64, 64, 32, 512, 3), (64, 64, 32, 512, 4),
    (64, 64, 64, 512, 3), (64, 64, 64, 512, 4), (64, 128, 32, 128, 3),
    (64, 128, 32, 128, 4), (64, 128, 32, 256, 3), (64, 128, 32, 256, 4),
    (64, 128, 32, 512, 3), (64, 128, 32, 512, 4), (64, 256, 32, 512, 3),
    (64, 256, 32, 512, 4), (64, 256, 64, 128, 4), (128, 64, 32, 128, 3),
    (128, 64, 32, 128, 4), (128, 64, 32, 256, 3), (128, 64, 32, 256, 4),
    (128, 64, 32, 512, 3), (128, 64, 32, 512, 4), (128, 64, 64, 512, 3),
    (128, 64, 64, 512, 4), (128, 128, 32, 512, 4), (128, 256, 32, 512, 3),
    (128, 256, 32, 512, 4), (64, 128, 64, 512, 3),
    (64, 128, 64, 512, 4), (128, 128, 32, 512, 3),
})

_ref_state: dict = {}


def _autotune_ref(a, b, c):
    _ref_state["c"] = c
    _ref_state["ref"] = impl_torch.run(a, b)
    return None


def _autotune_check(lib_outs, ref_outs):
    torch.testing.assert_close(
        _ref_state["c"], _ref_state["ref"], atol=0, rtol=0
    )


def matmul_configs():
    # from Triton implementation
    def fits_triton_smem_budget(bm, bn, bk, ns):
        return (bm * bk + bn * bk) * ns + bm * bn * 4 <= 220_000

    def runtime_or_correctness_prone(bm, bn, bk, nt, ns):
        return (bm, bn, bk, nt, ns) in _HANG

    return [
        dict(
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            GROUP_SIZE_M=8,
            threads=nt,
            num_stages=ns,
        )
        for bm in [64, 128, 256]
        for bn in [64, 128, 256]
        for bk in [32, 64]
        for nt in [128, 256, 512]
        for ns in [3, 4]
        if bm * bn <= 128 * 256
        and fits_triton_smem_budget(bm, bn, bk, ns)
        and not runtime_or_correctness_prone(bm, bn, bk, nt, ns)
    ]


@tilelang.autotune(
    configs=matmul_configs(),
    warmup=3,
    rep=10,
    timeout=60,
    ref_prog=_autotune_ref,
    manual_check_prog=_autotune_check,
)
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def matmul_kernel(
    a,
    b,
    c,
    BLOCK_SIZE_M: int = 128,
    BLOCK_SIZE_N: int = 128,
    BLOCK_SIZE_K: int = 64,
    GROUP_SIZE_M: int = 8,
    threads: int = 256,
    num_stages: int = 2,
):
    M, K, K_b, N = T.const("M, K, K_b, N")
    a: T.Tensor((M, K), "int8")
    b: T.Tensor((K_b, N), "uint8")
    c: T.Tensor((M, N), "int32")

    with T.Kernel(
        T.ceildiv(M, BLOCK_SIZE_M),
        T.ceildiv(N, BLOCK_SIZE_N),
        threads=threads,
    ) as (pid_m, pid_n):
        T.use_swizzle(panel_size=GROUP_SIZE_M, enable=True)

        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N

        a_shared = T.alloc_shared((BLOCK_SIZE_M, BLOCK_SIZE_K), "int8")
        b_packed_shared = T.alloc_shared((BLOCK_SIZE_K, BLOCK_SIZE_N), "uint8")
        b_packed_local = T.alloc_fragment((BLOCK_SIZE_K, BLOCK_SIZE_N), "uint8")
        b_unpacked_local = T.alloc_fragment((BLOCK_SIZE_K, BLOCK_SIZE_N), "int8")
        b_unpacked_shared = T.alloc_shared((BLOCK_SIZE_K, BLOCK_SIZE_N), "int8")
        acc = T.alloc_fragment((BLOCK_SIZE_M, BLOCK_SIZE_N), "int32")
        acc_tmem = T.alloc_tmem((BLOCK_SIZE_M, BLOCK_SIZE_N), "int32")
        mbar = T.alloc_barrier(1)

        for kb_tile in T.Pipelined(T.ceildiv(K_b, BLOCK_SIZE_K), num_stages=num_stages):
            T.copy(b[kb_tile * BLOCK_SIZE_K, start_n], b_packed_shared)
            T.copy(b_packed_shared, b_packed_local)

            for field_i in T.serial(4):
                k_pos = field_i * K_b + kb_tile * BLOCK_SIZE_K
                T.copy(a[start_m, k_pos], a_shared)

                for kk, nn in T.Parallel(BLOCK_SIZE_K, BLOCK_SIZE_N):
                    field = T.bitwise_and(
                        T.shift_right(T.cast(b_packed_local[kk, nn], "int32"), 2 * field_i),
                        3,
                    )
                    b_unpacked_local[kk, nn] = T.cast(field, "int8") - T.cast(1, "int8")

                T.copy(b_unpacked_local, b_unpacked_shared)
                T.sync_threads()
                T.gemm(
                    a_shared,
                    b_unpacked_shared,
                    acc_tmem,
                    mbar=mbar,
                    clear_accum=kb_tile + field_i == 0,
                )
                T.sync_threads()

        T.copy(acc_tmem, acc)
        T.copy(acc, c[start_m, start_n])


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False) -> torch.Tensor:
    assert a.shape[1] == b.shape[0] * 4, (
        "Incompatible dims: A's K must equal 4 * B's K_b (B is packed 4-per-byte)"
    )
    assert a.is_contiguous(), "A must be contiguous"

    a = a.contiguous()
    b = b.contiguous()
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)

    if autotune:
        with set_autotune_inputs(a, b, c):
            kernel = matmul_kernel.compile(a, b, c)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(a, b, c)
    else:
        _last_autotune_config.clear()
        cfg = dict(_DEFAULT_CONFIG)
        matmul_kernel(
            a,
            b,
            c,
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
        )

    return c


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
