"""The shipped TileLang matmul kernel, with warp specialization as a switch.

Body is copied verbatim from benchmarks/operators/matmul_fp32_fp16_fp8/
impl_tilelang.py (the fp16/fp8 tmem branch). The only thing that varies is the
TL_DISABLE_WARP_SPECIALIZED pass config.
"""

import tilelang
import tilelang.language as T
import torch

# impl_tilelang.py _DEFAULT_CONFIGS
DEFAULT_CFG = {
    torch.float16: dict(bm=128, bn=128, bk=64, gs=8, threads=256, stages=3),
    torch.float8_e4m3fn: dict(bm=128, bn=256, bk=64, gs=8, threads=256, stages=3),
    torch.float32: dict(bm=128, bn=128, bk=32, gs=8, threads=256, stages=3),
}


def build_matmul(bm, bn, bk, gs, threads, stages, warp_spec: bool):
    def kern(a, b, c, dtype):
        M, K, N = T.const("M, K, N")
        a: T.Tensor((M, K), dtype)
        b: T.Tensor((K, N), dtype)
        c: T.Tensor((M, N), dtype)
        with T.Kernel(T.ceildiv(M, bm), T.ceildiv(N, bn), threads=threads) as (pm, pn):
            T.use_swizzle(panel_size=gs, enable=True)
            sm, sn = pm * bm, pn * bn
            a_tile = T.alloc_shared((bm, bk), dtype)
            b_tile = T.alloc_shared((bk, bn), dtype)
            acc = T.alloc_fragment((bm, bn), "float32")
            acc_tmem = T.alloc_tmem((bm, bn), "float32")
            mbar = T.alloc_barrier(1)
            for k in T.Pipelined(T.ceildiv(K, bk), num_stages=stages):
                T.copy(a[sm, k * bk], a_tile)
                T.copy(b[k * bk, sn], b_tile)
                T.gemm(a_tile, b_tile, acc_tmem, mbar=mbar, clear_accum=k == 0)
            T.copy(acc_tmem, acc)
            T.copy(acc, c[sm, sn])

    return tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: not warp_spec
        }
    )(kern)
