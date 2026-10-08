from types import SimpleNamespace

import cuda.tile as ct
import torch

from tilebench.core.cutile_autotune import CutileAutotuner


ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_KV_CONFIG = SimpleNamespace(
    block_m=32,
    block_n=64,
    block_k=32,
    occupancy=8,
)

_DEFAULT_OUT_CONFIG = SimpleNamespace(
    block_m=32,
    block_n=32,
    block_k=32,
    occupancy=4,
)


# Tile shapes mirror impl_triton's _TILE_SHAPES so both backends search the
# same GEMM decompositions; block_m >= 64 is required for tcgen05 MMA lowering.
_GEMM_SEARCH_SPACE = [
    SimpleNamespace(block_m=bm, block_n=bn, block_k=bk, occupancy=occ)
    for bm in [32, 64, 128]
    for bn in [32, 64, 128]
    for bk in [32, 64]
    for occ in [4, 8, 16, 32]
]


def _phi_tile(x):
    return ct.where(x > 0, x + 1.0, ct.exp(x))


@ct.kernel
def phi_kernel(Y, X, BLOCK_M: ConstInt, BLOCK_D: ConstInt):
    pid_m = ct.bid(0)
    pid_d = ct.bid(1)
    x = ct.load(
        X,
        index=(pid_m, pid_d),
        shape=(BLOCK_M, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(Y, index=(pid_m, pid_d), tile=_phi_tile(x))


@ct.kernel
def kv_gemm_kernel(
    S, PhiK, V,
    M, D,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    num_k_tiles = ct.cdiv(M, BLOCK_K)

    k_tile_idx = 0
    while k_tile_idx < num_k_tiles:
        k = ct.load(
            PhiK,
            index=(k_tile_idx, pid_m),
            shape=(BLOCK_K, BLOCK_M),
            padding_mode=ct.PaddingMode.ZERO,
        )
        v = ct.load(
            V,
            index=(k_tile_idx, pid_n),
            shape=(BLOCK_K, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
        )

        acc = ct.mma(
            ct.transpose(k.astype(ct.tfloat32), 0, 1),
            v.astype(ct.tfloat32),
            acc,
        )
        k_tile_idx = k_tile_idx + 1

    ct.store(S, index=(pid_m, pid_n), tile=acc)


@ct.kernel
def z_kernel(
    Z, PhiK,
    M, D,
    BLOCK_M: ConstInt,
    BLOCK_D: ConstInt,
):
    pid_d = ct.bid(0)
    acc = ct.zeros((BLOCK_D,), dtype=ct.float32)
    num_m_tiles = ct.cdiv(M, BLOCK_M)

    m_tile = 0
    while m_tile < num_m_tiles:
        k = ct.load(
            PhiK,
            index=(m_tile, pid_d),
            shape=(BLOCK_M, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = acc + ct.sum(k, axis=0)
        m_tile = m_tile + 1

    ct.store(Z, index=(pid_d,), tile=acc)


@ct.kernel
def out_gemm_kernel(
    O, PhiQ, S, Z,
    M, D,
    eps: ct.Constant[float],
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    numer = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    denom = ct.zeros((BLOCK_M, 1), dtype=ct.float32)
    num_k_tiles = ct.cdiv(D, BLOCK_K)

    k_tile_idx = 0
    while k_tile_idx < num_k_tiles:
        q = ct.load(
            PhiQ,
            index=(pid_m, k_tile_idx),
            shape=(BLOCK_M, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        s = ct.load(
            S,
            index=(k_tile_idx, pid_n),
            shape=(BLOCK_K, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        z = ct.load(
            Z,
            index=(k_tile_idx,),
            shape=(BLOCK_K,),
            padding_mode=ct.PaddingMode.ZERO,
        )

        numer = ct.mma(
            q.astype(ct.tfloat32),
            s.astype(ct.tfloat32),
            numer,
        )
        denom = denom + ct.sum(
            q * ct.reshape(z, (1, BLOCK_K)),
            axis=1,
            keepdims=True,
        )
        k_tile_idx = k_tile_idx + 1

    ct.store(O, index=(pid_m, pid_n), tile=numer / (denom + eps))


_kv_tuner = CutileAutotuner(kv_gemm_kernel)
_out_tuner = CutileAutotuner(out_gemm_kernel)


def _launch_z(stream, Z, PhiK, M, D):
    ct.launch(
        stream,
        ((D + 31) // 32, 1, 1),
        z_kernel,
        (Z, PhiK, M, D, 32, 32),
    )


def run(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    eps: float = 1e-6,
    block_size: int = None,
    autotune: bool = False,
    **kwargs,
):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == K.dtype == V.dtype == torch.float32

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    M, D = Q.shape
    PhiQ = torch.empty_like(Q)
    PhiK = torch.empty_like(K)
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((D,), device=Q.device, dtype=torch.float32)
    O = torch.empty((M, D), device=Q.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()

    phi_grid = ((M + 31) // 32, (D + 31) // 32, 1)
    ct.launch(stream, phi_grid, phi_kernel, (PhiQ, Q, 32, 32))
    ct.launch(stream, phi_grid, phi_kernel, (PhiK, K, 32, 32))

    if autotune:
        kv_cfg = _kv_tuner.tune_or_cached(
            shape_key=(M, D, str(Q.dtype)),
            search_space=_GEMM_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                (D + cfg.block_m - 1) // cfg.block_m,
                (D + cfg.block_n - 1) // cfg.block_n,
                1,
            ),
            args_fn=lambda cfg: (
                S, PhiK, V, M, D,
                cfg.block_m, cfg.block_n, cfg.block_k,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
    else:
        kv_cfg = SimpleNamespace(**vars(_DEFAULT_KV_CONFIG))

    kv_kernel = _kv_tuner.kernel_with_hints(occupancy=kv_cfg.occupancy)
    ct.launch(
        stream,
        (
            (D + kv_cfg.block_m - 1) // kv_cfg.block_m,
            (D + kv_cfg.block_n - 1) // kv_cfg.block_n,
            1,
        ),
        kv_kernel,
        (
            S, PhiK, V, M, D,
            kv_cfg.block_m, kv_cfg.block_n, kv_cfg.block_k,
        ),
    )

    _launch_z(stream, Z, PhiK, M, D)

    if autotune:
        out_cfg = _out_tuner.tune_or_cached(
            shape_key=(M, D, str(Q.dtype)),
            search_space=_GEMM_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                (M + cfg.block_m - 1) // cfg.block_m,
                (D + cfg.block_n - 1) // cfg.block_n,
                1,
            ),
            args_fn=lambda cfg: (
                O, PhiQ, S, Z, M, D, float(eps),
                cfg.block_m, cfg.block_n, cfg.block_k,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "kv_block_m": kv_cfg.block_m,
            "kv_block_n": kv_cfg.block_n,
            "kv_block_k": kv_cfg.block_k,
            "kv_occupancy": kv_cfg.occupancy,
            "out_block_m": out_cfg.block_m,
            "out_block_n": out_cfg.block_n,
            "out_block_k": out_cfg.block_k,
            "out_occupancy": out_cfg.occupancy,
        })
    else:


        out_cfg = SimpleNamespace(**vars(_DEFAULT_OUT_CONFIG))

    out_kernel = _out_tuner.kernel_with_hints(occupancy=out_cfg.occupancy)
    ct.launch(
        stream,
        (
            (M + out_cfg.block_m - 1) // out_cfg.block_m,
            (D + out_cfg.block_n - 1) // out_cfg.block_n,
            1,
        ),
        out_kernel,
        (
            O, PhiQ, S, Z, M, D, float(eps),
            out_cfg.block_m, out_cfg.block_n, out_cfg.block_k,
        ),
    )

    return O


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
