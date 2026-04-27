"""cuTile linear self-attention via the same 3 kernels as impl_triton.py.

Stage 1 (`_kv_kernel`):  S = phi(K)^T @ V into a (D, D) buffer.
  Grid (D, D); each CTA owns one S[d0, d1] scalar and reduces along M.

Stage 2 (`_z_kernel`):   Z = sum_m phi(K[m, :]) into a (D,) buffer.
  Grid (D,); each CTA owns one Z[d] scalar.

Stage 3 (`_out_kernel`): O = (phi(Q) @ S) / (phi(Q) @ Z + eps).
  Grid (cdiv(M, BLOCK_M), cdiv(D, BLOCK_D)); each CTA emits a
  (BLOCK_M, BLOCK_D) output tile.

Only stage 3 is autotuned (matches Triton). Stages 1/2 use a fixed
KV_BLOCK_M to keep the search space narrow on both backends.
"""
from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_KV_BLOCK_M = 32

_DEFAULT_OUT = SimpleNamespace(block_m=32, block_d=16, occupancy=4)
_OUT_SEARCH_SPACE = [
    SimpleNamespace(block_m=bm, block_d=bd, occupancy=occ)
    for bm in [16, 32, 64]
    for bd in [8, 16, 32]
    for occ in [4, 8, 16]
]


def _phi_tile(x):
    # phi(x) = ELU(x) + 1
    return ct.where(x > 0, x + 1.0, ct.exp(x))


@ct.kernel
def _kv_kernel(
    S, K, V,
    M, D,
    BLOCK_M: ConstInt,
):
    pid_d0 = ct.bid(0)
    pid_d1 = ct.bid(1)

    acc = ct.full((1, 1), 0.0, dtype=ct.float32)

    num_m_tiles = ct.cdiv(M, BLOCK_M)
    m_tile = 0
    while m_tile < num_m_tiles:
        offs_m = m_tile * BLOCK_M + ct.expand_dims(ct.arange(BLOCK_M, dtype=ct.int32), 1)
        valid_m = offs_m < M

        k_tile = ct.load(K, index=(m_tile, pid_d0), shape=(BLOCK_M, 1),
                         padding_mode=ct.PaddingMode.ZERO)
        v_tile = ct.load(V, index=(m_tile, pid_d1), shape=(BLOCK_M, 1),
                         padding_mode=ct.PaddingMode.ZERO)

        phi_k = _phi_tile(k_tile)
        phi_k = ct.where(valid_m, phi_k, 0.0)
        v_tile = ct.where(valid_m, v_tile, 0.0)

        acc = acc + ct.sum(phi_k * v_tile, axis=0, keepdims=True)
        m_tile = m_tile + 1

    ct.store(S, index=(pid_d0, pid_d1), tile=acc)


@ct.kernel
def _z_kernel(
    Z, K,
    M, D,
    BLOCK_M: ConstInt,
):
    pid_d = ct.bid(0)

    acc = ct.full((1,), 0.0, dtype=ct.float32)

    num_m_tiles = ct.cdiv(M, BLOCK_M)
    m_tile = 0
    while m_tile < num_m_tiles:
        offs_m = m_tile * BLOCK_M + ct.expand_dims(ct.arange(BLOCK_M, dtype=ct.int32), 1)
        valid_m = offs_m < M

        k_tile = ct.load(K, index=(m_tile, pid_d), shape=(BLOCK_M, 1),
                         padding_mode=ct.PaddingMode.ZERO)

        phi_k = _phi_tile(k_tile)
        phi_k = ct.where(valid_m, phi_k, 0.0)

        acc = acc + ct.sum(phi_k, axis=0)
        m_tile = m_tile + 1

    ct.store(Z, index=(pid_d,), tile=acc)


@ct.kernel
def _out_kernel(
    O, Q, S, Z,
    M, D,
    eps: ct.Constant[float],
    BLOCK_M: ConstInt,
    BLOCK_D: ConstInt,
):
    pid_m = ct.bid(0)
    pid_do = ct.bid(1)

    offs_m = pid_m * BLOCK_M + ct.expand_dims(ct.arange(BLOCK_M, dtype=ct.int32), 1)
    valid_m = offs_m < M

    numer = ct.full((BLOCK_M, BLOCK_D), 0.0, dtype=ct.float32)
    denom = ct.full((BLOCK_M, 1), 0.0, dtype=ct.float32)

    d_idx = 0
    while d_idx < D:
        q_tile = ct.load(Q, index=(pid_m, d_idx), shape=(BLOCK_M, 1),
                         padding_mode=ct.PaddingMode.ZERO)
        s_tile = ct.load(S, index=(d_idx, pid_do), shape=(1, BLOCK_D),
                         padding_mode=ct.PaddingMode.ZERO)
        z_tile = ct.load(Z, index=(d_idx,), shape=(1,),
                         padding_mode=ct.PaddingMode.ZERO)

        phi_q = _phi_tile(q_tile)
        phi_q = ct.where(valid_m, phi_q, 0.0)

        numer = numer + phi_q * s_tile
        denom = denom + phi_q * ct.reshape(z_tile, (1, 1))

        d_idx = d_idx + 1

    out_tile = numer / (denom + eps)
    ct.store(O, index=(pid_m, pid_do), tile=out_tile)


# Only the OUT kernel is autotuned (mirrors Triton).
_out_tuner = CutileAutotuner(_out_kernel)


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6,
        block_size: int = None, autotune: bool = False, **kwargs):
    global _last_autotune_config

    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == K.dtype == V.dtype == torch.float32

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    M, D = Q.shape
    O = torch.empty((M, D), device=Q.device, dtype=torch.float32)
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((D,), device=Q.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    # Stage 1: S = phi(K)^T @ V
    ct.launch(stream, (D, D, 1), _kv_kernel, (S, K, V, M, D, _KV_BLOCK_M))

    # Stage 2: Z = sum_m phi(K)
    ct.launch(stream, (D, 1, 1), _z_kernel, (Z, K, M, D, _KV_BLOCK_M))

    # Stage 3: O = (phi(Q) @ S) / (phi(Q) @ Z + eps)
    if autotune:
        out_cfg = _out_tuner.tune_or_cached(
            shape_key=(M, D),
            search_space=_OUT_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (
                (M + cfg.block_m - 1) // cfg.block_m,
                (D + cfg.block_d - 1) // cfg.block_d,
                1,
            ),
            args_fn=lambda cfg: (
                O, Q, S, Z, M, D, float(eps), cfg.block_m, cfg.block_d,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config = {
            "block_m":   out_cfg.block_m,
            "block_d":   out_cfg.block_d,
            "occupancy": out_cfg.occupancy,
        }
    else:
        BLOCK_M = int(block_size) if block_size is not None else _DEFAULT_OUT.block_m
        out_cfg = SimpleNamespace(
            block_m=BLOCK_M,
            block_d=_DEFAULT_OUT.block_d,
            occupancy=_DEFAULT_OUT.occupancy,
        )

    out_kernel = _out_tuner.kernel_with_hints(occupancy=out_cfg.occupancy)
    grid = (
        (M + out_cfg.block_m - 1) // out_cfg.block_m,
        (D + out_cfg.block_d - 1) // out_cfg.block_d,
        1,
    )
    ct.launch(
        stream, grid, out_kernel,
        (O, Q, S, Z, M, D, float(eps), out_cfg.block_m, out_cfg.block_d),
    )

    return O


def get_last_config() -> dict | None:
    return _last_autotune_config
