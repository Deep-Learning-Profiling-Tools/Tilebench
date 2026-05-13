import math
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

_LAST_CONFIG = None


def _phi_tile(x):
    return ct.where(x > 0, x + 1.0, ct.exp(x))


@ct.kernel
def linear_attention_kv_kernel(
    S,
    K,
    V,
    M: ConstInt,
    D: ConstInt,
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

        k_tile = ct.load(
            K,
            index=(m_tile, pid_d0),
            shape=(BLOCK_M, 1),
            padding_mode=ct.PaddingMode.ZERO,
        )
        v_tile = ct.load(
            V,
            index=(m_tile, pid_d1),
            shape=(BLOCK_M, 1),
            padding_mode=ct.PaddingMode.ZERO,
        )

        phi_k = _phi_tile(k_tile)
        phi_k = ct.where(valid_m, phi_k, 0.0)
        v_tile = ct.where(valid_m, v_tile, 0.0)

        acc = acc + ct.sum(phi_k * v_tile, axis=0, keepdims=True)
        m_tile = m_tile + 1

    ct.store(S, index=(pid_d0, pid_d1), tile=acc)


@ct.kernel
def linear_attention_z_kernel(
    Z,
    K,
    M: ConstInt,
    D: ConstInt,
    BLOCK_M: ConstInt,
):
    pid_d = ct.bid(0)

    acc = ct.full((1,), 0.0, dtype=ct.float32)

    num_m_tiles = ct.cdiv(M, BLOCK_M)
    m_tile = 0
    while m_tile < num_m_tiles:
        offs_m = m_tile * BLOCK_M + ct.expand_dims(ct.arange(BLOCK_M, dtype=ct.int32), 1)
        valid_m = offs_m < M

        k_tile = ct.load(
            K,
            index=(m_tile, pid_d),
            shape=(BLOCK_M, 1),
            padding_mode=ct.PaddingMode.ZERO,
        )

        phi_k = _phi_tile(k_tile)
        phi_k = ct.where(valid_m, phi_k, 0.0)

        acc = acc + ct.sum(phi_k, axis=0)
        m_tile = m_tile + 1

    ct.store(Z, index=(pid_d,), tile=acc)


@ct.kernel
def linear_attention_out_kernel(
    O,
    Q,
    S,
    Z,
    M: ConstInt,
    D: ConstInt,
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
        q_tile = ct.load(
            Q,
            index=(pid_m, d_idx),
            shape=(BLOCK_M, 1),
            padding_mode=ct.PaddingMode.ZERO,
        )
        s_tile = ct.load(
            S,
            index=(d_idx, pid_do),
            shape=(1, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
        )
        z_tile = ct.load(
            Z,
            index=(d_idx,),
            shape=(1,),
            padding_mode=ct.PaddingMode.ZERO,
        )

        phi_q = _phi_tile(q_tile)
        phi_q = ct.where(valid_m, phi_q, 0.0)

        numer = numer + phi_q * s_tile
        denom = denom + phi_q * ct.reshape(z_tile, (1, 1))

        d_idx = d_idx + 1

    out_tile = numer / (denom + eps)
    ct.store(O, index=(pid_m, pid_do), tile=out_tile)


def run(
    Q,
    K,
    V,
    eps: float = 1e-6,
    BLOCK_M: int = 32,
    BLOCK_D: int = 16,
    block_size: int = None,
    autotune: bool = False,
    **kwargs,
):
    global _LAST_CONFIG

    if block_size is not None:
        BLOCK_M = int(block_size)

    BLOCK_M = int(BLOCK_M)
    BLOCK_D = int(BLOCK_D)

    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.ndim == 2 and K.ndim == 2 and V.ndim == 2
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == torch.float32 and K.dtype == torch.float32 and V.dtype == torch.float32

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    M, D = Q.shape
    O = torch.empty((M, D), device=Q.device, dtype=torch.float32)
    S = torch.empty((D, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((D,), device=Q.device, dtype=torch.float32)

    ct.launch(
        torch.cuda.current_stream(),
        (D, D, 1),
        linear_attention_kv_kernel,
        (
            S,
            K,
            V,
            M,
            D,
            BLOCK_M,
        ),
    )

    ct.launch(
        torch.cuda.current_stream(),
        (D, 1, 1),
        linear_attention_z_kernel,
        (
            Z,
            K,
            M,
            D,
            BLOCK_M,
        ),
    )

    ct.launch(
        torch.cuda.current_stream(),
        (math.ceil(M / BLOCK_M), math.ceil(D / BLOCK_D), 1),
        linear_attention_out_kernel,
        (
            O,
            Q,
            S,
            Z,
            M,
            D,
            float(eps),
            BLOCK_M,
            BLOCK_D,
        ),
    )

    _LAST_CONFIG = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_D": BLOCK_D,
        "eps": float(eps),
        "kernel_style": "scalar_reduction",
    }
    return O


def get_last_config() -> dict | None:
    return _LAST_CONFIG