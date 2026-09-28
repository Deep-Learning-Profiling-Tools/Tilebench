from types import SimpleNamespace

import torch

from tilebench.core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

TILE_M = 128
TILE_K = 128
TILE_N = 512

BLOCK_M = 256
BLOCK_N = 64
BLOCK_KB = 512

if nki is not None:
    @nki.jit
    def matmul_int8_kernel(a_hbm, b_hbm, tiles_in_block_m, block_kb, tile_n):
        M, K = a_hbm.shape
        K_b, N = b_hbm.shape

        assert len(a_hbm.shape) == 2 and len(b_hbm.shape) == 2, "a and b must be 2D"
        assert K == 4 * K_b, "A's K dim must be 4x B's packed K_b dim"

        block_m = TILE_M * tiles_in_block_m
        assert M % block_m == 0, "M must be a multiple of TILE_M * tiles_in_block_m"
        assert N % tile_n == 0, "N must be a multiple of tile_n"
        assert K_b % block_kb == 0, "K_b must be a multiple of block_kb"
        assert block_kb % TILE_K == 0, "block_kb must be a multiple of TILE_K"
        assert tile_n <= 512, "tile_n exceeds the fp32 PSUM bank size"

        num_block_m = M // block_m
        num_block_n = N // tile_n
        num_kb_blocks = K_b // block_kb
        num_j = block_kb // TILE_K

        out_hbm = nl.ndarray((M, N), dtype=nl.int32, buffer=nl.shared_hbm)

        a_slab_w = tiles_in_block_m * 4 * block_kb
        at_slab_w = tiles_in_block_m * 4 * num_j * TILE_M

        for m_blk in range(num_block_m):
            m0 = m_blk * block_m

            for n_blk in range(num_block_n):
                n0 = n_blk * tile_n

                acc_psum = nl.ndarray((TILE_M, tiles_in_block_m * TILE_N),
                                      dtype=nl.float32, buffer=nl.psum)

                for kbb in range(num_kb_blocks):
                    kb0 = kbb * block_kb

                    a_i8 = nl.ndarray((TILE_M, a_slab_w), dtype=nl.int8, buffer=nl.sbuf)
                    for bm in range(tiles_in_block_m):
                        row0 = m0 + bm * TILE_M
                        for i in range(4):
                            off = (bm * 4 + i) * block_kb
                            col0 = i * K_b + kb0
                            nisa.dma_copy(
                                dst=a_i8[0:TILE_M, off:off + block_kb],
                                src=a_hbm[row0:row0 + TILE_M, col0:col0 + block_kb],
                            )

                    a_bf16 = nl.ndarray((TILE_M, a_slab_w), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=a_bf16, src=a_i8)

                    a_t = nl.ndarray((TILE_K, at_slab_w), dtype=nl.bfloat16, buffer=nl.sbuf)
                    for bm in range(tiles_in_block_m):
                        for i in range(4):
                            src_off = (bm * 4 + i) * block_kb
                            for j in range(num_j):
                                t_psum = nl.ndarray((TILE_K, TILE_M), dtype=nl.bfloat16,
                                                    buffer=nl.psum)
                                nisa.nc_transpose(
                                    dst=t_psum,
                                    data=a_bf16[0:TILE_M,
                                                src_off + j * TILE_K:
                                                src_off + (j + 1) * TILE_K],
                                )
                                dst_off = ((bm * 4 + i) * num_j + j) * TILE_M
                                nisa.tensor_copy(
                                    dst=a_t[0:TILE_K, dst_off:dst_off + TILE_M],
                                    src=t_psum,
                                )

                    for j in range(num_j):
                        k0 = kb0 + j * TILE_K

                        b_u8 = nl.ndarray((TILE_K, tile_n), dtype=nl.uint8, buffer=nl.sbuf)
                        nisa.dma_copy(dst=b_u8,
                                      src=b_hbm[k0:k0 + TILE_K, n0:n0 + tile_n])

                        for i in range(4):
                            shift = 2 * i

                            masked = nl.ndarray((TILE_K, tile_n), dtype=nl.uint8, buffer=nl.sbuf)
                            nisa.tensor_scalar(dst=masked, data=b_u8, op0=nl.bitwise_and, operand0=(3 << shift))

                            b_val = nl.ndarray((TILE_K, tile_n), dtype=nl.bfloat16, buffer=nl.sbuf)
                            nisa.tensor_scalar(dst=b_val, data=masked, op0=nl.multiply, operand0=1.0 / float(1 << shift), op1=nl.add, operand1=-1.0)

                            first = (j == 0 and i == 0)
                            for bm in range(tiles_in_block_m):
                                stat_off = ((bm * 4 + i) * num_j + j) * TILE_M
                                nisa.nc_matmul(
                                    dst=acc_psum[0:TILE_M, bm * TILE_N:bm * TILE_N + tile_n],
                                    stationary=a_t[0:TILE_K, stat_off:stat_off + TILE_M],
                                    moving=b_val,
                                    accumulate=(kbb > 0) if first else True,
                                )

                for bm in range(tiles_in_block_m):
                    out_sb = nl.ndarray((TILE_M, tile_n), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=out_sb, src=acc_psum[0:TILE_M, bm * TILE_N:bm * TILE_N + tile_n])
                    row0 = m0 + bm * TILE_M
                    nisa.dma_copy(dst=out_hbm[row0:row0 + TILE_M, n0:n0 + tile_n], src=out_sb)

        return out_hbm


def _largest_divisor(dim: int, candidates) -> int | None:
    for c in candidates:
        if dim % c == 0:
            return c
    return None


_tuner = NkiAutotuner(matmul_int8_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    M, K = a.shape
    K_b, N = b.shape

    if K != 4 * K_b:
        raise ValueError("Incompatible dimensions: A's K must equal 4 * B's K_b")

    if a.dtype != torch.int8:
        a = a.to(torch.int8)
    if b.dtype != torch.uint8:
        b = b.to(torch.uint8)

    tile_n = _largest_divisor(N, (BLOCK_N, 128, 256, TILE_N))
    tiles_in_block_m = _largest_divisor(M, (BLOCK_M, 2 * TILE_M, TILE_M))
    block_kb = _largest_divisor(K_b, (BLOCK_KB, 256, TILE_K))

    if tile_n is None or tiles_in_block_m is None or block_kb is None:
        raise NotImplementedError(
            f"matmul_int8 NKI: shape (M={M}, N={N}, K={K}) unsupported -- "
            f"M must be a multiple of {TILE_M}, N a multiple of 128 and "
            f"K a multiple of {4 * TILE_K}"
        )

    tiles_in_block_m //= TILE_M
    _default = SimpleNamespace(block_size_m=TILE_M * tiles_in_block_m, block_size_k=4 * block_kb,
                               block_size_n=tile_n)
    if autotune:
        _space = [SimpleNamespace(block_size_m=bm, block_size_k=_default.block_size_k,
                                  block_size_n=bn)
                  for bm in (128, 256, 512) for bn in (64, 128, 256, 512)
                  if M % bm == 0 and N % bn == 0]
        if not any(vars(c) == vars(_default) for c in _space):
            _space.append(_default)
        cfg = _tuner.tune_or_cached(
            shape_key=((M, N, K), str(a.dtype)),
            search_space=_space,
            args_fn=lambda cfg: (a, b, cfg.block_size_m // TILE_M, cfg.block_size_k // 4, cfg.block_size_n),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _default
    return matmul_int8_kernel(a, b, cfg.block_size_m // TILE_M, cfg.block_size_k // 4, cfg.block_size_n)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
