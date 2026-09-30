from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# Hardware constants (NeuronCore-v2/v3): SBUF partition count and the maximum
# free-dimension size of an ``nc_matmul`` moving tile (== one fp32 PSUM bank).
P_MAX = 128
MOVING_FMAX = 512

# Additive penalty applied to the log-sum-exp of out-of-range blocks. It has to
# be large enough that ``exp(lse - max)`` underflows to exactly 0 in fp32, while
# staying well inside the fp32 range so the subsequent subtraction cannot
# overflow to -inf (which would produce NaNs in the exponential).
NEG_BIG = 1.0e30


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


if nki is not None:
    @nki.jit
    def flash_decode_kernel(mid_o_t, mid_o_lse, valid_blocks, block_size_seq):
        """Flash-decode stage-2 combine for a single (batch, head) pair.

        Merges the ``num_blocks`` partial attention outputs produced by stage 1
        into a single ``[head_dim, 1]`` result, weighting each block by a
        numerically-stable softmax over its log-sum-exp::

            mask   = arange(num_blocks) < valid_blocks
            lse    = where(mask, mid_o_lse, -inf)
            w      = where(mask, exp(lse - max(lse)), 0)
            out[d] = sum_k mid_o_t[d, k] * w[k] / (sum_k w[k] + 1e-10)

        Args:
            mid_o_t: per-block partial outputs [head_dim, num_blocks] in HBM,
                already transposed so that ``head_dim`` is the partition axis.
            mid_o_lse: per-block log-sum-exp [1, num_blocks] in HBM.
            valid_blocks: [1, 1] fp32 tensor holding the number of blocks that
                actually contain data for this sequence. Its value is a runtime
                (not compile-time) quantity, so the masking is done with an
                ``iota``/compare on device rather than by sizing the tiles.

        Returns:
            [head_dim, 1] tensor in HBM with the same dtype as ``mid_o_t``.

        Notes:
            * Everything up to the softmax lives on a single partition
              (``[1, num_blocks]`` tiles), so ``valid_blocks`` and the running
              ``max``/``sum`` are ``[1, 1]`` tiles fed to ``tensor_scalar`` as
              per-partition scalar operands -- the hardware broadcasts them
              along the free axis for free.
            * The normalized weights do need a genuine *partition* broadcast
              (1 -> ``head_dim`` partitions) before they can be multiplied into
              ``mid_o_t``. That is done with a single ``nc_matmul`` against a
              row of ones (``ones[1, head_dim]^T @ w[1, K] -> [head_dim, K]``),
              chunked to the 512-wide PSUM bank.
            * Dividing by the softmax denominator is folded into the weights
              while they are still a ``[1, num_blocks]`` row, so no second
              partition broadcast is needed for it.
        """
        kernel_assert(len(mid_o_t.shape) == 2, "mid_o_t must be 2D [head_dim, num_blocks]")
        head_dim, num_blocks = mid_o_t.shape
        kernel_assert(head_dim <= P_MAX, "head_dim exceeds the SBUF partition count")
        kernel_assert(num_blocks <= nl.tile_size.sbuf_fmax,
                      "num_blocks exceeds the SBUF free dimension")

        out_hbm = nl.ndarray((head_dim, 1), dtype=mid_o_t.dtype, buffer=nl.shared_hbm)

        n_blk_tiles = div_ceil(num_blocks, block_size_seq)

        # ---- block index row: 0, 1, ..., num_blocks - 1 (free axis) ---------
        block_iota = nl.ndarray((1, num_blocks), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=block_iota, pattern=[[1, num_blocks]], offset=0, channel_multiplier=0)

        block_iota_f = nl.ndarray((1, num_blocks), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=block_iota_f, src=block_iota)

        # ---- mask = block_index < valid_blocks (1.0 / 0.0) ------------------
        valid_raw = nl.ndarray((1, 1), dtype=valid_blocks.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=valid_raw, src=valid_blocks[0:1, 0:1])

        valid_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=valid_f, src=valid_raw)

        mask = nl.ndarray((1, num_blocks), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mask, data=block_iota_f, op0=nl.less, operand0=valid_f)

        # ---- masked log-sum-exp row ----------------------------------------
        lse_raw = nl.ndarray((1, num_blocks), dtype=mid_o_lse.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=lse_raw, src=mid_o_lse[0:1, 0:num_blocks])

        lse_f = nl.ndarray((1, num_blocks), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lse_f, src=lse_raw)

        # masked_lse = lse * mask + (mask - 1) * NEG_BIG
        #            = lse            where valid
        #            = -NEG_BIG       where invalid
        penalty = nl.ndarray((1, num_blocks), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=penalty, data=mask,
                           op0=nl.subtract, operand0=1.0,
                           op1=nl.multiply, operand1=NEG_BIG)

        masked_lse = nl.ndarray((1, num_blocks), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=masked_lse, data1=lse_f, data2=mask, op=nl.multiply)
        nisa.tensor_tensor(dst=masked_lse, data1=masked_lse, data2=penalty, op=nl.add)

        # ---- stable softmax over the valid blocks ---------------------------
        global_max = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=global_max, op=nl.maximum, data=masked_lse, axis=(1,))

        shifted = nl.ndarray((1, num_blocks), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=shifted, data=masked_lse, op0=nl.subtract, operand0=global_max)

        weights = nl.ndarray((1, num_blocks), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=weights, op=nl.exp, data=shifted)
        # Re-mask: exp() of the penalized entries already underflows to 0, but
        # forcing it keeps the semantics exact regardless of the lse magnitudes.
        nisa.tensor_tensor(dst=weights, data1=weights, data2=mask, op=nl.multiply)

        denom = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=denom, op=nl.add, data=weights, axis=(1,))
        nisa.tensor_scalar(dst=denom, data=denom, op0=nl.add, operand0=1.0e-10)

        inv_denom = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_denom, data=denom)

        # Fold the normalization into the weight row while it is still a single
        # partition, so only one partition broadcast is needed below.
        nisa.tensor_scalar(dst=weights, data=weights, op0=nl.multiply, operand0=inv_denom)

        # ---- combine: out[d] = sum_k mid_o_t[d, k] * weights[k] -------------
        ones_row = nl.ndarray((1, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_row, value=1.0)

        acc = nl.ndarray((head_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)

        for blk_tile in range(n_blk_tiles):
            blk_start = blk_tile * block_size_seq
            blk_size = min(block_size_seq, num_blocks - blk_start)
            blk_end = blk_start + blk_size

            # partition broadcast: [1, blk_size] -> [head_dim, blk_size]
            psum_w = nl.ndarray((head_dim, blk_size), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=psum_w, stationary=ones_row,
                           moving=weights[0:1, blk_start:blk_end])

            weights_bcast = nl.ndarray((head_dim, blk_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=weights_bcast, src=psum_w)

            mo_tile = nl.ndarray((head_dim, blk_size), dtype=mid_o_t.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=mo_tile, src=mid_o_t[0:head_dim, blk_start:blk_end])

            weighted = nl.ndarray((head_dim, blk_size), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=weighted, data1=mo_tile, data2=weights_bcast,
                               op=nl.multiply)

            part = nl.ndarray((head_dim, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=part, op=nl.add, data=weighted, axis=(1,))
            nisa.tensor_tensor(dst=acc, data1=acc, data2=part, op=nl.add)

        out_tile = nl.ndarray((head_dim, 1), dtype=mid_o_t.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_tile, src=acc)
        nisa.dma_copy(dst=out_hbm[0:head_dim, 0:1], src=out_tile)

        return out_hbm


_DEFAULT_CONFIG = SimpleNamespace(block_size_seq=MOVING_FMAX)
_SEARCH_SPACE = [SimpleNamespace(block_size_seq=b) for b in (128, 256, 512)]
_tuner = NkiAutotuner(flash_decode_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(mid_o: torch.Tensor, mid_o_lse: torch.Tensor, b_seqlen: torch.Tensor,
        block_seq, block_size: int = None, autotune: bool = False, **kwargs) -> torch.Tensor:
    if isinstance(block_seq, torch.Tensor):
        block_seq = block_seq.item()

    batch, heads, num_blocks, head_dim = mid_o.shape
    valid_blocks_count = ((b_seqlen + block_seq - 1) // block_seq).to(torch.float32).reshape(batch, 1, 1)

    mid_o_t = mid_o.permute(0, 1, 3, 2).contiguous()
    lse_2d = mid_o_lse.reshape(batch, heads, 1, num_blocks)

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(mid_o_t.shape), str(mid_o_t.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (mid_o_t[0, 0], lse_2d[0, 0], valid_blocks_count[0], cfg.block_size_seq),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    out = torch.empty(batch, heads, head_dim, dtype=mid_o.dtype, device=mid_o.device)
    for b in range(batch):
        for h in range(heads):
            res = flash_decode_kernel(mid_o_t[b, h], lse_2d[b, h], valid_blocks_count[b], cfg.block_size_seq)
            out[b, h, :] = res.reshape(-1)
    return out


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
