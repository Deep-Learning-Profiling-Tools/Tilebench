from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


if nki is not None:
    @nki.jit
    def cross_entropy_kernel(logits, targets, block_size):
        """Per-row cross-entropy loss against a hard target class.

        Mirrors ``F.cross_entropy(logits, targets, reduction='none')`` in its
        numerically stable form::

            shifted[r, c] = logits[r, c] - max_c logits[r, c]
            loss[r]       = log(sum_c exp(shifted[r, c])) - shifted[r, target[r]]

        Args:
            logits: [B, C] tensor in HBM (fp32 / bf16 / fp16).
            targets: [B, 1] int32 tensor in HBM with the target class per row.

        Returns:
            [B, 1] tensor in HBM with the same dtype as ``logits``.

        Notes:
            * ``shifted[r, target[r]]`` is a gather at a *data-dependent* column,
              which the vector engines cannot address directly. It is computed
              with the standard one-hot trick instead: a free-axis iota row
              ``0..C-1`` is compared against the row's target id, the resulting
              one-hot predicate selects ``shifted`` (0 elsewhere) via
              ``nisa.select_reduce``, and a free-axis sum extracts the single
              surviving column exactly (all other addends are exactly +0.0).
            * ``nisa.tensor_scalar``'s ``operand0`` must be fp32 when it is a
              ``[P, 1]`` tile (an int32 ``operand0`` is rejected by the MLIR
              verifier), so the int32 target ids are converted to fp32 by a
              ``nisa.tensor_copy`` and compared against an fp32 iota. Class ids
              are far below 2^24, so the compare is exact.
            * ``nisa.select_reduce`` is the NKI 0.4.0 replacement for
              ``np.where``; its ``on_false`` operand must be a scalar (or a
              ``[P, 1]`` column), which the constant 0.0 satisfies directly, and
              its ``predicate`` must be integer-typed, hence the ``uint8``
              comparison result.
            * Row blocks are clamped to ``B`` with ``min(...)``, so every tile is
              allocated at its real extent and no load/store masking (nor any
              ``-inf`` / zero padding fed to the max and sum reductions) is
              required. The full class axis is loaded in one tile.
        """
        kernel_assert(len(logits.shape) == 2, "logits must be 2D [B, C]")
        kernel_assert(len(targets.shape) == 2 and targets.shape[1] == 1,
                      "targets must be a [B, 1] column")
        kernel_assert(logits.shape[0] == targets.shape[0],
                      "logits and targets must agree on the batch dimension")

        B, C = logits.shape
        kernel_assert(C <= nl.tile_size.sbuf_fmax, "class axis exceeds the SBUF free dimension")

        hbm_result = nl.ndarray((B, 1), dtype=logits.dtype, buffer=nl.shared_hbm)
        n_row_tiles = div_ceil(B, block_size)

        for row_tile in range(n_row_tiles):
            row_start = row_tile * block_size
            row_size = min(block_size, B - row_start)
            row_end = row_start + row_size

            # DMA cannot convert dtypes: load in the input dtype and let the
            # Vector/Scalar engines widen to fp32 on their way out.
            logits_tile = nl.ndarray((row_size, C), dtype=logits.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=logits_tile, src=logits[row_start:row_end, 0:C])

            targets_tile = nl.ndarray((row_size, 1), dtype=targets.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=targets_tile, src=targets[row_start:row_end, 0:1])

            targets_f32 = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=targets_f32, src=targets_tile)

            # class_iota[r, c] = c on every partition (channel_multiplier=0).
            class_iota = nl.ndarray((row_size, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.iota(dst=class_iota, pattern=[[1, C]], offset=0, channel_multiplier=0)

            # ---- shifted = logits - row_max ------------------------------
            row_max = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=logits_tile, axis=(1,))

            shifted = nl.ndarray((row_size, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=shifted, data=logits_tile, op0=nl.subtract, operand0=row_max)

            # ---- log_sum = log(sum_c exp(shifted)) -----------------------
            exp_tile = nl.ndarray((row_size, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=exp_tile, op=nl.exp, data=shifted)

            row_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=row_sum, op=nl.add, data=exp_tile, axis=(1,))

            log_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=log_sum, op=nl.log, data=row_sum)

            # ---- target_shifted = shifted[r, target[r]] (one-hot gather) --
            target_mask = nl.ndarray((row_size, C), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=target_mask, data=class_iota, op0=nl.equal,
                               operand0=targets_f32)

            target_only = nl.ndarray((row_size, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.select_reduce(dst=target_only, predicate=target_mask, on_true=shifted,
                               on_false=0.0)

            target_shifted = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=target_shifted, op=nl.add, data=target_only, axis=(1,))

            loss = nl.ndarray((row_size, 1), dtype=logits.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=loss, data1=log_sum, data2=target_shifted, op=nl.subtract)

            nisa.dma_copy(dst=hbm_result[row_start:row_end, 0:1], src=loss)

        return hbm_result


_DEFAULT_CONFIG = SimpleNamespace(block_size=128)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (32, 64, 128)]
_tuner = NkiAutotuner(cross_entropy_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(logits: torch.Tensor, targets: torch.Tensor, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    targets_2d = targets.reshape(-1, 1).to(torch.int32)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(logits.shape), str(logits.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (logits, targets_2d, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    result = cross_entropy_kernel(logits, targets_2d, cfg.block_size)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
