"""Two-layer cache helper for cuTile 1.3 autotune workflow.

cuTile 1.3's `ct.tune.exhaustive_search` separates tuning from launching
(unlike the deprecated `autotune_launch`, which did both). To reproduce
the semantics of Triton's `@triton.autotune` — best config cached per
shape, autotune-selected hints actually applied — every cuTile operator
needs the same two cache layers:

  1. `replace_hints` results cached per hint tuple (so repeated launches
     with the same hint reuse the same kernel object — CUDA graph
     friendly, no re-compile churn).

  2. The autotune outcome cached per problem shape (so the engine's
     timing loop's repeated `run(autotune=True)` calls don't each rerun
     the full sweep).

Without these two layers:
  - The autotune-selected `occupancy` (or other hints) is reported via
    `get_last_config()` but never actually applied at launch — final
    `ct.launch` uses cuTile's default hint values.
  - Or, worse, you call `replace_hints` per launch and crash CUDA Graph
    capture from compile churn.
  - Or, the timing loop measures the full autotune sweep repeating
    every iteration, not the kernel's steady-state perf.

Use this helper in every `impl_cutile.py` that has an autotune path:

    from core.cutile_autotune import CutileAutotuner

    @ct.kernel
    def my_kernel(...): ...

    _tuner = CutileAutotuner(my_kernel)

    def run(*args, autotune=False):
        ...
        if autotune:
            cfg = _tuner.tune_or_cached(
                shape_key=(M, N, K),                # caller decides
                search_space=_SEARCH_SPACE,
                stream=stream,
                grid_fn=lambda cfg: (...),
                args_fn=lambda cfg: (...),
                hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            )
        else:
            cfg = _DEFAULT_CONFIG

        kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
        ct.launch(stream, grid, kernel, args)
"""
from __future__ import annotations

from typing import Any, Callable, Hashable, Sequence

import cuda.tile as ct


class CutileAutotuner:
    """Caches `replace_hints` results and autotune outcomes for one kernel.

    Each operator should hold one instance per `@ct.kernel`-decorated
    function. The two caches are independent dicts on the instance, so
    different operators do not share state.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self._kernel_cache: dict[tuple, Any] = {}
        self._tuned_cache: dict[Hashable, Any] = {}

    def kernel_with_hints(self, **hints):
        """Return the kernel with the given hints applied, caching by hints.

        Equivalent to `kernel.replace_hints(**hints)` but memoised so
        repeated calls with the same hints return the *same* kernel
        object. This is required for CUDA graph capture stability —
        creating a fresh kernel per launch breaks graph replay.
        """
        key = tuple(sorted(hints.items()))
        cached = self._kernel_cache.get(key)
        if cached is None:
            cached = self.kernel.replace_hints(**hints)
            self._kernel_cache[key] = cached
        return cached

    def tune_or_cached(
        self,
        *,
        shape_key: Hashable,
        search_space: Sequence,
        stream,
        grid_fn: Callable,
        args_fn: Callable,
        hints_fn: Callable | None = None,
    ):
        """Run `ct.tune.exhaustive_search` on first call per `shape_key`,
        cache the winning `result.best.config`, and return it directly on
        subsequent calls with the same `shape_key`. Mirrors Triton's
        built-in `@triton.autotune(key=[...])` cache semantics.

        `shape_key` is the caller's responsibility — pass a hashable
        tuple of the problem dimensions that affect the optimal config
        (e.g. `(M, N, K)` for matmul, `(seqlen, head_dim, causal)` for
        attention).
        """
        cached = self._tuned_cache.get(shape_key)
        if cached is None:
            result = ct.tune.exhaustive_search(
                search_space, stream,
                grid_fn=grid_fn, kernel=self.kernel,
                args_fn=args_fn, hints_fn=hints_fn,
            )
            cached = result.best.config
            self._tuned_cache[shape_key] = cached
        return cached
