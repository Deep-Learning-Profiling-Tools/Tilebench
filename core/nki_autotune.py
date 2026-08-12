"""Autotune helper for the NKI (AWS Neuron / Trainium) backend.

NKI has no official autotune API (no analog of ``@triton.autotune`` or
``ct.tune.exhaustive_search``), so this module provides one, mirroring the
conventions of ``core/cutile_autotune.py``:

  - the operator holds one ``NkiAutotuner`` per ``@nki.jit`` kernel;
  - ``tune_or_cached(shape_key=..., search_space=..., args_fn=...)`` sweeps
    the search space ONCE per ``shape_key`` (the caller includes the input
    dtype in the key, same as the cuTile convention), caches the winning
    config, and returns it on every later call;
  - the operator stores the winner in a module-level mutable dict so
    ``get_last_config()`` can report it (never ``global``).

Unlike Triton (``num_warps``) and cuTile (``occupancy`` hints), NKI has no
launch-config side channel: every tunable is an ordinary kernel argument
(tile sizes, blocking factors, ...). Python ints are compile-time constants
to ``@nki.jit``, so each candidate config triggers one compile — exactly
what a tuning sweep wants. ``args_fn(cfg)`` therefore returns the FULL
argument tuple for the kernel, tunables included:

    from core.nki_autotune import NkiAutotuner

    if nki is not None:
        @nki.jit
        def my_kernel(a_input, TILE_FREE):
            ...
        _tuner = NkiAutotuner(my_kernel)

    _DEFAULT_CONFIG = SimpleNamespace(tile_free=2048)
    _SEARCH_SPACE = [SimpleNamespace(tile_free=t)
                     for t in (512, 1024, 2048, 4096, 8192)]
    _last_autotune_config: dict = {}

    def run(x, block_size=1024, autotune=False, **kwargs):
        if autotune:
            cfg = _tuner.tune_or_cached(
                shape_key=(tuple(x.shape), str(x.dtype)),
                search_space=_SEARCH_SPACE,
                args_fn=lambda cfg: (x, cfg.tile_free),
            )
            _last_autotune_config.clear()
            _last_autotune_config.update(vars(cfg))
        else:
            cfg = _DEFAULT_CONFIG
        return my_kernel(x, cfg.tile_free)

Candidate timing (pluggable via ``timer=``, default ``"auto"``):

  1. ``"benchmark"`` — ``nki.benchmark``: re-decorates the kernel and runs
     it baremetal on a NeuronCore with numpy inputs; the reported
     ``nc_latency`` percentile is true on-device time (the Trainium analog
     of what Proton gives the GPU backends). Preferred.
  2. ``"wallclock"`` — runs the ``@nki.jit`` kernel through torch_xla with
     the original (XLA-device) tensors and takes a host wall-clock median
     around ``mark_step``/``wait_device_ops``. Includes dispatch overhead;
     used only when (1) is unavailable, and still fine for RANKING
     candidates of the same kernel.
  ``"auto"`` tries (1) and falls back to (2).

STATUS: written against the Neuron docs ("NKI performance guide",
``nki.benchmark`` API) but NOT yet run on Neuron hardware — this machine has
no trn1/trn2. Spots most likely to need adjustment on real hardware are
marked ``# VERIFY ON TRN2`` (same convention as core/nki_timer.py):
  - whether ``nki.benchmark`` accepts an already-``@nki.jit``-decorated
    kernel or needs the undecorated function (we try both);
  - the exact ``benchmark_result.nc_latency`` accessor and its unit
    (docs: microseconds);
  - whether failed candidates raise or hang (a per-candidate timeout is
    passed through when supported).
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Hashable, Sequence

import torch


def _to_numpy_args(args: Sequence) -> tuple:
    """torch.Tensor -> numpy (via CPU) for baremetal nki.benchmark runs;
    scalars and non-tensors pass through unchanged."""
    out = []
    for a in args:
        if isinstance(a, torch.Tensor):
            out.append(a.detach().cpu().numpy())
        else:
            out.append(a)
    return tuple(out)


def _kernel_func(kernel):
    """Best-effort unwrap of a ``@nki.jit`` object back to the plain python
    function, for re-decoration with ``nki.benchmark``.  # VERIFY ON TRN2"""
    for attr in ("func", "__wrapped__", "fn"):
        f = getattr(kernel, attr, None)
        if callable(f):
            return f
    return kernel


class NkiAutotuner:
    """Sweep-and-cache autotuner for one ``@nki.jit`` kernel.

    Mirrors ``CutileAutotuner``'s contract: the cache key is entirely the
    caller's responsibility — include every input dimension AND the dtype
    string that affect the optimal config (per the repo-wide per-dtype
    autotune convention).
    """

    def __init__(self, kernel, *, timer: str = "auto",
                 warmup: int = 5, iters: int = 10,
                 quiet: bool = False):
        if timer not in ("auto", "benchmark", "wallclock"):
            raise ValueError(f"unknown timer '{timer}'")
        self.kernel = kernel
        self.timer = timer
        self.warmup = int(warmup)
        self.iters = int(iters)
        self.quiet = quiet
        self._tuned_cache: dict[Hashable, Any] = {}

    # ------------------------------------------------------------------
    # Candidate timing backends
    # ------------------------------------------------------------------
    def _time_benchmark(self, args: Sequence) -> float:
        """On-device latency (ms) via ``nki.benchmark`` (baremetal, numpy in).

        The decorator runs the kernel ``warmup + iters`` times on a
        NeuronCore and records per-execution device latency; we return the
        p50 in ms.  # VERIFY ON TRN2 (accessor name + microsecond unit)
        """
        import neuronxcc.nki as nki

        np_args = _to_numpy_args(args)
        bench = nki.benchmark(warmup=self.warmup, iters=self.iters)
        try:
            timed = bench(self.kernel)
            timed(*np_args)
        except TypeError:
            timed = bench(_kernel_func(self.kernel))
            timed(*np_args)
        lat = timed.benchmark_result.nc_latency
        return float(lat.get_latency_percentile(50)) / 1e3  # us -> ms

    def _time_wallclock(self, args: Sequence) -> float:
        """Host wall-clock median (ms) through torch_xla for XLA-device args."""
        from torch_xla.core import xla_model as xm

        def _once() -> float:
            t0 = time.perf_counter()
            out = self.kernel(*args)
            xm.mark_step()
            xm.wait_device_ops()
            del out
            return (time.perf_counter() - t0) * 1e3

        for _ in range(self.warmup):
            _once()
        samples = sorted(_once() for _ in range(self.iters))
        return samples[len(samples) // 2]

    def _time_candidate(self, args: Sequence) -> float:
        if self.timer == "benchmark":
            return self._time_benchmark(args)
        if self.timer == "wallclock":
            return self._time_wallclock(args)
        try:
            return self._time_benchmark(args)
        except Exception as e:
            if not self.quiet:
                print(f"  nki autotune: nki.benchmark path failed "
                      f"({type(e).__name__}: {e}); falling back to XLA wall-clock")
            self.timer = "wallclock"
            return self._time_wallclock(args)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def tune_or_cached(
        self,
        *,
        shape_key: Hashable,
        search_space: Sequence,
        args_fn: Callable[[Any], Sequence],
    ):
        """Time every config in ``search_space`` once per ``shape_key`` and
        return the fastest; later calls with the same key return the cached
        winner without re-running the sweep (so the engine's timing loop
        measures steady-state kernel perf, not the sweep itself).

        Configs whose compile or execution fails are skipped (invalid tile
        sizes are expected in a search space); if every config fails, the
        first error is re-raised.
        """
        cached = self._tuned_cache.get(shape_key)
        if cached is not None:
            return cached

        best_cfg, best_ms = None, math.inf
        failures: list[tuple[Any, str]] = []
        for cfg in search_space:
            try:
                ms = self._time_candidate(args_fn(cfg))
            except Exception as e:
                failures.append((cfg, f"{type(e).__name__}: {e}"))
                continue
            if not self.quiet:
                print(f"  nki autotune: {cfg} -> {ms:.4f} ms")
            if ms < best_ms:
                best_cfg, best_ms = cfg, ms

        if best_cfg is None:
            raise RuntimeError(
                f"nki autotune: all {len(search_space)} candidate configs "
                f"failed; first error: {failures[0][1] if failures else 'n/a'}"
            )
        if failures and not self.quiet:
            print(f"  nki autotune: skipped {len(failures)} failing config(s), "
                  f"e.g. {failures[0][1][:80]}")
        if not self.quiet:
            print(f"  nki autotune: best {best_cfg} ({best_ms:.4f} ms)")

        self._tuned_cache[shape_key] = best_cfg
        return best_cfg
