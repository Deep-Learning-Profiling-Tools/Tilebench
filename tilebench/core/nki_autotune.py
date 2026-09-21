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

    from tilebench.core.nki_autotune import NkiAutotuner

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

  1. ``"benchmark"`` — the standalone (baremetal) path of the ``nki``
     package (>= 0.6): the ``@nki.jit`` kernel is compiled with numpy
     inputs and timed with ``CompiledKernel.benchmark`` (device mode),
     whose ``latency`` is true on-device time (the Trainium analog of what
     Proton gives the GPU backends). Preferred. NOTE: the legacy
     ``neuronxcc.nki.benchmark`` decorator this module originally called
     predates the standalone ``nki`` package and rejects its Kernel
     objects (``'Kernel' object has no attribute 'grid'`` on trn2) — do
     not reintroduce it.
  2. ``"wallclock"`` — runs the ``@nki.jit`` kernel through torch_xla with
     the original (XLA-device) tensors and takes a host wall-clock median
     around ``mark_step``/``wait_device_ops``. Includes dispatch overhead
     (~1 ms host round-trip per launch), which drowns out sub-millisecond
     kernels — for those, its ranking is unreliable; it remains a last
     resort (e.g. bf16 inputs, which numpy cannot represent).
  ``"auto"`` tries (1) and falls back to (2).

STATUS: verified on a trn2.3xlarge (nki 0.6.0, neuronx-cc 2.27): the
benchmark path executes on-device and its latency agrees with the
neuron-profile numbers from core/nki_timer.py; the wallclock fallback was
also exercised (it ranks millisecond-scale kernels fine, sub-millisecond
ones poorly — see above).
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import math
import time
from types import SimpleNamespace
from typing import Any, Callable, Hashable, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Canonical serialization (winner identity across processes)
# ---------------------------------------------------------------------------
#
# The profiling flow (see core/nki_orchestrator.py) replays an autotune winner
# in a FRESH python process, so both the shape key and the winning config must
# have a deterministic, JSON-compatible representation with no dependence on
# python object identity, memory addresses, or repr() stability.


class NkiAutotuneSerializationError(TypeError):
    """A shape_key or config cannot be canonically serialized."""


class NkiAutotuneReplayError(RuntimeError):
    """Replay mode could not map a recorded winner onto the live tuner state."""


_CANON_SCALARS = (type(None), bool, int, float, str)


def canonical_shape_key(key: Any) -> Any:
    """Recursively convert a shape key into a canonical JSON-compatible value.

    Supported: None, bool, int, float, str, tuple, list, dict (str keys).
    Tuples become lists (JSON has no tuples). Anything else fails loudly —
    never fall back to repr().
    """
    if isinstance(key, _CANON_SCALARS):
        return key
    if isinstance(key, (tuple, list)):
        return [canonical_shape_key(v) for v in key]
    if isinstance(key, dict):
        out = {}
        for k in sorted(key):
            if not isinstance(k, str):
                raise NkiAutotuneSerializationError(
                    f"shape_key dict keys must be str, got {type(k).__name__}: {k!r}")
            out[k] = canonical_shape_key(key[k])
        return out
    raise NkiAutotuneSerializationError(
        f"unsupported shape_key element of type {type(key).__name__}: {key!r}")


def canonical_config(cfg: Any) -> dict:
    """Convert an autotune config object into a canonical sorted dict.

    Supported: dict, dataclass instance, SimpleNamespace, namedtuple, and
    ordinary objects whose public __dict__ fields are canonically
    serializable. Unsupported objects fail loudly.
    """
    if isinstance(cfg, dict):
        mapping = cfg
    elif dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        mapping = dataclasses.asdict(cfg)
    elif isinstance(cfg, SimpleNamespace):
        mapping = vars(cfg)
    elif isinstance(cfg, tuple) and hasattr(cfg, "_asdict"):
        mapping = cfg._asdict()
    elif hasattr(cfg, "__dict__"):
        mapping = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
        if not mapping:
            raise NkiAutotuneSerializationError(
                f"config object {type(cfg).__name__} has no public fields to serialize")
    else:
        raise NkiAutotuneSerializationError(
            f"unsupported config type {type(cfg).__name__}: cannot canonicalize")
    out = {}
    for k in sorted(mapping):
        if not isinstance(k, str):
            raise NkiAutotuneSerializationError(
                f"config field names must be str, got {type(k).__name__}: {k!r}")
        out[k] = canonical_shape_key(mapping[k])
    return out


def _canon_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Winner trace + replay registry (module level, process global)
# ---------------------------------------------------------------------------

_TUNING_TRACE: list[dict] = []
_REPLAY: dict | None = None


def clear_tuning_trace() -> None:
    """Reset the recorded winner trace (call before a selector run)."""
    _TUNING_TRACE.clear()


def export_tuning_trace() -> list[dict]:
    """Deep copy of every {tuner_name, shape_key, config} winner record."""
    return copy.deepcopy(_TUNING_TRACE)


def _record_trace(tuner_name: str, key_c: Any, cfg_c: dict) -> None:
    rec = {"tuner_name": tuner_name, "shape_key": key_c, "config": cfg_c}
    rec_j = _canon_json(rec)
    if any(_canon_json(r) == rec_j for r in _TUNING_TRACE):
        return
    _TUNING_TRACE.append(copy.deepcopy(rec))


def install_tuning_replay(records: Sequence[dict], strict: bool = True) -> None:
    """Install winner records; subsequent tune_or_cached calls replay them.

    Each record: {"tuner_name": str, "shape_key": canonical, "config": canonical}.
    Two records for the same (tuner_name, shape_key) with different configs are
    rejected; identical duplicates are collapsed.
    """
    global _REPLAY
    validated: list[dict] = []
    seen: dict[str, str] = {}  # (tuner,key) json -> config json
    for i, r in enumerate(records):
        if not isinstance(r, dict) or not isinstance(r.get("tuner_name"), str):
            raise NkiAutotuneReplayError(f"replay record {i} malformed: {r!r}")
        if "shape_key" not in r or "config" not in r:
            raise NkiAutotuneReplayError(f"replay record {i} missing shape_key/config: {r!r}")
        key_c = canonical_shape_key(r["shape_key"])
        cfg_c = canonical_config(r["config"])
        ident = _canon_json([r["tuner_name"], key_c])
        cfg_j = _canon_json(cfg_c)
        if ident in seen:
            if seen[ident] != cfg_j:
                raise NkiAutotuneReplayError(
                    f"conflicting replay records for tuner/key {ident}: "
                    f"{seen[ident]} vs {cfg_j}")
            continue  # identical duplicate
        seen[ident] = cfg_j
        validated.append({"tuner_name": r["tuner_name"], "shape_key": key_c, "config": cfg_c})
    _REPLAY = {"records": validated, "strict": bool(strict),
               "consumed": [False] * len(validated)}


def clear_tuning_replay() -> None:
    global _REPLAY
    _REPLAY = None


def replay_active() -> bool:
    return _REPLAY is not None


def assert_tuning_replay_consumed() -> None:
    """Fail if any installed replay record was never used (stale winner spec)."""
    if _REPLAY is None:
        raise NkiAutotuneReplayError("assert_tuning_replay_consumed: no replay installed")
    unused = [r for r, c in zip(_REPLAY["records"], _REPLAY["consumed"]) if not c]
    if unused:
        raise NkiAutotuneReplayError(
            f"{len(unused)} replay record(s) were never consumed "
            f"(changed code path or stale winner spec): "
            + "; ".join(_canon_json(r) for r in unused))


@contextlib.contextmanager
def tuning_replay(records: Sequence[dict], strict: bool = True):
    """Context manager: install replay, assert full consumption on clean exit."""
    install_tuning_replay(records, strict=strict)
    try:
        yield
        assert_tuning_replay_consumed()
    finally:
        clear_tuning_replay()


def _default_tuner_name(kernel) -> str:
    f = _kernel_func(kernel)
    mod = getattr(f, "__module__", None)
    qn = getattr(f, "__qualname__", getattr(f, "__name__", None))
    if not mod or not qn:
        raise ValueError(
            "cannot derive a stable tuner name from the kernel; "
            "pass NkiAutotuner(kernel, name='explicit_stable_name')")
    return f"{mod}.{qn}"


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
                 quiet: bool = False, name: str | None = None):
        if timer not in ("auto", "benchmark", "wallclock"):
            raise ValueError(f"unknown timer '{timer}'")
        self.kernel = kernel
        # Stable cross-process identity used by the winner trace / replay flow.
        self.name = name or _default_tuner_name(kernel)
        self.timer = timer
        self.warmup = int(warmup)
        self.iters = int(iters)
        self.quiet = quiet
        self._tuned_cache: dict[str, Any] = {}  # canonical shape-key JSON -> config

    # ------------------------------------------------------------------
    # Candidate timing backends
    # ------------------------------------------------------------------
    def _time_benchmark(self, args: Sequence) -> float:
        """On-device latency (ms) via the standalone path of ``nki`` >= 0.6.

        The kernel is compiled baremetal with numpy inputs; a custom
        executor swaps ``CompiledKernel.execute`` for
        ``CompiledKernel.benchmark``, which runs ``warmup + iterations``
        times on a NeuronCore and reports mean device latency in seconds.
        """
        import dataclasses
        import inspect

        from nki.framework.compiled import StandaloneKernel

        np_args = _to_numpy_args(args)
        # benchmark() takes input arrays by kernel-parameter name.
        param_names = list(inspect.signature(_kernel_func(self.kernel)).parameters)
        tensor_inputs = {
            name: a for name, a in zip(param_names, np_args)
            if isinstance(a, np.ndarray)
        }

        holder: dict = {}

        def _bench_executor(compiled, exec_inputs, output_arrays,
                            rank_id=0, world_size=1):
            res = compiled.benchmark(
                warmup=self.warmup, iterations=self.iters,
                rank_id=rank_id, world_size=world_size, **tensor_inputs,
            )
            holder["result"] = res
            for name, arr in res.outputs.items():
                if name in output_arrays:
                    np.copyto(output_arrays[name], arr)

        standalone = dataclasses.replace(
            self.kernel._to_subclass(StandaloneKernel), _executor=_bench_executor
        )
        standalone(*np_args)
        return float(holder["result"].latency) * 1e3  # seconds -> ms

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

        Replay mode: when ``install_tuning_replay`` is active, the recorded
        winner for (tuner name, shape_key) is looked up in ``search_space`` by
        canonical equality and returned WITHOUT timing any candidate. The
        explicit replay record is authoritative — it is checked before the
        in-process tuned cache.
        """
        key_c = canonical_shape_key(shape_key)
        # Cache by the canonical JSON: every shape_key form canonical_shape_key
        # accepts (including lists/dicts, which are unhashable) must work.
        cache_key = _canon_json(key_c)

        if _REPLAY is not None:
            replayed = self._replay_lookup(key_c, search_space)
            if replayed is not _NO_REPLAY_RECORD:
                return replayed
            # non-strict replay with no record for this tuner/key: tune normally.

        cached = self._tuned_cache.get(cache_key)
        if cached is not None:
            _record_trace(self.name, key_c, canonical_config(cached))
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

        # Serializability is enforced at tune time so the winner is always
        # exactly replayable in a fresh process.
        _record_trace(self.name, key_c, canonical_config(best_cfg))
        self._tuned_cache[cache_key] = best_cfg
        return best_cfg

    def _replay_lookup(self, key_c, search_space: Sequence):
        """Map the installed replay record for (self.name, key_c) onto
        ``search_space`` by canonical config equality. Requires exactly one
        match; never times a candidate."""
        assert _REPLAY is not None
        ident = _canon_json([self.name, key_c])
        matches = [i for i, r in enumerate(_REPLAY["records"])
                   if _canon_json([r["tuner_name"], r["shape_key"]]) == ident]
        if not matches:
            if _REPLAY["strict"]:
                known = [_canon_json([r["tuner_name"], r["shape_key"]])
                         for r in _REPLAY["records"]]
                raise NkiAutotuneReplayError(
                    f"strict replay: no record for tuner {self.name!r} with "
                    f"shape_key {_canon_json(key_c)}; installed records: {known}")
            return _NO_REPLAY_RECORD
        # install_tuning_replay collapses duplicates and rejects conflicts, so
        # at most one record can match here.
        idx = matches[0]
        want = _canon_json(_REPLAY["records"][idx]["config"])
        cands = [c for c in search_space if _canon_json(canonical_config(c)) == want]
        if not cands:
            raise NkiAutotuneReplayError(
                f"replay: recorded winner config {want} for tuner {self.name!r} "
                f"no longer exists in the search space "
                f"({len(search_space)} candidates)")
        if len(cands) > 1:
            raise NkiAutotuneReplayError(
                f"replay: {len(cands)} search-space candidates canonicalize to "
                f"the same recorded config {want} for tuner {self.name!r}; "
                f"identity is ambiguous")
        _REPLAY["consumed"][idx] = True
        _record_trace(self.name, key_c, _REPLAY["records"][idx]["config"])
        return cands[0]


# Sentinel: non-strict replay found no record for this tuner/key.
_NO_REPLAY_RECORD = object()
