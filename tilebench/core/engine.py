import importlib
import inspect
import os

import torch
import yaml
from tilebench.core.dtypes import resolve_dtype
from tilebench.core.timer import report_benchmark
from tilebench.core.verifier import verify
from tilebench.data.tensors import expand_cases, get_generator, infer_problem_size
from tilebench.paths import operator_config


# CUDA is required for the GPU-only backends (Triton, cuTile, TileLang) and for
# Proton-based torch timing. On non-CUDA hosts (e.g. AWS Trainium) those are
# skipped; torch and NKI are instead both timed on the XLA (Neuron) device via
# the Neuron runtime's execution trace (see tilebench/core/nki_timer.py).
HAS_CUDA = torch.cuda.is_available()

# On Neuron hosts every benchmark entry point (scripts/run_bench.py and
# scripts/run_bench_all.py both go through this module) holds ONE logical
# NeuronCore: the Trainium2 peak numbers in tilebench/data/peak_performance are per
# logical core. setdefault keeps an explicit user setting.
if not HAS_CUDA:
    os.environ.setdefault("NEURON_RT_NUM_CORES", "1")


def _nki_artifact_paths(logs_dir) -> dict:
    """Where the NKI profiling flow keeps its artifacts: inside the run's result
    namespace, results/<gpu>/logs/, next to the timing and autotune logs."""
    if logs_dir is None:
        raise ValueError(
            "NKI profiling stores its artifacts under results/<gpu>/logs/; call "
            "run_benchmark_suite(..., logs_dir=tilebench.paths.results_logs_dir(<gpu>))")
    return {"base_dir": os.path.join(logs_dir, "nki_profiles"),
            "index_path": os.path.join(logs_dir, "nki_neff_manifest.jsonl")}


def _sync():
    """Synchronize the CUDA device when present; no-op on non-CUDA hosts."""
    if HAS_CUDA:
        torch.cuda.synchronize()


def run_benchmark_suite(operator_name, benchmark_overrides=None, enabled_backends=None,
                        logs_dir=None):
    # Only the Neuron profiling flow writes files from inside the engine; every
    # other output is written by the caller. Resolved up front so a missing
    # namespace fails before any case runs, not once per case.
    nki_paths = None if HAS_CUDA else _nki_artifact_paths(logs_dir)

    config_path = operator_config(operator_name)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Which tile-language backends to run this invocation. torch always runs —
    # it is the speedup baseline. None → all (backward-compatible default).
    if enabled_backends is None:
        enabled_backends = {"triton", "cutile", "tilelang", "nki"}
    else:
        enabled_backends = set(enabled_backends)

    impl_torch = importlib.import_module(f"tilebench.benchmarks.operators.{operator_name}.impl_torch")

    if "triton" in enabled_backends:
        impl_triton = importlib.import_module(f"tilebench.benchmarks.operators.{operator_name}.impl_triton")
    else:
        impl_triton = None
        print("  Triton not selected (--tile-language)")

    if "cutile" in enabled_backends:
        try:
            impl_cutile = importlib.import_module(f"tilebench.benchmarks.operators.{operator_name}.impl_cutile")
        except ImportError as e:
            print(f"  cuTile import skipped: {e}")
            impl_cutile = None
    else:
        impl_cutile = None
        print("  cuTile not selected (--tile-language)")

    if "tilelang" in enabled_backends:
        try:
            impl_tilelang = importlib.import_module(f"tilebench.benchmarks.operators.{operator_name}.impl_tilelang")
        except ImportError as e:
            print(f"  TileLang import skipped: {e}")
            impl_tilelang = None
    else:
        impl_tilelang = None
        print("  TileLang not selected (--tile-language)")

    if "nki" in enabled_backends:
        try:
            impl_nki = importlib.import_module(f"tilebench.benchmarks.operators.{operator_name}.impl_nki")
            # Conforming impls set a module-level `nki = None` when the Neuron SDK is
            # missing (see NKI authoring guide §3); treat that like an absent backend.
            if getattr(impl_nki, "nki", object()) is None:
                print("  NKI import skipped: Neuron SDK not installed")
                impl_nki = None
        except ImportError as e:
            print(f"  NKI import skipped: {e}")
            impl_nki = None
    else:
        impl_nki = None
        print("  NKI not selected (--tile-language)")

    generate_inputs = get_generator(operator_name)

    results = []
    bench_cfg = config.get("benchmark", {}).copy()
    if benchmark_overrides:
        bench_cfg.update(benchmark_overrides)

    warmup            = int(bench_cfg.get("warmup", 20))
    repeat            = int(bench_cfg.get("repeat", 100))
    autotune          = bool(bench_cfg.get("autotune", False))

    verify_cfg = config.get("verify", {})
    verify_atol = float(verify_cfg["atol"]) if "atol" in verify_cfg else None
    verify_rtol = float(verify_cfg["rtol"]) if "rtol" in verify_cfg else None
    use_cuda_graph    = bool(bench_cfg.get("use_cuda_graph", False))
    proton_scope_name = str(bench_cfg.get("proton_scope_name", "launch"))
    proton_backend    = bench_cfg.get("proton_backend")
    proton_context    = bench_cfg.get("proton_context", "shadow")
    flush_l2          = bool(bench_cfg.get("flush_l2", False))
    keep_proton_files = bool(bench_cfg.get("keep_proton_files", False))
    proton_output_dir = bench_cfg.get("proton_output_dir")

    cases = expand_cases(operator_name, config)
    case_indices = bench_cfg.get("case_indices")
    if case_indices is not None:
        selected = []
        for idx in case_indices:
            if idx < 0 or idx >= len(cases):
                raise ValueError(f"case index {idx} out of range for {len(cases)} generated cases")
            selected.append(cases[idx])
        cases = selected

    def _bench(fn, args, kw=None, *, label: str | None = None):
        return report_benchmark(
            fn,
            args,
            kwargs=kw or {},
            warmup=warmup,
            repeat=repeat,
            use_cuda_graph=use_cuda_graph,
            proton_scope_name=proton_scope_name,
            proton_context=proton_context,
            proton_backend=proton_backend,
            flush_l2=flush_l2,
            keep_proton_files=keep_proton_files,
            proton_output_dir=proton_output_dir,
            proton_file_label=label,
        )

    def _run_kwargs(fn, block_size):
        sig = inspect.signature(fn).parameters
        kw = {}
        if "block_size" in sig:
            kw["block_size"] = block_size
        if "autotune" in sig:
            kw["autotune"] = autotune
        return kw

    for case_idx, case in enumerate(cases):
        params     = {k: v for k, v in case.items() if k not in ("dtype", "block_size")}
        dtype_str  = case.get("dtype", "fp32")
        dtype      = resolve_dtype(dtype_str)
        block_size = case.get("block_size", 1024)

        print(f"Running case: {params}, dtype={dtype_str}")

        try:
            inputs = generate_inputs(**params, dtype=dtype)
        except (RuntimeError, TypeError) as e:
            print(f"  Skipped: dtype={dtype_str} not supported for input generation ({type(e).__name__}: {e})")
            continue

        try:
            ref_output = impl_torch.run(*inputs)
            _sync()
        except (RuntimeError, TypeError) as e:
            print(f"  Skipped: dtype={dtype_str} not supported by torch ({type(e).__name__}: {e})")
            continue

        lbl = f"{operator_name}_{dtype_str}_c{case_idx:03d}"

        # --- Torch (baseline) + NKI on non-CUDA hosts ---
        # Proton-timed on CUDA. On non-CUDA hosts (e.g. Trainium), Proton can't
        # observe the device; the torch baseline and (when selected) the NKI
        # run() are both hardware-timed from the Neuron runtime's execution
        # trace (real inputs, warmup + repeat iterations, multi-graph run()s
        # summed), every executed NEFF matched by SHA256 to the compiled
        # artifact resolved with deterministic identity (private per-spec
        # working dir + compile cache, exact autotune-winner replay in a fresh
        # process, AwsNeuronCustomNativeKernel marker validation, and a
        # per-spec manifest). See core/nki_orchestrator.py. Nothing here is
        # timed by XLA wall-clock.
        neuron_result = None
        if HAS_CUDA:
            torch_stats = _bench(impl_torch.run, inputs, label=f"{lbl}_torch")
            torch_ms    = torch_stats["mean"]
        else:
            try:
                from tilebench.core.nki_orchestrator import profile_case_on_neuron

                neuron_result = profile_case_on_neuron(
                    operator=operator_name, params=params, dtype_str=dtype_str,
                    inputs=inputs, ref_output=ref_output, impl_nki=impl_nki,
                    block_size=block_size, autotune=autotune,
                    verify_atol=verify_atol, verify_rtol=verify_rtol,
                    warmup=warmup, repeat=repeat, **nki_paths)
                torch_stats = neuron_result["torch_stats"]
                torch_ms    = neuron_result["torch_ms"]
                if neuron_result.get("torch_err"):
                    print(f"  Torch (Neuron) baseline FAILED: {neuron_result['torch_err']}")
            except Exception as e:
                print(f"  Neuron (torch+NKI) profiling FAILED: {e}")
                neuron_result = {
                    "torch_stats": None, "torch_ms": float("nan"),
                    "nki_stats": None, "nki_ms": float("nan"),
                    "nki_ok": False, "nki_err": str(e), "nki_cfg": None,
                }
                torch_stats = None
                torch_ms    = float("nan")

        # --- Triton ---
        triton_cfg = None
        if impl_triton is None or not HAS_CUDA:
            triton_ok = False
            triton_err = ("Triton requires CUDA (host has no NVIDIA GPU)"
                          if impl_triton is not None else "Triton not selected (--tile-language)")
            triton_ms = float("nan")
            triton_stats = None
        else:
            try:
                triton_kw = _run_kwargs(impl_triton.run, block_size)
                triton_output = impl_triton.run(*inputs, **triton_kw)
                _sync()
                triton_ok, triton_err = verify(triton_output, ref_output, atol=verify_atol, rtol=verify_rtol)
                triton_cfg = getattr(impl_triton, "get_last_config", lambda: None)() if autotune else None
                if triton_cfg:
                    print(f"  Triton autotune → {triton_cfg}")
                if not triton_ok:
                    print(f"  Triton verification FAILED: {triton_err}")
                triton_stats = (
                    _bench(impl_triton.run, inputs, triton_kw, label=f"{lbl}_triton")
                    if triton_ok else None
                )
                triton_ms = triton_stats["mean"] if triton_stats is not None else float("nan")
            except Exception as e:
                triton_ok    = False
                triton_err   = str(e)
                triton_ms    = float("nan")
                triton_stats = None
                print(f"  Triton execution FAILED: {triton_err}")

        # --- cuTile ---
        cutile_cfg = None
        if impl_cutile is None or not HAS_CUDA:
            cutile_ok = False
            cutile_err = ("cuTile requires CUDA (host has no NVIDIA GPU)"
                          if impl_cutile is not None else "cuTile not available (import failed)")
            cutile_ms = float("nan")
            cutile_stats = None
            print(f"  cuTile skipped ({cutile_err})")
        else:
            try:
                cutile_kw = _run_kwargs(impl_cutile.run, block_size)
                cutile_output = impl_cutile.run(*inputs, **cutile_kw)
                _sync()
                cutile_ok, cutile_err = verify(cutile_output, ref_output, atol=verify_atol, rtol=verify_rtol)
                cutile_cfg = getattr(impl_cutile, "get_last_config", lambda: None)() if autotune else None
                if cutile_cfg:
                    print(f"  cuTile  autotune → {cutile_cfg}")
                if not cutile_ok:
                    print(f"  cuTile verification FAILED: {cutile_err}")
                cutile_stats = (
                    _bench(impl_cutile.run, inputs, cutile_kw, label=f"{lbl}_cutile")
                    if cutile_ok else None
                )
                cutile_ms = cutile_stats["mean"] if cutile_stats is not None else float("nan")
            except Exception as e:
                cutile_ok    = False
                cutile_err   = str(e)
                cutile_ms    = float("nan")
                cutile_stats = None
                print(f"  cuTile execution FAILED: {cutile_err}")

        # --- TileLang ---
        tilelang_cfg = None
        if impl_tilelang is None or not HAS_CUDA:
            tilelang_ok = False
            tilelang_err = ("TileLang requires CUDA (host has no NVIDIA GPU)"
                            if impl_tilelang is not None else "TileLang not available (import failed)")
            tilelang_ms = float("nan")
            tilelang_stats = None
            print(f"  TileLang skipped ({tilelang_err})")
        else:
            try:
                tilelang_kw = _run_kwargs(impl_tilelang.run, block_size)
                tilelang_output = impl_tilelang.run(*inputs, **tilelang_kw)
                _sync()
                tilelang_ok, tilelang_err = verify(tilelang_output, ref_output, atol=verify_atol, rtol=verify_rtol)
                tilelang_cfg = getattr(impl_tilelang, "get_last_config", lambda: None)() if autotune else None
                if tilelang_cfg:
                    print(f"  TileLang autotune → {tilelang_cfg}")
                if not tilelang_ok:
                    print(f"  TileLang verification FAILED: {tilelang_err}")
                tilelang_stats = (
                    _bench(impl_tilelang.run, inputs, tilelang_kw, label=f"{lbl}_tilelang")
                    if tilelang_ok else None
                )
                tilelang_ms = tilelang_stats["mean"] if tilelang_stats is not None else float("nan")
            except Exception as e:
                tilelang_ok    = False
                tilelang_err   = str(e)
                tilelang_ms    = float("nan")
                tilelang_stats = None
                print(f"  TileLang execution FAILED: {tilelang_err}")

        # --- NKI (AWS Trainium; exact-NEFF neuron-profile timing) ---
        # The heavy lifting happened in the orchestrator call above (fresh
        # selector + profile worker processes, exact winner replay, validated
        # artifact identity). Here we only unpack its result. An identity
        # failure is a benchmark failure: nki_ok=False, nki_ms=nan, detailed
        # nki_err — never a latency from an uncertain artifact.
        nki_cfg = None
        if impl_nki is None:
            nki_ok = False
            nki_err = "NKI not available (no impl or Neuron SDK not installed)"
            nki_ms = float("nan")
            nki_stats = None
            print("  NKI skipped (not available)")
        elif neuron_result is None:
            # CUDA host: torch-xla/Neuron runtime is not present.
            nki_ok = False
            nki_err = "NKI timing requires a Neuron (non-CUDA) host"
            nki_ms = float("nan")
            nki_stats = None
            print(f"  NKI skipped ({nki_err})")
        else:
            nki_ok    = neuron_result["nki_ok"]
            nki_err   = neuron_result["nki_err"]
            nki_ms    = neuron_result["nki_ms"]
            nki_stats = neuron_result["nki_stats"]
            nki_cfg   = neuron_result["nki_cfg"] if autotune else None
            if nki_cfg:
                print(f"  NKI     autotune → {nki_cfg}")
            if not nki_ok:
                print(f"  NKI FAILED: {nki_err}")

        results.append({
            "params":                params,
            "problem_size":          infer_problem_size(operator_name, params),
            "dtype":                 dtype_str,
            "torch_ms":              torch_ms,
            "torch_stats":           torch_stats,
            "triton_ms":             triton_ms,
            "triton_stats":          triton_stats or {},
            "triton_ok":             triton_ok,
            "triton_err":            triton_err,
            "triton_autotune_cfg":   triton_cfg,
            "cutile_ms":             cutile_ms,
            "cutile_stats":          cutile_stats or {},
            "cutile_ok":             cutile_ok,
            "cutile_err":            cutile_err,
            "cutile_autotune_cfg":   cutile_cfg,
            "tilelang_ms":           tilelang_ms,
            "tilelang_stats":        tilelang_stats or {},
            "tilelang_ok":           tilelang_ok,
            "tilelang_err":          tilelang_err,
            "tilelang_autotune_cfg": tilelang_cfg,
            "nki_ms":                nki_ms,
            "nki_stats":             nki_stats or {},
            "nki_ok":                nki_ok,
            "nki_err":               nki_err,
            "nki_autotune_cfg":      nki_cfg,
            "speedup_triton":        torch_ms / triton_ms if triton_ms > 0 else 0.0,
            "speedup_cutile":        torch_ms / cutile_ms if cutile_ms > 0 else 0.0,
            "speedup_tilelang":      torch_ms / tilelang_ms if tilelang_ms > 0 else 0.0,
            "speedup_nki":           torch_ms / nki_ms if nki_ms > 0 else 0.0,
        })

    return results
