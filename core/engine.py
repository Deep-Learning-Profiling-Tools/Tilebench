import importlib
import inspect

import torch
import yaml
from core.dtypes import resolve_dtype
from core.timer import report_benchmark
from core.verifier import verify
from data.tensors import expand_cases, get_generator, infer_problem_size


def run_benchmark_suite(operator_name, benchmark_overrides=None):
    config_path = f"benchmarks/operators/{operator_name}/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    impl_torch = importlib.import_module(f"benchmarks.operators.{operator_name}.impl_torch")
    impl_triton = importlib.import_module(f"benchmarks.operators.{operator_name}.impl_triton")
    impl_cutile = importlib.import_module(f"benchmarks.operators.{operator_name}.impl_cutile")

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

    for case_idx, case in enumerate(cases):
        params     = {k: v for k, v in case.items() if k not in ("dtype", "block_size")}
        dtype_str  = case.get("dtype", "float32")
        dtype      = resolve_dtype(dtype_str)
        block_size = case.get("block_size", 1024)

        print(f"Running case: {params}, dtype={dtype_str}")

        # Skip the case if input generation or the torch reference run fails
        # (e.g. a dtype not supported by the operator, such as FP8 element-wise).
        try:
            inputs = generate_inputs(**params, dtype=dtype)
            ref_output = impl_torch.run(*inputs)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  Skipped: dtype={dtype_str} not supported ({type(e).__name__}: {e})")
            continue

        # Label prefix shared by all three backends for this case.
        lbl = f"{operator_name}_{dtype_str}_c{case_idx:03d}"

        # --- Torch ---
        torch_stats = _bench(impl_torch.run, inputs, label=f"{lbl}_torch")
        torch_ms    = torch_stats["mean"]

        def _run_kwargs(fn):
            sig = inspect.signature(fn).parameters
            kw = {}
            if "block_size" in sig:
                kw["block_size"] = block_size
            if "autotune" in sig:
                kw["autotune"] = autotune
            return kw

        # --- Triton ---
        triton_output = impl_triton.run(*inputs, **_run_kwargs(impl_triton.run))
        torch.cuda.synchronize()
        triton_ok, triton_err = verify(triton_output, ref_output, atol=verify_atol, rtol=verify_rtol)
        triton_cfg = getattr(impl_triton, "get_last_config", lambda: None)() if autotune else None
        if triton_cfg:
            print(f"  Triton autotune → {triton_cfg}")
        if not triton_ok:
            print(f"  Triton verification FAILED: {triton_err}")
        triton_stats = (
            _bench(impl_triton.run, inputs, _run_kwargs(impl_triton.run), label=f"{lbl}_triton")
            if triton_ok else None
        )
        triton_ms = triton_stats["mean"] if triton_stats is not None else float("nan")

        # --- cuTile ---
        try:
            cutile_output = impl_cutile.run(*inputs, **_run_kwargs(impl_cutile.run))
            torch.cuda.synchronize()
            cutile_ok, cutile_err = verify(cutile_output, ref_output, atol=verify_atol, rtol=verify_rtol)
            cutile_cfg = getattr(impl_cutile, "get_last_config", lambda: None)() if autotune else None
            if cutile_cfg:
                print(f"  cuTile  autotune → {cutile_cfg}")
            if not cutile_ok:
                print(f"  cuTile verification FAILED: {cutile_err}")
            cutile_stats = (
                _bench(impl_cutile.run, inputs, _run_kwargs(impl_cutile.run), label=f"{lbl}_cutile")
                if cutile_ok else None
            )
            cutile_ms = cutile_stats["mean"] if cutile_stats is not None else float("nan")
        except Exception as e:
            cutile_ok    = False
            cutile_err   = str(e)
            cutile_ms    = float("nan")
            cutile_stats = None
            print(f"  cuTile execution FAILED: {cutile_err}")

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
            "triton_autotune_cfg":   getattr(impl_triton, "get_last_config", lambda: None)() if autotune else None,
            "cutile_ms":             cutile_ms,
            "cutile_stats":          cutile_stats or {},
            "cutile_ok":             cutile_ok,
            "cutile_err":            cutile_err,
            "cutile_autotune_cfg":   getattr(impl_cutile, "get_last_config", lambda: None)() if autotune else None,
            "speedup_triton":        torch_ms / triton_ms if triton_ms > 0 else 0.0,
            "speedup_cutile":        torch_ms / cutile_ms if cutile_ms > 0 else 0.0,
        })

    return results
