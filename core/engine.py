import importlib

import torch
import yaml

from core.timer import report_benchmark
from core.verifier import verify
from data.tensors import get_generator


def run_benchmark_suite(operator_name):
    config_path = f"benchmarks/operators/{operator_name}/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Dynamically import implementations
    impl_torch = importlib.import_module(
        f"benchmarks.operators.{operator_name}.impl_torch"
    )
    impl_triton = importlib.import_module(
        f"benchmarks.operators.{operator_name}.impl_triton"
    )
    try:
        impl_cutile = importlib.import_module(
            f"benchmarks.operators.{operator_name}.impl_cutile"
        )
    except ImportError as e:
        print(f"  cuTile import skipped: {e}")
        impl_cutile = None

    # Get the registered input generator for this operator
    generate_inputs = get_generator(operator_name)

    results = []

    for case in config["test_cases"]:
        # Extract all parameters from the case except 'dtype'
        params = {k: v for k, v in case.items() if k != "dtype" and k != "block_size"}
        dtype_str = case.get("dtype", "float32")
        dtype = getattr(torch, dtype_str)
        block_size = case.get("block_size", 1024)

        print(f"Running case: {params}, dtype={dtype_str}, block_size={block_size}")

        # 1. Data preparation using the generic generator
        inputs = generate_inputs(**params, dtype=dtype)

        # 2. Reference run (Torch)
        ref_output = impl_torch.run(*inputs)
        torch.cuda.synchronize()
        torch_ms = report_benchmark(impl_torch.run, inputs)["mean_time_ms"]

        # 3. Triton run
        triton_output = impl_triton.run(*inputs, block_size=block_size)
        torch.cuda.synchronize()
        triton_ok, triton_err = verify(triton_output, ref_output)
        if not triton_ok:
            print(f"  Triton verification FAILED: {triton_err}")
        triton_ms = (
            report_benchmark(
                impl_triton.run, inputs, kwargs={"block_size": block_size}
            )["mean_time_ms"]
            if triton_ok
            else float("nan")
        )

        # 4. cuTile run
        try:
            cutile_output = impl_cutile.run(*inputs, block_size=block_size)
            torch.cuda.synchronize()
            cutile_ok, cutile_err = verify(cutile_output, ref_output)
            if not cutile_ok:
                print(f"  cuTile verification FAILED: {cutile_err}")
            cutile_ms = (
                report_benchmark(
                    impl_cutile.run, inputs, kwargs={"block_size": block_size}
                )["mean_time_ms"]
                if cutile_ok
                else float("nan")
            )
        except Exception as e:
            cutile_ok = False
            cutile_err = str(e)
            print(f"  cuTile execution FAILED: {cutile_err}")
            cutile_ms = float("nan")

        res = {
            "params": params,
            "dtype": dtype_str,
            "torch_ms": torch_ms,
            "triton_ms": triton_ms,
            "triton_ok": triton_ok,
            "triton_err": triton_err,
            "cutile_ms": cutile_ms,
            "cutile_ok": cutile_ok,
            "cutile_err": cutile_err,
            "speedup_triton": torch_ms / triton_ms if triton_ms > 0 else 0,
            "speedup_cutile": torch_ms / cutile_ms if cutile_ms > 0 else 0,
        }
        results.append(res)

    return results
