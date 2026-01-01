import yaml
import torch
import importlib
from core.timer import report_benchmark
from core.verifier import verify
from data.tensors import generate_vector_add_inputs

def run_benchmark_suite(operator_name):
    config_path = f"benchmarks/operators/{operator_name}/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Dynamically import implementations
    impl_torch = importlib.import_module(f"benchmarks.operators.{operator_name}.impl_torch")
    impl_triton = importlib.import_module(f"benchmarks.operators.{operator_name}.impl_triton")
    impl_cutile = importlib.import_module(f"benchmarks.operators.{operator_name}.impl_cutile")
    
    results = []
    
    for case in config['test_cases']:
        n = case['n']
        dtype_str = case['dtype']
        dtype = getattr(torch, dtype_str)
        
        print(f"Running case: n={n}, dtype={dtype_str}")
        
        # 1. Data preparation
        inputs = generate_vector_add_inputs(n, dtype=dtype)
        
        # 2. Reference run (Torch)
        ref_output = impl_torch.run(*inputs)
        torch_ms = report_benchmark(impl_torch.run, inputs)['mean_time_ms']
        
        # 3. Triton run
        triton_output = impl_triton.run(*inputs)
        triton_ok, triton_err = verify(triton_output, ref_output)
        triton_ms = report_benchmark(impl_triton.run, inputs)['mean_time_ms'] if triton_ok else float('nan')
        
        # 4. cuTile run
        try:
            cutile_output = impl_cutile.run(*inputs)
            cutile_ok, cutile_err = verify(cutile_output, ref_output)
            cutile_ms = report_benchmark(impl_cutile.run, inputs)['mean_time_ms'] if cutile_ok else float('nan')
        except Exception as e:
            cutile_ok = False
            cutile_err = str(e)
            cutile_ms = float('nan')
        
        results.append({
            "n": n,
            "dtype": dtype_str,
            "torch_ms": torch_ms,
            "triton_ms": triton_ms,
            "triton_ok": triton_ok,
            "triton_err": triton_err,
            "cutile_ms": cutile_ms,
            "cutile_ok": cutile_ok,
            "cutile_err": cutile_err,
            "speedup_triton": torch_ms / triton_ms if triton_ms > 0 else 0,
            "speedup_cutile": torch_ms / cutile_ms if cutile_ms > 0 else 0
        })
        
    return results
