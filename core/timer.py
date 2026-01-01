import torch
import time

def _time_ms(f, args, warmup_rounds, iterations, rounds):
    # Warmup
    for _ in range(warmup_rounds):
        f(*args)
    
    torch.cuda.synchronize()
    
    latencies = []
    for _ in range(rounds):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            f(*args)
        end_event.record()
        
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event) / iterations)
    
    return sum(latencies) / len(latencies)

def _estimate_bench_iter(f, args):
    # Default values
    return 10, 100, 5

def report_benchmark(f, args) -> dict[str, float]:
    warmup_rounds, iterations, rounds = _estimate_bench_iter(f, args)
    mean_time_ms = _time_ms(f, args, warmup_rounds, iterations, rounds)
    return {"mean_time_ms": mean_time_ms}
