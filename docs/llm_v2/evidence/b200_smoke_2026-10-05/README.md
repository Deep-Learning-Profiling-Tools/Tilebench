# B200 functional smoke of the v2 evaluator (2026-10-05, dgx003)

Purpose: verify the isolated worker end to end on a real GPU — import/compile,
the three fresh-input numerical checks, 1 warmup + 3 timed launches with one
Proton scope per launch, CUDA-graph capture, 253 MB eviction. This is NOT a
performance result and NOT a study artifact: the candidates are hand-written
for the smoke and the problem size (n = 2^22, fp16) is not the task's
representative case.

Host: NVIDIA B200, driver 595.58.03, torch 2.10.0+cu130, triton 3.6.0,
`tilebench_env`. GPU 0 was idle (0 % utilisation, no compute processes) when
the smoke ran. Sandbox and compile caches were under the session scratchpad;
only the worker's JSON results are kept here.

| file | content |
|---|---|
| `candidate_ok.py` | minimal Triton vector add with fixed BLOCK=1024 (no autotune, no cache) |
| `result_ok.json` | status `valid`; samples 0.006784 / 0.006528 / 0.006496 ms, mean 0.006603 ms; `capture_succeeded: true`, `timing_execution_mode: graph`, `graph_prep_runs: 4`, `flush_buffer_mb: 253`; checks fresh_storage / same_address_new_values / repeat_same_inputs all ok |
| `candidate_cheat.py` | same kernel, but `run()` caches its output by `(x.data_ptr(), y.data_ptr())` |
| `result_cheat.json` | static verdict `review_required` (module-level mutable container, tensor address used as a key); worker status `numerical_error` at `same_address_new_values` (99.9 % of elements mismatch: the cached output was returned for refilled inputs) |

The cheat case is the execution-based evidence the protocol requires beyond
static regex: static analysis alone only yields `review_required`; the
evaluator's same-address refill check is what proves the output cache.
