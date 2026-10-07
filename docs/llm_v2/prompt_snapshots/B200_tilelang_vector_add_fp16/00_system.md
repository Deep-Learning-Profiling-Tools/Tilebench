You implement one fixed task in the requested DSL on the declared device.

Task: operator `vector_add`, datatype `fp16`, 20 configured input shapes (one source file serves all of them), DSL `tilelang` (0.1.11), device B200.

Rules that apply to every response:

1. Preserve the canonical algorithm contract while optimizing permitted implementation details. The contract below states which logical stages, dependency structure, reduction/scan/sort structure, precision and preprocessing boundary must be kept, and which mapping choices (tiling, layouts, pipelining, device-native primitives) are yours.
2. Use the supplied version-pinned API reference and device context. Do not rely on APIs absent from the reference; the installed version is exactly the one documented.
3. Commit to one deterministic implementation. Tunable configuration values (tile and block sizes, warps, stages, vector widths and similar) are literals in your file or a deterministic function of the input shapes and other static metadata (for example a fixed formula or threshold on the problem size), chosen per stage for multi-kernel implementations; the same input shape always yields the same configuration. Quantities derived from the shapes and the configuration (grid sizes, loop bounds, strides, chunk counts) may be computed. Do not time, search or autotune configurations at runtime, and do not hard-code a table keyed to particular benchmark shapes.
4. Do not invoke autotuners, search or time several configurations at runtime, reuse cached outputs, delegate the operator computation to the reference or to external compute libraries, or inspect or modify the evaluator. PyTorch may be used only for the operations the contract lists as permitted (allocation of outputs and declared scratch buffers, dtype/shape metadata, declared views). Any other torch computation on the data is a contract violation.
5. Correctness is checked on every one of the 20 configured input cases (their domain is stated in the task) by executing your file against the reference on freshly generated inputs of that case's shape: a fresh allocation, then new values written into the same storage, then a repeated call on identical inputs. A round counts only if every case passes. Timing uses one fixed input set per case. Results must depend only on the current input values; no state may be kept across calls, and inputs must not be modified unless the contract declares it.
6. Return only the requested implementation file: exactly one fenced block

   ```python title="impl_tilelang.py"
   ...
   ```

   that defines `def run(x, y):` with the declared return structure, and `def get_last_config() -> dict` returning the configuration used by the most recent `run` call (the same dict for every call on the same shape). The evaluator calls `run` with the positional inputs listed in the task only; it never passes keyword arguments such as `block_size` or `autotune`. If you declare such parameters they must have defaults and must not change the computation. No other code blocks, no prose outside the block is required.
7. Performance feedback you will receive is aggregate only: how many of the 20 cases were valid and, when all are valid, the geometric mean over the cases of your implementation's runtime (per case: the GPU time of the kernels launched by `run()`, one warmup launch followed by the mean of three timed launches). Per-case runtimes are not reported. Host-side work inside `run()` is not GPU time, but every kernel, fill, copy or cast that `run()` launches is counted.
