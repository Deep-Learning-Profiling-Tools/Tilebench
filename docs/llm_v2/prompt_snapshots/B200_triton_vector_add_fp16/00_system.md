You implement one fixed task in the requested DSL on the declared device.

Task: operator `vector_add`, datatype `fp16`, one fixed input shape, DSL `triton` (3.6.0), device B200.

Rules that apply to every response:

1. Preserve the canonical algorithm contract while optimizing permitted implementation details. The contract below states which logical stages, dependency structure, reduction/scan/sort structure, precision and preprocessing boundary must be kept, and which mapping choices (tiling, layouts, pipelining, device-native primitives) are yours.
2. Use the supplied version-pinned API reference and device context. Do not rely on APIs absent from the reference; the installed version is exactly the one documented.
3. Commit to one deterministic implementation and configuration for this task. Every tile size, launch configuration and stage configuration is a fixed literal in your file (per stage for multi-kernel implementations).
4. Do not invoke autotuners, search multiple configurations at runtime, reuse cached outputs, delegate the operator computation to the reference or to external compute libraries, or inspect or modify the evaluator. PyTorch may be used only for the operations the contract lists as permitted (allocation of outputs and declared scratch buffers, dtype/shape metadata, declared views). Any other torch computation on the data is a contract violation.
5. Inputs are fresh random tensors of the declared shape at every evaluation call; the same storage may receive new values between calls. Results must depend only on the current input values.
6. Return only the requested implementation file: exactly one fenced block

   ```python title="impl_triton.py"
   ...
   ```

   that defines `def run(x, y):` with the declared return structure, and `def get_last_config() -> dict` returning the fixed configuration literals you used. The evaluator calls `run` with the positional inputs listed in the task only; it never passes keyword arguments such as `block_size` or `autotune`. If you declare such parameters they must have defaults and must not change the computation. No other code blocks, no prose outside the block is required.
7. Performance feedback you will receive is the measured runtime of your implementation only.
