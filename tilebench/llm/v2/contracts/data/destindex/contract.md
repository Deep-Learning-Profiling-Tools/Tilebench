# destindex: canonical algorithm contract

## Functional semantics
KV-cache "copy to destination index" with a nope part and a rope part. Given
`kv_nope` of shape (T, Hn, Dn), `kv_rope` of shape (T, Hr, Dr), an int64
index vector `dest_loc` of shape (T,), and cache buffers `o_nope`
(T, Hn, Dn) and `o_rope` (T, Hr, Dr), the two results are

    out_nope[dest_loc[t], :, :] = kv_nope[t, :, :]  for all t
    out_rope[dest_loc[t], :, :] = kv_rope[t, :, :]  for all t

i.e. the reference `index_copy_` along dimension 0 into a copy of `o_*`.
The task domain is fixed: `dest_loc` is a permutation of [0, T). Hence
destinations are unique (no write ever collides) and every output row is
overwritten by the scatter, so the results are fully determined by `kv_*`
and `dest_loc`; no output row retains an `o_*` value. Duplicate or
out-of-range destinations are outside the task domain and need not be
handled. Values are copied bit-exactly; no arithmetic is performed.

## Inputs and outputs
- `kv_nope`, `kv_rope`, `o_nope`, `o_rope`: contiguous, same dtype
  (fp16, bf16, fp32 or int8); T = batch_size * seq_len.
- `dest_loc`: (T,), int64, contiguous. Indices may be narrowed to int32 for
  addressing (T fits in int32).
- Output: a tuple `(out_nope, out_rope)`, in this order, with the shapes and
  dtypes of `o_nope` and `o_rope`.
- `kv_nope`, `kv_rope`, `dest_loc`, `o_nope` and `o_rope` must never be
  modified; in particular the results must not be written into `o_nope` /
  `o_rope` themselves, and the outputs must not alias any input.
- Each call must return results that are correct for the arguments of that
  call, and no result value may be carried over from an earlier call: no
  cached outputs, no reuse of previously computed rows, no skipping of the
  scatter when the arguments look unchanged. Reusing an output allocation
  across calls is allowed only if the current call fully overwrites every
  element of both results before returning (which the scatter does in this
  task domain); a fresh allocation on every call is equally acceptable.
- Call form: `run(kv_nope, kv_rope, dest_loc, o_nope, o_rope)`, all positional;
  no keyword arguments are passed. Never run a configuration search.

## Required logical stages
1. Scatter-copy of the nope part: for every (t, h, d), read `kv_nope[t, h, d]`
   and write it to `out_nope[dest_loc[t], h, d]`.
2. Scatter-copy of the rope part: likewise for `kv_rope` into `out_rope`.

The two stages are independent of each other; they may run as two launches
of one kernel, or be fused into a single launch. Initialising the output
buffers from `o_nope` / `o_rope` (a device copy of each `o_*` into the
output buffer before the scatter) is not a required stage in this task
domain, because every row is overwritten by the scatter; it remains a
permitted mapping, and when performed it is device work inside `run()`.
The scatter itself must not read `o_*` per element.

## Algorithm family and structure
Index-driven scatter copy (row permutation along dimension 0): a streaming
copy whose destination address is computed from an index looked up per
source row. No reduction, scan, sort, atomics or conflict resolution. The
index may be read once per token or once per element; either way each
source element is read once and written once in the algorithm (logical
accesses, not a guarantee about physical DRAM transactions).

## Precision and accumulation
Exact copy for every dtype, including int8; no conversion of values. Index
arithmetic in int32 or int64.

## Preprocessing and timing boundary

The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the measured number.

Besides host work (shape/stride reads, metadata-only flattening (`.view`) of
contiguous tensors, allocation calls), run() issues only the scatter launches
and, if the implementation chooses to initialise its output buffers from
`o_*`, that device copy, which is counted. No casts, transposes or packing of
inputs; no precomputation outside run(); no derived operand and no result
value cached across calls (an output allocation may be kept only under the
full-overwrite condition stated above).

## Permitted implementation mappings
- Flat element addressing with offset decomposition into (token, head, d)
  versus a (token, head)-row tiling; elements per program; launch geometry.
- One kernel launched per part versus one fused launch handling both parts.
- Reading `dest_loc` per element or per token; int32 or int64 index
  arithmetic.
- Masked tail handling versus zero-padded loads with dropped destinations.
- Fresh output allocations on every call versus an allocation reused across
  calls that the call fully overwrites; output buffers left uninitialised
  before the scatter versus initialised from `o_*` by a device copy.

## Forbidden substitutions
- Host-side `index_copy_`, `index_put_`, `scatter_`, advanced indexing
  (`out[dest_loc] = kv`) or `torch.index_select` for the copy itself.
- Writing the results into `o_nope` / `o_rope` (input mutation).
- Any sort or inverse-permutation preprocessing of `dest_loc`.
- Reading `o_*` inside the scatter, or any additional logical traversal of
  the data beyond the single scatter-copy (and the optional one-time copy of
  `o_*` into the output buffers).
- Carrying result values across calls: returning cached outputs, reusing
  previously computed rows, or skipping the scatter for arguments seen
  before.
- Autotuning, timing, benchmarking or configuration search inside the
  generated file.

## Permitted PyTorch operations
- `torch.empty_like` / `torch.empty` for output buffers and, only for the
  optional initialisation of an output buffer from `o_*`, `Tensor.clone()`
  or `Tensor.copy_()`.
- `Tensor.view(-1)` / `.reshape(-1)` on contiguous tensors (metadata only);
  `Tensor.contiguous()` only as a no-op.
- Tensor metadata: `.shape`, `.numel()`, `.dtype`, `.device`, `.is_contiguous()`.
- `torch.cuda.current_stream()` to obtain the launch stream.
Everything else is forbidden.
