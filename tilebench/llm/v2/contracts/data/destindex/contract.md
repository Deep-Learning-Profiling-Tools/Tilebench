# destindex: canonical algorithm contract

## Functional semantics
KV-cache "copy to destination index" with a nope part and a rope part. Given
`kv_nope` of shape (T, Hn, Dn), `kv_rope` of shape (T, Hr, Dr), an int64
index vector `dest_loc` of shape (T,), and initial cache contents `o_nope`
(T, Hn, Dn) and `o_rope` (T, Hr, Dr), the two results are

    out_nope = copy of o_nope;  out_nope[dest_loc[t], :, :] = kv_nope[t, :, :]  for all t
    out_rope = copy of o_rope;  out_rope[dest_loc[t], :, :] = kv_rope[t, :, :]  for all t

i.e. `index_copy_` along dimension 0 into a copy of the initial contents.
Rows of the output not named by `dest_loc` keep the corresponding `o_*`
values. In every benchmark case `dest_loc` is a permutation of [0, T), so
destinations are unique (no write ever collides) and every row is
overwritten; duplicate or out-of-range destinations need not be handled.
Values are copied bit-exactly; no arithmetic is performed.

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
  call. Whether the two output buffers may persist across calls (allocated
  and initialised from `o_*` once, then rewritten in place on each later
  call) or must be freshly produced on every call is an open review item
  (see below); until it is resolved either policy is accepted, and the
  evaluator will flag the persistent policy for review.
- Call form: `run(kv_nope, kv_rope, dest_loc, o_nope, o_rope, autotune=False)`.
  `autotune` is a framework knob; ignore it and never run a configuration
  search.

## Required logical stages
1. Output initialisation: the output buffers hold the contents of `o_nope` /
   `o_rope` before the copy (a device copy of each `o_*` into the output
   buffer).
2. Scatter-copy of the nope part: for every (t, h, d), read `kv_nope[t, h, d]`
   and write it to `out_nope[dest_loc[t], h, d]`.
3. Scatter-copy of the rope part: likewise for `kv_rope` into `out_rope`.

Stages 2 and 3 depend on stage 1 (for the rows they do not overwrite) and are
independent of each other; they may run as two launches of one kernel, or be
fused into a single launch. Stage 1 may be a plain device copy; it must not
be fused into the scatter in a way that re-reads `o_*` per element.

## Algorithm family and structure
Index-driven scatter copy (row permutation along dimension 0): a streaming
copy whose destination address is computed from an index looked up per
source row. No reduction, scan, sort, atomics or conflict resolution. The
index may be read once per token or once per element; either way each
source element is read exactly once and written exactly once.

## Precision and accumulation
Exact copy for every dtype, including int8; no conversion of values. Index
arithmetic in int32 or int64.

## Preprocessing and timing boundary
run() performs only: shape/stride reads, metadata-only flattening (`.view`)
of contiguous tensors, the output initialisation of stage 1 (when the policy
requires it on this call), and the launches. No casts, transposes or packing
of inputs; no precomputation outside run(); no derived operand cached across
calls other than the output buffers covered by the open review item.

## Permitted implementation mappings
- Flat element addressing with offset decomposition into (token, head, d)
  versus a (token, head)-row tiling; elements per program; launch geometry.
- One kernel launched per part versus one fused launch handling both parts.
- Reading `dest_loc` per element or per token; int32 or int64 index
  arithmetic.
- Masked tail handling versus zero-padded loads with dropped destinations.

## Forbidden substitutions
- Host-side `index_copy_`, `index_put_`, `scatter_`, advanced indexing
  (`out[dest_loc] = kv`) or `torch.index_select` for the copy itself.
- Writing the results into `o_nope` / `o_rope` (input mutation).
- Any sort or inverse-permutation preprocessing of `dest_loc`.
- Re-reading `o_*` inside the scatter, or any additional full pass over the
  data beyond stage 1 and the single scatter-copy.
- Autotuning, timing, benchmarking or configuration search inside the
  generated file.

## Permitted PyTorch operations
- `torch.empty_like` / `torch.empty` for output buffers and
  `Tensor.clone()` or `Tensor.copy_()` for stage 1 (copy of `o_*` into the
  output buffer).
- `Tensor.view(-1)` / `.reshape(-1)` on contiguous tensors (metadata only);
  `Tensor.contiguous()` only as a no-op.
- Tensor metadata: `.shape`, `.numel()`, `.dtype`, `.device`, `.is_contiguous()`.
- `torch.cuda.current_stream()` to obtain the launch stream.
Everything else is forbidden.

## Open review items
- Output-buffer policy: must outputs be freshly initialised from `o_*` on
  every call, or may they persist across calls (initialised once per input
  identity and rewritten in place)? This fixes what the timed window
  contains.
- Whether the byte-count formula should include the output initialisation
  when per-call initialisation is required.
- Whether rows not addressed by `dest_loc` (never produced by the current
  benchmark inputs) are to be verified explicitly.
