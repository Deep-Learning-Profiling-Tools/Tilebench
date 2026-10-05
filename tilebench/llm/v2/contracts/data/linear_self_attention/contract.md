# linear_self_attention: canonical algorithm contract

## Functional semantics
Non-causal, single-head linear attention with the feature map
phi(x) = x + 1 if x > 0 else exp(x), applied elementwise. For Q, K, V of
shape (M, D) in fp32 and a scalar eps:

    PHI_Q = phi(Q),  PHI_K = phi(K)                         (M, D)
    S     = PHI_K^T @ V                                     (D, D)
    Z     = sum over rows m of PHI_K[m, :]                  (D,)
    O[m, :] = (PHI_Q[m, :] @ S) / (PHI_Q[m, :] . Z + eps)   (M, D)

There is no 1/sqrt(D) scaling, no mask and no causal structure. eps is
added exactly once to the completed per-row dot product, before the
division.

## Inputs and outputs
- Q, K, V: shape (M, D), fp32, contiguous row-major. Read-only.
- eps: Python float, fourth positional argument (default 1e-6).
- O: a freshly allocated (M, D) fp32 tensor allocated inside the entry
  point. No aliasing.
- The entry point takes (Q, K, V, eps) positionally plus `block_size`,
  `autotune` and `**kwargs`; extra keywords must be accepted.

## Required logical stages
1. Feature map on Q: PHI_Q = phi(Q).
2. Feature map on K: PHI_K = phi(K).
3. State: S = PHI_K^T @ V, a (D, D) matrix reduced over the sequence axis M.
4. Normaliser: Z = column sums of PHI_K over M, a (D,) vector.
5. Output: numerator PHI_Q @ S (reduction over D); denominator
   PHI_Q . Z + eps per row; O = numerator / denominator.
Dependencies: 3 and 4 depend on 2; 5 depends on 1, 3 and 4; 1 is
independent of 2-4. Permitted fusions: stage 2 into the operand loads of 3
and/or 4 (phi applied on the fly, PHI_K never materialised); stage 4 into
stage 3 (Z accumulated in the pass that forms S); stage 1 into stage 5 (phi
applied to Q tiles as they are loaded); the denominator inside the
numerator's reduction loop or computed once per row block. Permitted
splits: each stage as its own launch with global scratch for PHI_Q, PHI_K,
S and Z allocated inside the entry point. S and Z must be completely
reduced over all M rows before stage 5 consumes them (no streaming or
online approximation of the state).

## Algorithm family and structure
Materialised- or fused-feature-map linear attention: two dense reductions
(over M for S, over D for the output) plus a column sum. Within each
reduction the accumulation order is free; a split over M for S with a
deterministic final combine is permitted. Rows or columns beyond M or D
must contribute 0 to S, Z and the output reductions. Note that phi(0) = 1,
so zero-filled padding must be masked before the feature map or clipped
so that it never reaches a reduction.

## Precision and accumulation
- phi is evaluated in fp32 (exp at fp32 precision).
- S and the numerator: operands may be rounded to TF32 (10-bit mantissa)
  for the multiply; accumulation is fp32. Operand formats narrower than
  TF32 (fp16, bf16, fp8) are forbidden.
- Z and the denominator PHI_Q . Z: fp32 operands without TF32 rounding,
  fp32 accumulation.
- The division is fp32; O is fp32.

## Preprocessing and timing boundary
Everything inside the entry point is timed: allocation of O and of any
scratch, any layout transform (for example a transposed copy of an operand
or of an intermediate) and every launch. No cross-call cache of PHI_K, S,
Z, a transposed V or any other derived buffer; no prepacked or
pre-transposed inputs are provided or may be assumed.

## Permitted implementation mappings
Logical tile shapes and launch parameters of the two GEMM-like stages;
materialised versus fused feature maps; separate versus fused normaliser;
transposed loads, in-register transposes or in-run transposed copies for
operands that are consumed along the reduction axis; runtime versus
compile-time loop bounds; grid orders; eps as a runtime or compile-time
scalar; the number of launches (anywhere from two to eight).

## Forbidden substitutions
torch.matmul / torch.mm / torch.bmm / torch.einsum / the @ operator /
torch.nn.functional.linear for S, the numerator or the denominator;
torch.exp / torch.where for phi; torch.sum / Tensor.sum for Z;
torch.nn.functional.scaled_dot_product_attention; any PyTorch arithmetic on
the data path; caching any intermediate across calls.

## Permitted PyTorch operations
- torch.empty / torch.empty_like for O and for declared per-call scratch
  (PHI_Q, PHI_K, S, Z and, if used, transposed copies of operands or
  intermediates).
- Tensor.contiguous() / Tensor.t() / Tensor.transpose() only to produce such
  per-call, in-run scratch (timed).
- Reads of shape / stride / dtype / device metadata.
- torch.cuda.current_stream() to obtain the launch stream.
Everything else is forbidden.
