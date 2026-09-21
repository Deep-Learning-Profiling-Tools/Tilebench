# Flash Decode (Stage 2 — Block Reduction)

Implement Stage 2 of the Flash Decoding algorithm: a numerically-stable reduction across pre-computed partial attention outputs from Stage 1. Given partial outputs and their log-sum-exp values for each of `num_blocks` blocks per (batch, head), produce the final attention output by online-merge with global max-shifted exponentials.

The input consists of:

- `mid_o`: A 4D array of shape $(batch, heads, num\_blocks, head\_dim)$. Each `mid_o[b, h, k, :]` is the partial weighted-value output from Stage 1 block $k$.
- `mid_o_lse`: A 3D array of shape $(batch, heads, num\_blocks)$. `mid_o_lse[b, h, k]` is the log-sum-exp of block $k$'s pre-softmax scores.
- `b_seqlen`: A 1D int32 array of shape $(batch,)$ holding the actual valid sequence length per batch (used to ignore tail blocks beyond `ceil(b_seqlen / block_seq)`).
- `block_seq`: A scalar int specifying the block size used in Stage 1 partitioning.

The output should be written to the `output` array of shape $(batch, heads, head\_dim)$, with the same dtype as `mid_o`.

For each $(b, h)$ pair, let $K_{valid}[b] = \lceil b\_seqlen[b] / block\_seq \rceil$. Compute:

$$
m^*[b, h] = \max_{k \in [0, K_{valid}[b])} mid\_o\_lse[b, h, k]
$$

$$
w[b, h, k] = e^{\, mid\_o\_lse[b, h, k] - m^*[b, h]} \quad \text{for valid } k; \; 0 \text{ otherwise}
$$

$$
output[b, h, :] = \frac{\sum_{k=0}^{K_{valid}[b]-1} w[b, h, k] \cdot mid\_o[b, h, k, :]}{\sum_{k=0}^{K_{valid}[b]-1} w[b, h, k]}
$$

Blocks with $k \geq K_{valid}[b]$ are ignored (set their weight to $0$ / their LSE to $-\infty$).

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- Use online merge (running max + running denominator) for numerical stability
- Honour per-batch valid block count from `b_seqlen` / `block_seq`

## Example 1

Conceptual single-batch, single-head, head_dim=2, num_blocks=2, both valid:

```text
mid_o = [[[[1.0, 2.0], [10.0, 20.0]]]] (shape (1,1,2,2))
mid_o_lse = [[[0.0, 1.0]]]
m* = 1.0; w = [exp(-1), exp(0)] = [0.368, 1.0]; w_sum = 1.368
numer = 0.368*[1,2] + 1.0*[10,20] = [10.368, 20.736]
output = [10.368/1.368, 20.736/1.368] ≈ [7.580, 15.160]
```

## Constraints

- $batch \geq 1$, $heads \geq 1$, $num\_blocks \geq 1$, $head\_dim \geq 1$
- `block_seq` may be passed as an int or a 0-d tensor
- $b\_seqlen[b] \leq num\_blocks \cdot block\_seq$
