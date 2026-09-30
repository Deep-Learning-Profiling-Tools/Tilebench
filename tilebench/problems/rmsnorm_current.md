# RMS Normalization (Forward)

Implement a program that performs Root-Mean-Square normalization over the last dimension. Unlike LayerNorm, RMSNorm does NOT subtract the mean — it only rescales by the root-mean-square magnitude, then applies an element-wise per-channel weight.

The input consists of:

- `x`: A 3D array of shape $(batch, M, K)$.
- `rms_w`: A 1D array of shape $(K,)$ (per-channel scale).
- `eps`: A small positive scalar for numerical stability.

The output should be written to the `output` array of shape $(batch, M, K)$, with the same dtype as `x`. Internal arithmetic must be done in fp32.

For each $(b, m)$ row of length $K$:

$$
\text{rms}[b, m] = \sqrt{\frac{1}{K} \sum_{k=0}^{K-1} x[b, m, k]^2 + \epsilon}
$$

$$
output[b, m, k] = \frac{x[b, m, k]}{\text{rms}[b, m]} \cdot rms\_w[k]
$$

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- Compute sum-of-squares and reciprocal-rms in fp32

## Example 1

```text
Input: x = [[[1.0, 2.0, 3.0]]], rms_w = [1.0, 1.0, 1.0], eps = 1e-6
sum_sq / K = (1+4+9)/3 = 4.667; rms ≈ 2.160
Output: [[[0.463, 0.926, 1.389]]]
```

## Constraints

- $batch \geq 1$, $M \geq 1$, $K \geq 1$
- $\epsilon > 0$
- `rms_w` has length $K$
