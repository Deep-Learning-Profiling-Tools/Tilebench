# L2 Normalization (Last-Dimension)

Implement a program that L2-normalizes each row of the input along its last dimension. Each row is rescaled so that its L2 norm equals 1 (modulo an `eps` floor for numerical stability).

The input consists of:

- `x`: A 3D array of shape $(batch, M, K)$.
- `eps`: A small positive scalar (typical $10^{-6}$) added inside the norm for stability.

The output should be written to the `output` array of shape $(batch, M, K)$, with the same dtype as `x`. Internal accumulation must be done in fp32.

For each $(b, m)$ row of length $K$, the operation is defined as:

$$
\text{rstd}[b, m] = \frac{1}{\sqrt{\sum_{k=0}^{K-1} x[b, m, k]^2 + \epsilon}}
$$

$$
output[b, m, k] = x[b, m, k] \cdot \text{rstd}[b, m]
$$

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- Sum-of-squares and rstd must be computed in fp32

## Example 1

```text
Input: x = [[[3.0, 4.0]]], eps = 1e-6
sum_sq = 9 + 16 = 25; rstd ≈ 1/sqrt(25) = 0.2
Output: [[[0.6, 0.8]]]
```

## Constraints

- $batch \geq 1$, $M \geq 1$, $K \geq 1$
- $\epsilon > 0$
