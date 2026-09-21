# Layer Normalization (Forward)

Implement a program that performs Layer Normalization over the last dimension of the input. For each row, compute its mean and variance, normalize, then apply an element-wise affine transform with `weight` and `bias`.

The input consists of:

- `x`: A 3D array of shape $(batch, M, K)$.
- `weight`: A 1D array of shape $(K,)$.
- `bias`: A 1D array of shape $(K,)$.
- `eps`: A small positive scalar for numerical stability.

The output should be written to the `output` array of shape $(batch, M, K)$, with the same dtype as `x`. Statistics and intermediate arithmetic must be done in fp32.

For each $(b, m)$ row of length $K$:

$$
\mu[b, m] = \frac{1}{K} \sum_{k=0}^{K-1} x[b, m, k]
$$

$$
\sigma^2[b, m] = \frac{1}{K} \sum_{k=0}^{K-1} \big( x[b, m, k] - \mu[b, m] \big)^2
$$

$$
output[b, m, k] = \frac{x[b, m, k] - \mu[b, m]}{\sqrt{\sigma^2[b, m] + \epsilon}} \cdot weight[k] + bias[k]
$$

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- Compute mean / variance / reciprocal-stddev in fp32

## Example 1

```text
Input: x = [[[1.0, 2.0, 3.0]]], weight = [1.0, 1.0, 1.0], bias = [0.0, 0.0, 0.0], eps = 1e-5
mean = 2.0; var = 2/3 ≈ 0.6667; rstd ≈ 1.2247
Output: [[[-1.2247, 0.0, 1.2247]]]
```

## Constraints

- $batch \geq 1$, $M \geq 1$, $K \geq 1$
- $\epsilon > 0$
- `weight` and `bias` have length $K$
