# SwiGLU Activation

Implement a program that applies the SwiGLU activation element-wise. Given two 2D arrays `x` and `y` of identical shape, produce the SiLU (Swish) of `x` multiplied element-wise by `y`. SiLU is defined as $\text{SiLU}(z) = z \cdot \sigma(z) = z / (1 + e^{-z})$.

The input consists of:

- `x`: A 2D array of shape $(M, N)$.
- `y`: A 2D array of shape $(M, N)$ (same as `x`).

The output should be written to the `output` array of shape $(M, N)$.

The operation is defined mathematically as:

$$
output[m, n] = \text{SiLU}(x[m, n]) \cdot y[m, n] = \frac{x[m, n]}{1 + e^{-x[m, n]}} \cdot y[m, n]
$$

where $m \in [0, M)$ and $n \in [0, N)$.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`

## Example 1

```text
Input: x = [[0.0, 1.0]], y = [[2.0, 3.0]]
SiLU(0) = 0; SiLU(1) ≈ 0.731
Output: [[0.0, 2.193...]]
```

## Example 2

```text
Input: x = [[-1.0]], y = [[4.0]]
SiLU(-1) ≈ -0.269
Output: [[-1.076...]]
```

## Constraints

- $1 \leq M$, $1 \leq N$
- `x.shape == y.shape`
