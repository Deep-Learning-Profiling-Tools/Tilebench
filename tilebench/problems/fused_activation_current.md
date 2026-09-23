# Fused Element-wise Activation (SiLU of MAD)

Implement a fused element-wise activation: given three 1D arrays `x`, `gate`, and `bias` of the same length, compute the SiLU activation of `x * gate + bias`. SiLU is defined as $\text{SiLU}(z) = z \cdot \sigma(z) = z / (1 + e^{-z})$.

The input consists of three arrays:

- `x`: A 1D array.
- `gate`: A 1D array of the same length as `x`.
- `bias`: A 1D array of the same length as `x` (element-wise, not broadcast).

The output should be written to the `output` array, which has the same length as the inputs.

The operation is defined mathematically as:

$$
z[i] = x[i] \cdot gate[i] + bias[i]
$$

$$
output[i] = \text{SiLU}(z[i]) = \frac{z[i]}{1 + e^{-z[i]}}
$$

where $i$ ranges from $0$ to $n-1$.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- The three inputs are independent vectors of length $n$ — `bias` is not broadcast

## Example 1

```text
Input: x = [1.0, 2.0], gate = [3.0, 0.5], bias = [0.0, -1.0]
z = [3.0, 0.0]; SiLU(3.0) ≈ 2.857; SiLU(0.0) = 0.0
Output: [2.857..., 0.0]
```

## Constraints

- $1 \leq n$
- `x`, `gate`, `bias` all have length $n$
