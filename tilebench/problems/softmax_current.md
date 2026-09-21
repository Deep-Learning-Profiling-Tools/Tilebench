# Softmax (Row-wise, Last Dimension)

Implement a program that applies the softmax function row-wise over the last dimension of a 2D array. Use the standard numerically-stable formulation (subtract row max before exp).

The input consists of:

- `x`: A 2D array of shape $(n\_rows, n\_cols)$.

The output should be written to the `output` array of shape $(n\_rows, n\_cols)$, with the same dtype as `x`.

For each row $r \in [0, n\_rows)$ of length $n\_cols$:

$$
m_r = \max_{c \in [0, n\_cols)} x[r, c]
$$

$$
output[r, c] = \frac{e^{\,x[r, c] - m_r}}{\displaystyle \sum_{c'=0}^{n\_cols-1} e^{\,x[r, c'] - m_r}}
$$

The subtraction of `m_r` is for numerical stability and does not change the mathematical result.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- Use the max-subtraction trick for stability; accumulate in fp32

## Example 1

```text
Input: x = [[1.0, 2.0, 3.0]]
After subtracting max=3: exp([-2, -1, 0]) = [0.135, 0.368, 1.0]; sum = 1.503
Output: [[0.090, 0.245, 0.665]]
```

## Constraints

- $n\_rows \geq 1$, $n\_cols \geq 1$
- Each row of the output sums to 1
