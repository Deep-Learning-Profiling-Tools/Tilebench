# Mean Reduction (Row-wise)

Implement a program that, given a 2D array, computes the arithmetic mean of each row. The reduction must be done in fp32 for numerical consistency, regardless of the input dtype.

The input consists of:

- `x`: A 2D array of shape $(M, N)$.

The output should be written to the `output` array of shape $(M,)$, in fp32.

The operation is defined mathematically as:

$$
output[m] = \frac{1}{N} \sum_{n=0}^{N-1} x[m, n]
$$

for $m \in [0, M)$. The summation is performed in fp32 even if the input dtype is fp16 or bf16.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output` as fp32

## Example 1

```text
Input: x = [[1.0, 2.0, 3.0, 4.0], [10.0, -5.0, 0.0, 0.0]]
Output: [2.5, 1.25]
```

## Constraints

- $1 \leq M$, $1 \leq N$
- Output dtype is fp32
