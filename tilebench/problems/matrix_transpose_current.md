# Matrix Transpose

Implement a program that returns the transpose of a 2D array. The output has rows and columns swapped relative to the input.

The input consists of:

- `x`: A 2D array of shape $(m, n)$.

The output should be written to the `output` array of shape $(n, m)$, with the same dtype as `x`. The output must be contiguous in memory.

The operation is defined mathematically as:

$$
output[j, i] = x[i, j]
$$

for $i \in [0, m)$, $j \in [0, n)$.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- The output must be a contiguous tensor (not just a stride-only view)

## Example 1

```text
Input: x = [[1, 2, 3], [4, 5, 6]]   (shape (2, 3))
Output: [[1, 4], [2, 5], [3, 6]]    (shape (3, 2))
```

## Constraints

- $m \geq 1$, $n \geq 1$
- `output` must be a fresh contiguous tensor with the transposed shape
