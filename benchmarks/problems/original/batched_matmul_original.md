# Batched Matrix Multiplication

Implement a batched matrix multiplication in FP32. Given a batch of matrices `A` of shape `[B, M, K]` and a batch of matrices `B` of shape `[B, K, N]`, compute the output batch `C` of shape `[B, M, N]` such that for each batch index `b`:

$$
C_b = A_b \times B_b
$$

All matrices are stored in row-major order and use 32-bit floating point numbers (FP32).

## Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in the `C` array

## Example 1

```text
Input:
B = 2, M = 2, K = 3, N = 2
A = [
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
  [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
]
B = [
  [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
  [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]]
]
Output:
C = [
  [[22.0, 28.0], [49.0, 64.0]],
  [[92.0, 68.0], [128.0, 95.0]]
]
```

## Constraints

- $1 \leq B \leq 128$
- $1 \leq M, N, K \leq 1024$
- Performance is measured with $K = 256$, $M = 256$, $N = 256$
