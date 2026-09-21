# Stream-K Matrix Multiplication

Implement a program that performs standard matrix multiplication $C = A \cdot B$ using the Stream-K scheduling strategy (Osama et al., PPoPP 2023) to maximize SM utilization on irregular tile counts. The accumulator is fp32, the result is cast back to the input dtype.

The input consists of:

- `a`: A 2D array of shape $(m, k)$.
- `b`: A 2D array of shape $(k, n)$, with the same dtype as `a`.

The output should be written to the `output` array of shape $(m, n)$, with the same dtype as `a`.

The operation is defined mathematically as:

$$
output[i, j] = \sum_{\ell=0}^{k-1} a[i, \ell] \cdot b[\ell, j]
$$

The implementation must use **Stream-K scheduling**, which differs from standard tile-based scheduling: the K-axis iteration range is partitioned across all SMs (rather than each SM owning a complete output tile), and partial results from the partitioned K work are combined via atomic add into a pre-zeroed output buffer.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- The kernel must implement Stream-K scheduling (persistent kernel + per-SM K-iteration ranges + atomic accumulation), not the simple data-parallel one-tile-per-SM pattern
- Accumulator dtype is fp32; output dtype matches input dtype
- Tensor Core matrix multiplication is recommended

## Example 1

```text
Input: a = [[1, 2], [3, 4]], b = [[5, 6], [7, 8]]
Output: [[19, 22], [43, 50]]
```

## Constraints

- $m \geq 1$, $n \geq 1$, $k \geq 1$
- `a` and `b` have the same dtype (fp16, bf16, or fp32)
