# Generic Matrix Multiplication (FP32 / FP16 / FP8)

Implement a program that performs standard matrix multiplication $C = A \cdot B$ across four supported dtypes: fp32, fp16, fp8 e4m3fn, and fp8 e5m2. The accumulator is always fp32, and the result is cast back to the input dtype for the output.

The input consists of:

- `a`: A 2D array of shape $(M, K)$.
- `b`: A 2D array of shape $(K, N)$, with the same dtype as `a`.

The output should be written to the `output` array of shape $(M, N)$, with the same dtype as `a`.

The operation is defined mathematically as:

$$
output[m, n] = \sum_{k=0}^{K-1} a[m, k] \cdot b[k, n]
$$

For fp8 inputs (e4m3fn or e5m2), perform the inner product in fp32 to avoid overflow, then cast the result back to the input fp8 dtype.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- Accumulator dtype is fp32; output dtype matches input dtype
- Tensor Core matrix multiplication is allowed (and recommended) for performance

## Example 1

```text
Input: a = [[1, 2], [3, 4]], b = [[5, 6], [7, 8]]
Output: [[19, 22], [43, 50]]
```

## Constraints

- $M \geq 1$, $N \geq 1$, $K \geq 1$
- `a` and `b` have the same dtype (one of fp32, fp16, fp8 e4m3fn, fp8 e5m2)
- For fp8 dtypes the kernel must internally accumulate in fp32
