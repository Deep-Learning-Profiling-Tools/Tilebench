# Quantize Global (fp32 → fp16)

Implement a program that casts every element of a 1D fp32 array to fp16 (half precision). This is a pure dtype-conversion operator; no scaling, no rounding mode choices beyond IEEE-754 round-to-nearest-even.

The input consists of:

- `x`: A 1D array of fp32 values.

The output should be written to the `output` array of the same length, in fp16.

The operation is defined mathematically as:

$$
output[i] = \text{fp16}(x[i])
$$

where the cast follows IEEE-754 binary16 conversion (round-to-nearest-even, with values outside the fp16 range saturating to $\pm \infty$ or $0$).

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output` as fp16

## Example 1

```text
Input: x = [1.5, -2.0, 0.0, 65500.0]
Output: [1.5, -2.0, 0.0, 65504.0]   (fp16 round-to-nearest-even of 65500 is 65504)
```

## Constraints

- $1 \leq n$
- Input is fp32, output is fp16
