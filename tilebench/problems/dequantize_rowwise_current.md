# Dequantize Row-wise (int8 → fp16)

Implement a program that dequantizes an int8 matrix to fp16 using one fp32 scale per row. Each row of the int8 input is rescaled by its corresponding fp32 absmax-like scale, then divided by 127 (the int8 magnitude normalizer) to produce the fp16 output. This matches the `bitsandbytes` row-wise quant convention.

The input consists of:

- `x`: A 2D int8 array of shape $(rows, cols)$.
- `state_x`: A 1D fp32 array of shape $(rows,)$ holding the per-row scale.

The output should be written to the `output` array of shape $(rows, cols)$, in fp16.

The operation is defined mathematically as:

$$
output[r, c] = \text{fp16}\!\left( state\_x[r] \cdot x[r, c] \cdot \frac{1}{127} \right)
$$

where $r \in [0, rows)$ and $c \in [0, cols)$.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output` as fp16
- The scale is broadcast across the row dimension only (not the column)

## Example 1

```text
Input: x = [[10, -50, 127], [0, 64, -64]] (int8),  state_x = [2.0, 0.5] (fp32)
Output[0] = [2.0 * 10 / 127, 2.0 * -50 / 127, 2.0 * 127 / 127] ≈ [0.157, -0.787, 2.0]
Output[1] = [0.0, 0.5 * 64 / 127, 0.5 * -64 / 127] ≈ [0.0, 0.252, -0.252]
```

## Constraints

- $1 \leq rows$, $1 \leq cols$
- `cols` is a power of two (assumed by row-tile kernels)
- `x` dtype is int8; `state_x` is fp32; output is fp16
