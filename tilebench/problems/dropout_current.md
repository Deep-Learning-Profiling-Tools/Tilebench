# Dropout (Apply Pre-Generated Mask)

Implement the apply phase of dropout: given a 1D array `x`, a pre-generated keep-mask `x_keep` (0 or 1 per element), and a scalar drop probability `p`, produce an output that zeros out positions where `x_keep` is 0 and rescales the surviving positions by $1 / (1-p)$ so the expected value is preserved.

The input consists of:

- `x`: A 1D array of values.
- `x_keep`: A 1D array of the same length, where each element is either 0 (drop) or 1 (keep).
- `p`: A scalar drop probability, $0 \leq p < 1$.

The output should be written to the `output` array, which has the same length as `x`.

The operation is defined mathematically as:

$$
output[i] = \begin{cases} \dfrac{x[i]}{1 - p} & \text{if } x\_keep[i] = 1 \\ 0 & \text{if } x\_keep[i] = 0 \end{cases}
$$

where $i$ ranges from $0$ to $n-1$.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- The keep-mask is provided as input — do not regenerate it

## Example 1

```text
Input: x = [1.0, 2.0, 3.0, 4.0], x_keep = [1, 0, 1, 0], p = 0.5
Output: [2.0, 0.0, 6.0, 0.0]
```

## Example 2

```text
Input: x = [10.0, 20.0, 30.0], x_keep = [1, 1, 0], p = 0.25
Output: [13.333..., 26.666..., 0.0]
```

## Constraints

- $1 \leq n$
- $0 \leq p < 1$
- `x_keep` contains only 0s and 1s
