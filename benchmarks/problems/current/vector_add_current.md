# Vector Addition

Implement a program that performs element-wise addition of two 1D arrays of the same length.

The input consists of two arrays:

- `x`: A 1D array.
- `y`: A 1D array of the same length as `x`.

The output should be written to the `output` array, which has the same length as the inputs.

The operation is defined mathematically as:

$$
output[i] = x[i] + y[i]
$$

where $i$ ranges from $0$ to $n-1$.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`

## Example 1

```text
Input: x = [1, 2, 3, 4], y = [5, 6, 7, 8]
Output: [6, 8, 10, 12]
```

## Example 2

```text
Input: x = [-1.5, 2.0, 0.0], y = [0.5, -2.0, 3.0]
Output: [-1.0, 0.0, 3.0]
```

## Constraints

- $1 \leq n$
- `x` and `y` have the same length
