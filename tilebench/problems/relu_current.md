# ReLU Activation

Implement a program that applies the Rectified Linear Unit (ReLU) activation element-wise to a 1D array. ReLU outputs the input value if it is non-negative, and zero otherwise.

The input consists of one array:

- `x`: A 1D array.

The output should be written to the `output` array, which has the same length as `x`.

The operation is defined mathematically as:

$$
output[i] = \max(0, x[i])
$$

where $i$ ranges from $0$ to $n-1$.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`

## Example 1

```text
Input: x = [-2.0, 0.0, 1.5, -0.5, 3.0]
Output: [0.0, 0.0, 1.5, 0.0, 3.0]
```

## Example 2

```text
Input: x = [-1, -2, -3, 4]
Output: [0, 0, 0, 4]
```

## Constraints

- $1 \leq n$
