# Sigmoid Activation

Write a GPU program that applies the sigmoid activation function element-wise to a vector. For each element `x` in the input vector `X`, compute `sigmoid(x) = 1 / (1 + exp(-x))` and store the result in the output vector `Y`. The sigmoid function maps any real number to the range (0, 1).

## Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in vector `Y`

## Example 1

```text
Input:  X = [0.0, 1.0, -1.0, 2.0]
Output: Y = [0.5, 0.7311, 0.2689, 0.8808]
```

## Example 2

```text
Input:  X = [0.5, -0.5, 3.0, -3.0]
Output: Y = [0.6225, 0.3775, 0.9526, 0.0474]
```

## Constraints

- $1 \leq N$
