# Argmax (Row-wise)

Implement a program that, given a 2D array, returns the column index of the maximum value in each row.

The input consists of:

- `x`: A 2D array of shape $(M, N)$.

The output should be written to the `output` array of shape $(M,)$, holding int64 indices.

The operation is defined mathematically as:

$$
output[m] = \underset{n \in [0, N)}{\operatorname{argmax}} \; x[m, n]
$$

for $m \in [0, M)$. If multiple columns share the maximum, return the **smallest** index (first occurrence).

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output` as int64

## Example 1

```text
Input: x = [[1.0, 3.0, 2.0], [5.0, 5.0, -1.0]]
Output: [1, 0]
```

## Example 2

```text
Input: x = [[-2.0, -3.0, -1.5]]
Output: [2]
```

## Constraints

- $1 \leq M$, $1 \leq N$
- Output dtype is int64
