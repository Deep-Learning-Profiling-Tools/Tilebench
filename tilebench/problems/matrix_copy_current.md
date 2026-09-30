# Matrix Copy

**Difficulty:** Easy

Implement a program that copies an $N \times N$ matrix from input array $A$ to output array $B$ on the GPU. The program should perform a direct element-wise copy so that $B_{i,j} = A_{i,j}$ for all valid indices.

## Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in matrix `B`

## Example 1

```text
Input:  A = [[1.0, 2.0],
             [3.0, 4.0]]
Output: B = [[1.0, 2.0],
             [3.0, 4.0]]
```

## Example 2

```text
Input:  A = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
Output: B = [[5.5, 6.6, 7.7],
             [8.8, 9.9, 10.1],
             [11.2, 12.3, 13.4]]
```

## Constraints

- $1 \leq N$
