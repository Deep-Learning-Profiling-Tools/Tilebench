# 2D Jacobi Stencil

Given a 2D grid, apply one iteration of the 5-point Jacobi stencil: each interior cell of the output is set to the average of its four cardinal neighbors (top, bottom, left, right) from the input grid. Boundary cells (first/last row and column) are copied unchanged from the input to the output.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in `output`
- Read exclusively from `input` and write exclusively to `output` (do not update `input`)

## Example

Input (4 × 4):

```text
[
  [1.0,  2.0,  3.0,  4.0],
  [5.0,  6.0,  7.0,  8.0],
  [9.0, 10.0, 11.0, 12.0],
  [13.0, 14.0, 15.0, 16.0]
]
```

Output (4 × 4):

```text
[
  [1.0,  2.0,  3.0,  4.0],
  [5.0,  6.0,  7.0,  8.0],
  [9.0, 10.0, 11.0, 12.0],
  [13.0, 14.0, 15.0, 16.0]
]
```

Interior cell (1,1):

```text
0.25 × (input[0,1] + input[2,1] + input[1,0] + input[1,2])
= 0.25 × (2.0 + 10.0 + 5.0 + 7.0) = 6.0
```

Interior cell (1,2):

```text
0.25 × (input[0,2] + input[2,2] + input[1,1] + input[1,3])
= 0.25 × (3.0 + 11.0 + 6.0 + 8.0) = 7.0
```

Interior cell (2,1):

```text
0.25 × (input[1,1] + input[3,1] + input[2,0] + input[2,2])
= 0.25 × (6.0 + 14.0 + 9.0 + 11.0) = 10.0
```

Interior cell (2,2):

```text
0.25 × (input[1,2] + input[3,2] + input[2,1] + input[2,3])
= 0.25 × (7.0 + 15.0 + 10.0 + 12.0) = 11.0
```

## Constraints

- $1 \leq rows, cols$
