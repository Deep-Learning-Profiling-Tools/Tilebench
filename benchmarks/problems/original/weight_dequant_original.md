# Weight Dequantization

Implement a GPU program that "dequantizes" a weight matrix on the GPU. You are given an input matrix `X` of shape `[M, N]` containing quantized values and a scale matrix `S` of shape `[ceil(M/T), ceil(N/T)]`, where `T` is the tile size.

For each element $X_{i,j}$, the corresponding scale factor is $S_{row,col}$, where $row = \lfloor i/T \rfloor$ and $col = \lfloor j/T \rfloor$. The output $Y_{i,j}$ should be computed as:

$$
Y_{i,j} = X_{i,j} \times S_{row,col}
$$

## Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in the output buffer `Y`

## Example 1

```text
Input:
M = 4, N = 4, TILE_SIZE = 2
X = [
  [10, 10,  5,  5],
  [10, 10,  5,  5],
  [ 2,  2,  8,  8],
  [ 2,  2,  8,  8]
]
S = [
  [0.5, 2.0],
  [4.0, 0.25]
]

Output:
Y = [
  [ 5.0,  5.0, 10.0, 10.0],
  [ 5.0,  5.0, 10.0, 10.0],
  [ 8.0,  8.0,  2.0,  2.0],
  [ 8.0,  8.0,  2.0,  2.0]
]

Explanation:
Tile (0,0) of X is multiplied by S[0,0] (0.5).
Tile (0,1) of X is multiplied by S[0,1] (2.0).
Tile (1,0) is multiplied by S[1,0] (4.0).
Tile (1,1) is multiplied by S[1,1] (0.25).
```

## Constraints

- $1 \leq M, N \leq 8192$
- $TILE\_SIZE \in \{16, 32, 64, 128\}$
- Performance is measured with $M = 8192$, $N = 8192$, $TILE\_SIZE = 128$
