# INT8 Matrix Multiplication with 2-Bit Packed B

Implement a program that performs an int8 matrix multiplication where the right-hand operand $B$ is given in a 2-bit packed format. Each byte of `b` stores four 2-bit fields, which decode to $\{-1, 0, 1, 2\}$ via $((byte \gg (2 i)) \,\&\, 3) - 1$ for field $i \in \{0, 1, 2, 3\}$. The accumulator is int32.

The input consists of:

- `a`: A 2D int8 array of shape $(M, K)$.
- `b`: A 2D uint8 array of shape $(K_b, N)$ where $K = 4 \cdot K_b$. Each `b[kb, n]` byte holds 4 packed 2-bit values, one per K position.

The output should be written to the `output` array of shape $(M, N)$ as int32.

Let $B_{unpacked}$ be the dense int8 $(K, N)$ matrix obtained by unpacking:

$$
B_{unpacked}[i \cdot K_b + kb, n] = \big((b[kb, n] \gg (2i)) \;\&\; 3\big) - 1
$$

for $i \in \{0, 1, 2, 3\}$, $kb \in [0, K_b)$, $n \in [0, N)$. The operation is then:

$$
output[m, n] = \sum_{k=0}^{K-1} a[m, k] \cdot B_{unpacked}[k, n]
$$

computed in int32.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output` as int32
- IMMA tensor-core paths are allowed (and encouraged) for performance
- Unpack B in registers / shared memory; do NOT materialize a full int8 $(K, N)$ buffer in global memory

## Example 1

```text
Input: a = [[1, 2, 3, 4]] (int8, shape (1, 4)),
       b = [[0b01_11_10_01]] = [[0x79 = 121]] (uint8, shape (1, 1))
Unpacking field i ∈ {0,1,2,3}: ((121 >> 2i) & 3) - 1
i=0: (121 & 3) - 1 = 1 - 1 = 0
i=1: ((121>>2) & 3) - 1 = (30 & 3) - 1 = 2 - 1 = 1
i=2: ((121>>4) & 3) - 1 = (7 & 3) - 1 = 3 - 1 = 2
i=3: ((121>>6) & 3) - 1 = (1 & 3) - 1 = 1 - 1 = 0
B_unpacked = [[0], [1], [2], [0]] (4×1)
output[0, 0] = 1*0 + 2*1 + 3*2 + 4*0 = 8
Output: [[8]]
```

## Constraints

- $M \geq 1$, $N \geq 1$, $K \geq 4$
- $K$ is divisible by 4 (so $K_b = K/4$ is an integer)
- `a` is int8, `b` is uint8, `output` is int32
