# Rotary Position Embedding (RoPE)

Implement a program that applies rotary position embedding to a 4D query tensor. RoPE pairs adjacent elements of each head's feature dimension and rotates them by a per-position angle whose cosine and sine values are provided as auxiliary tables.

The input consists of:

- `q`: A 4D array of shape $(batch, seq\_len, n\_heads, head\_dim)$. `head_dim` is even.
- `cos`: A 2D array of shape $(seq\_len, head\_dim/2)$.
- `sin`: A 2D array of shape $(seq\_len, head\_dim/2)$.

The output should be written to the `output` array of shape $(batch, seq\_len, n\_heads, head\_dim)$.

Let `half = head_dim / 2`. Split `q[..., :head_dim]` into the first half `q1 = q[..., :half]` and the second half `q2 = q[..., half:]`. The operation is defined mathematically as:

$$
output[b, s, h, d] = \begin{cases}
q1[b, s, h, d] \cdot cos[s, d] - q2[b, s, h, d] \cdot sin[s, d] & \text{if } d < half \\
q2[b, s, h, d-half] \cdot cos[s, d-half] + q1[b, s, h, d-half] \cdot sin[s, d-half] & \text{if } d \geq half
\end{cases}
$$

i.e. the first half of each head is rotated using $(\cos, -\sin)$ and the second half using $(\sin, \cos)$, with `cos`/`sin` shared across batch and heads but per-position along `seq_len`.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- `cos` and `sin` are read-only inputs (don't recompute angles)

## Example 1

Conceptual single-head, head_dim=2 case:

```text
Input: q = [[[[1.0, 2.0]]]], cos = [[0.0]], sin = [[1.0]]   (seq_len=1, n_heads=1)
half = 1; q1 = [1.0]; q2 = [2.0]
output[0,0,0,0] = 1.0 * 0.0 - 2.0 * 1.0 = -2.0
output[0,0,0,1] = 2.0 * 0.0 + 1.0 * 1.0 =  1.0
Output: [[[[-2.0, 1.0]]]]
```

## Constraints

- $batch \geq 1$, $seq\_len \geq 1$, $n\_heads \geq 1$, $head\_dim \geq 2$
- `head_dim` is even
