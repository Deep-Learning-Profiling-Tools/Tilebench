# Flash Attention (Forward, Causal)

Implement a program that performs scaled dot-product attention with causal masking, using the FlashAttention algorithm (Dao 2022) — i.e. tile-by-tile online softmax with running max and denominator so the full attention matrix is never materialized.

The input consists of:

- `q`: A 4D array of shape $(batch\_size, n\_heads, seq\_len, head\_dim)$.
- `k`: A 4D array of shape $(batch\_size, n\_heads, seq\_len, head\_dim)$ (same shape as `q`).
- `v`: A 4D array of shape $(batch\_size, n\_heads, seq\_len, head\_dim)$.
- `causal`: A boolean flag; when `True`, position $i$ may only attend to positions $j \leq i$.

The output should be written to the `output` array of shape $(batch\_size, n\_heads, seq\_len, head\_dim)$, with the same dtype as `q`.

Let $s = 1 / \sqrt{head\_dim}$. The operation is defined mathematically as:

$$
attn[b, h, i, j] = \frac{q[b, h, i, :] \cdot k[b, h, j, :]}{\sqrt{head\_dim}}
$$

When `causal=True`, set $attn[b, h, i, j] = -\infty$ for $j > i$. Then softmax over $j$ and weighted-sum the values:

$$
output[b, h, i, :] = \sum_{j=0}^{seq\_len-1} \text{softmax}_j\!\big( attn[b, h, i, :] \big) \cdot v[b, h, j, :]
$$

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- The kernel **must use the FlashAttention online-softmax algorithm**: tile-by-tile K/V loading with running $(m_i, \ell_i, acc_i)$ state — do NOT materialize a full $(seq\_len, seq\_len)$ attention matrix in global memory
- Tensor cores must be used for both the QK and PV matmuls
- Accumulators (running max, denominator, output sum) in fp32; final cast to input dtype

## Example 1

Conceptual single-head, head_dim=2, seq_len=2, causal=True:

```text
q = [[1, 0], [0, 1]], k = [[1, 0], [0, 1]], v = [[10, 20], [30, 40]]
scaled qk = [[1/sqrt(2), 0], [0, 1/sqrt(2)]]
After causal mask: qk[0,:] = [1/sqrt(2), -inf]; qk[1,:] = [0, 1/sqrt(2)]
softmax(qk[0]) = [1.0, 0.0]; softmax(qk[1]) ≈ [0.330, 0.670]
output[0] = 1.0 * [10, 20] = [10, 20]
output[1] ≈ 0.330 * [10, 20] + 0.670 * [30, 40] ≈ [23.4, 33.4]
```

## Constraints

- $batch\_size \geq 1$, $n\_heads \geq 1$, $seq\_len \geq 1$, $head\_dim \geq 1$
- `causal` is the typical case (autoregressive generation)
- Q, K, V have identical shape (full attention, not GQA)
