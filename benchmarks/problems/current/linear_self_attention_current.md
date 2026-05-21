# Linear Self-Attention (ELU+1 Feature Map)

Implement a program that performs linear self-attention with the $\phi(x) = \text{ELU}(x) + 1$ feature map (Katharopoulos et al., 2020). Linear attention reorders the matrix multiplications so the cost is $O(M \cdot D^2)$ instead of $O(M^2 \cdot D)$.

The input consists of:

- `Q`: A 2D array of shape $(M, D)$, fp32.
- `K`: A 2D array of shape $(M, D)$, fp32.
- `V`: A 2D array of shape $(M, D)$, fp32.
- `eps`: A small positive scalar for numerical stability (default $10^{-6}$).

The output should be written to the `output` array of shape $(M, D)$, in fp32.

Let $\phi(x) = \text{ELU}(x) + 1$ applied element-wise — i.e. $\phi(x) = x + 1$ if $x > 0$ and $\phi(x) = e^x$ otherwise. Define:

$$
S = \phi(K)^\top \cdot V \in \mathbb{R}^{D \times D}
$$

$$
Z = \sum_{m=0}^{M-1} \phi(K)[m, :] \in \mathbb{R}^{D}
$$

$$
output[m, :] = \frac{\phi(Q)[m, :] \cdot S}{\phi(Q)[m, :] \cdot Z + \epsilon}
$$

(Division is element-wise of the numerator row by the scalar denominator $\phi(Q)[m, :] \cdot Z$.)

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output` in fp32
- Decompose into two reductions ($S$ and $Z$) followed by a tile-friendly elementwise/MMA combine — this is the **linear-time** variant of attention
- Tensor cores are recommended for the $S$ and $\phi(Q) \cdot S$ matmuls

## Example 1

Conceptual M=1, D=2:

```text
Q = [[1, -1]], K = [[1, -1]], V = [[2, 3]]
phi(Q) = phi(K) = [[2, e^(-1)]] ≈ [[2, 0.368]]
S = [[2, 0.368]]^T @ [[2, 3]] = [[4, 6], [0.736, 1.104]]
Z = [[2, 0.368]]
phi(Q) @ S = [[2*4 + 0.368*0.736, 2*6 + 0.368*1.104]] ≈ [[8.271, 12.406]]
phi(Q) @ Z = 2*2 + 0.368*0.368 ≈ 4.135
output = [[8.271/4.135, 12.406/4.135]] ≈ [[2.0, 3.0]]
```

## Constraints

- $M \geq 1$, $D \geq 1$
- `Q`, `K`, `V` all have shape $(M, D)$, dtype fp32
- $\epsilon > 0$
