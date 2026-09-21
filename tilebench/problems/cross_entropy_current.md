# Cross-Entropy Loss (Per-Sample, No Reduction)

Implement a program that computes the per-sample cross-entropy loss between a batch of logit vectors and integer class targets. The output is one scalar loss per row (no batch reduction).

The input consists of:

- `logits`: A 2D array of shape $(batch\_size, num\_classes)$.
- `targets`: A 1D array of shape $(batch\_size,)$ holding integer class indices in $[0, num\_classes)$.

The output should be written to the `output` array of shape $(batch\_size,)$, with the same dtype as `logits`.

For each sample $b$:

$$
m_b = \max_{c \in [0, num\_classes)} logits[b, c]
$$

$$
output[b] = -\Big( logits[b, targets[b]] - m_b \Big) + \log \sum_{c=0}^{num\_classes-1} e^{\, logits[b, c] - m_b}
$$

This is the standard log-sum-exp / max-subtraction formulation; the subtraction of $m_b$ is for numerical stability.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- No reduction across the batch dimension — return one loss per sample

## Example 1

```text
Input: logits = [[1.0, 2.0, 3.0]], targets = [2]
m = 3.0; log-sum-exp(0, -1, -2) = log(1 + 0.368 + 0.135) = log(1.503) ≈ 0.407
output[0] = -(3.0 - 3.0) + 0.407 = 0.407
```

## Constraints

- $batch\_size \geq 1$, $num\_classes \geq 1$
- $0 \leq targets[b] < num\_classes$ for all $b$
