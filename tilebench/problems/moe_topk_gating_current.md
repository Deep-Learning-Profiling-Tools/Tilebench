# MoE Top-K Gating

Implement a GPU program that performs Top-K Gating for Mixture of Experts (MoE) models. Given a logit matrix of shape `[M, E]` where M is the number of tokens and E is the number of experts, identify the k largest values in each row, extract their indices, and apply softmax to get mixing weights.

For each row i, the operation computes:

$$
indices_i = \operatorname{argsort}(logits_i)[-k:] \\
vals_i = logits_i[indices_i] \\
weights_i = \operatorname{Softmax}(vals_i)
$$

## Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in the `topk_weights` and `topk_indices` arrays

## Example 1

```text
Input:
  logits = [[1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0]]
  M = 2, E = 4, k = 2

Output:
  topk_weights = [[0.2689, 0.7311],
                  [0.7311, 0.2689]]
  topk_indices = [[2, 3],
                  [0, 1]]

Explanation:
  Row 0: Top-2 values are 3.0 and 4.0 at indices 2 and 3.
         Softmax([3.0, 4.0]) = [0.2689, 0.7311]
  Row 1: Top-2 values are 4.0 and 3.0 at indices 0 and 1.
         Softmax([4.0, 3.0]) = [0.7311, 0.2689]
```

## Constraints

- $1 \leq M$ (number of tokens)
- $1 \leq E$ (number of experts)
- $1 \leq k \leq E$ (top-k selection, typically k=2)
