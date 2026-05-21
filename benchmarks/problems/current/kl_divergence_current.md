# KL Divergence (Row-wise, log_target=False)

Implement a program that computes the per-row Kullback-Leibler divergence between a target distribution and a predicted log-probability distribution. The convention matches PyTorch's `F.kl_div(reduction='none').sum(dim=-1)` when `log_target=False`:

- `log_y_pred`: predicted log-probabilities (output of log-softmax)
- `y_true`: target probabilities (output of softmax)

The input consists of:

- `log_y_pred`: A 2D array of shape $(rows, cols)$, fp32.
- `y_true`: A 2D array of shape $(rows, cols)$, fp32 (each row sums to 1).

The output should be written to the `output` array of shape $(rows,)$, in fp32.

For each row $r \in [0, rows)$:

$$
output[r] = \sum_{c=0}^{cols-1} y\_true[r, c] \cdot \big( \log\, y\_true[r, c] - log\_y\_pred[r, c] \big)
$$

By the KL convention, the term contributes $0$ when $y\_true[r, c] = 0$ (treat $0 \log 0$ as $0$).

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output` as fp32
- Use `log(y_true)` only when $y\_true > 0$; otherwise skip the term

## Example 1

```text
Input: log_y_pred = [[-1.0, -2.0, -3.0]], y_true = [[0.5, 0.3, 0.2]]
log(y_true) ≈ [-0.693, -1.204, -1.609]
sum = 0.5*(-0.693 - -1.0) + 0.3*(-1.204 - -2.0) + 0.2*(-1.609 - -3.0)
    = 0.5*0.307 + 0.3*0.796 + 0.2*1.391
    ≈ 0.671
```

## Constraints

- $rows \geq 1$, $cols \geq 1$
- $y\_true[r, c] \geq 0$ and each row of `y_true` sums to 1
- `log_y_pred[r, c]` is finite
