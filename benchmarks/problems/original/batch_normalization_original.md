# Batch Normalization

Implement batch normalization forward pass for 2D input tensors. Given an input tensor of shape `[N, C]` where N is the batch size and C is the number of features, compute the normalized output using learnable scale (`gamma`) and shift (`beta`) parameters.

For each feature channel $j$, batch normalization computes:

$$
\mu_j = \frac{1}{N} \sum_{i=1}^{N} x_{i,j}
$$

$$
\sigma_j^2 = \frac{1}{N} \sum_{i=1}^{N} (x_{i,j} - \mu_j)^2
$$

$$
\hat{x}_{i,j} = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}
$$

$$
y_{i,j} = \gamma_j \hat{x}_{i,j} + \beta_j
$$

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `output` tensor

## Example 1

```text
Input:  input = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]   (N=3, C=2)
        gamma = [1.0, 1.0]
        beta = [0.0, 0.0]
        eps = 1e-5
Output: output = [[-1.224, -1.224], [0.0, 0.0], [1.224, 1.224]]
```

## Example 2

```text
Input:  input = [[0.0, 1.0], [2.0, 3.0]]   (N=2, C=2)
        gamma = [2.0, 0.5]
        beta = [1.0, -1.0]
        eps = 1e-5
Output: output = [[-1.0, -1.5], [3.0, -0.5]]
```

## Constraints

- $1 \leq N \leq 10{,}000$
- $1 \leq C \leq 1{,}024$
- `eps` = 1e-5
- $-100.0 \leq$ input values $\leq 100.0$
- $0.1 \leq$ gamma values $\leq 10.0$
- $-10.0 \leq$ beta values $\leq 10.0$
- Performance is measured with $N = 5{,}000$
