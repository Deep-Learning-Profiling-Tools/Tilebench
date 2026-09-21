# Leaky ReLU

Implement a program that performs the leaky ReLU activation function on a vector of floating-point numbers. The leaky ReLU function is defined as:

$$
f(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
$$

where $\alpha$ is a small positive constant (0.01 in this problem).

## Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in vector `output`
- Use $\alpha = 0.01$ as the leaky coefficient

## Example 1

```text
Input:  x = [1.0, -2.0, 3.0, -4.0]
Output: y = [1.0, -0.02, 3.0, -0.04]
```

## Example 2

```text
Input:  x = [-1.5, 0.0, 2.5, -3.0]
Output: y = [-0.015, 0.0, 2.5, -0.03]
```

## Constraints

- $1 \leq N$
