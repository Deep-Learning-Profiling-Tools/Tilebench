# Top-K Selection

Implement a program that, given a 1D fp32 array of length $N$ and an integer $k$, returns the $k$ largest values **sorted in descending order**.

The input consists of:

- `input`: A 1D fp32 array of length $N$.
- `N`: An integer specifying the length of `input`.
- `k`: An integer specifying how many top values to select, $1 \leq k \leq N$.

The output should be written to the `output` array of shape $(k,)$ in fp32, holding the $k$ largest values of `input` sorted in descending order.

The operation is defined as: produce $output$ such that for all $i \in [0, k)$ and $j \in [0, k-1)$,

$$
output[j] \geq output[j+1]
$$

and $\{output[0], \dots, output[k-1]\}$ is exactly the multiset of the $k$ largest values of `input`. (If `input` has ties at the threshold, any consistent tie-breaking is acceptable.)

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output` as fp32 of length $k$
- Output is sorted descending

## Example 1

```text
Input: input = [3.0, 1.0, 4.0, 1.5, 5.0], N = 5, k = 3
Top 3 values: 5.0, 4.0, 3.0
Output: [5.0, 4.0, 3.0]
```

## Example 2

```text
Input: input = [-1.0, -2.0, -3.0], N = 3, k = 2
Output: [-1.0, -2.0]
```

## Constraints

- $1 \leq k \leq N$
- Input dtype is fp32
- Output is sorted descending
