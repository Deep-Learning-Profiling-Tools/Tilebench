# Histogramming

Implement a program that, given a 1D int32 array of length $N$ and an integer `num_bins`, computes the histogram of the values — i.e. how many times each integer value in $[0, num\_bins)$ appears in the input.

The input consists of:

- `input`: A 1D int32 array of length $N$. All values are in $[0, num\_bins)$.
- `N`: An integer giving the length of `input`.
- `num_bins`: An integer giving the number of histogram bins.

The output should be written to the `output` array of shape $(num\_bins,)$ as int32.

The operation is defined mathematically as:

$$
output[b] = \big|\{ i \in [0, N) : input[i] = b \}\big|
$$

for $b \in [0, num\_bins)$.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output` as int32
- Values outside $[0, num\_bins)$ are not expected; behaviour on out-of-range input is undefined
- Atomic accumulation across parallel threads is acceptable (and typical)

## Example 1

```text
Input: input = [0, 2, 1, 0, 2, 2, 1], N = 7, num_bins = 3
Output: [2, 2, 3]
```

## Example 2

```text
Input: input = [0, 0, 0, 0], N = 4, num_bins = 4
Output: [4, 0, 0, 0]
```

## Constraints

- $N \geq 1$, $num\_bins \geq 1$
- $0 \leq input[i] < num\_bins$
- Input dtype is int32; output dtype is int32
- $\sum_b output[b] = N$
