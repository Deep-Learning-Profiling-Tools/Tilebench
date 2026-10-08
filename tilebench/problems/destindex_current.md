# Destination-Indexed Copy (KV-Cache Scatter)

Implement a program that performs an index-scatter copy used in autoregressive KV-cache writes. Given two "rows" tensors (`kv_nope` and `kv_rope`) and an integer permutation vector `dest_loc`, write each source row $t$ into the destination buffer at row `dest_loc[t]` for both `o_nope` and `o_rope`.

The input consists of:

- `kv_nope`: A 3D array of shape $(T, H_{nope}, D_{nope})$.
- `kv_rope`: A 3D array of shape $(T, H_{rope}, D_{rope})$.
- `dest_loc`: A 1D int32 array of shape $(T,)$. Each element is a destination row index, $0 \leq dest\_loc[t] < T$.
- `o_nope`, `o_rope`: Pre-existing output buffers of the same shape as their source counterparts (provided as starting values).

The output consists of two arrays:

- `out_nope`: A 3D array of shape $(T, H_{nope}, D_{nope})$ — `o_nope` with `kv_nope` scattered in.
- `out_rope`: A 3D array of shape $(T, H_{rope}, D_{rope})$ — `o_rope` with `kv_rope` scattered in.

The operation is defined as:

$$
out\_nope[\,dest\_loc[t], h, d\,] = kv\_nope[t, h, d]
$$

$$
out\_rope[\,dest\_loc[t], h, d\,] = kv\_rope[t, h, d]
$$

Positions in `o_nope` / `o_rope` not touched by any source `t` retain their original value.

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the arrays `out_nope` and `out_rope`
- Do not mutate `o_nope` / `o_rope` in place — start from a copy
- `dest_loc` is guaranteed to be a permutation (each value $0..T-1$ appears exactly once)

## Example 1

```text
Input: kv_nope = [[[1, 2]], [[3, 4]]] (T=2, H=1, D=2)
       dest_loc = [1, 0]
       o_nope (initial) = [[[0, 0]], [[0, 0]]]
out_nope[1] = kv_nope[0] = [[1, 2]]
out_nope[0] = kv_nope[1] = [[3, 4]]
Output: out_nope = [[[3, 4]], [[1, 2]]]
```

## Constraints

- $T \geq 1$, $H_{nope} \geq 1$, $D_{nope} \geq 1$, $H_{rope} \geq 1$, $D_{rope} \geq 1$
- `dest_loc` is a permutation of $\{0, \dots, T-1\}$
