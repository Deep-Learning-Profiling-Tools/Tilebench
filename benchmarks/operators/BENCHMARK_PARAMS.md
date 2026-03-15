# Benchmark Parameter Summary

Summary of fixed vs. swept parameters for each operator.

---

## mul2

**Operation:** element-wise `x * 2`

| Parameter | Role | Values |
|-----------|------|--------|
| `n` | swept | 1M, 2M, ..., 20M elements |
| `dtype` | swept | fp16, bf16, fp32, int8 |

No fixed parameters — all inputs are fully described by `n` and `dtype`.

---

## rmsnorm

**Operation:** RMSNorm over the last dimension of a 3D tensor `(batch, M, K)`

| Parameter | Role | Values |
|-----------|------|--------|
| `K` | swept | 512, 1024, ..., 10240 (step 512, 20 points) |
| `dtype` | swept | fp16, bf16, fp32 |
| `batch` | fixed | 1 |
| `M` | fixed | 2048 |

`batch=1, M=2048` mimics a typical LLM prefill batch (2k tokens).
`int8` is excluded because squaring int8 elements overflows and RMSNorm is always applied to floating-point activations in practice.

---

## destindex

**Operation:** scatter copy — `out[dest_loc[i], h, :] = kv[i, h, :]` (MLA KV cache layout)

| Parameter | Role | Values |
|-----------|------|--------|
| `seq_len` | swept | 2048, 4096, ..., 40960 (step 2048, 20 points) |
| `dtype` | swept | fp16, bf16, fp32, int8 |
| `batch_size` | fixed | 1 |
| `kv_nope_head_num` | fixed | 12 |
| `kv_rope_head_num` | fixed | 1 |
| `kv_nope_head_dim` | fixed | 128 |
| `kv_rope_head_dim` | fixed | 64 |

Head configuration matches MLA (Multi-head Latent Attention) typical architecture.
Total tokens per case = `batch_size * seq_len`.

**Note:** cuTile implementation is not available — `ct.store()` does not support
runtime-computed scatter indices. The Triton implementation is fully functional.
