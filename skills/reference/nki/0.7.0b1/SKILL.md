---
name: nki-reference
dsl: nki
version: 0.7.0b1
kind: reference
---

# NKI 0.7.0b1 Reference for AWS Trainium2 (native PyTorch, LNC=2)

> Attribution and licence. This reference was written for the TileArena (TileBench++) study. It is based on: (1) the API names, signatures and docstrings of the installed `nki` 0.7.0b1 package (Apache-2.0, Copyright Amazon.com, Inc. or its affiliates); (2) aws-neuron/neuron-agentic-development at commit `ee25e45` (Apache-2.0; NOTICE: "Neuron Agentic Development. Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved."); (3) the AWS Neuron documentation (CC BY-SA 4.0, https://awsdocs-neuron.readthedocs-hosted.com); and (4) device tests run on this host. This text is a modified, condensed adaptation, and it is shared under CC BY-SA 4.0 where it adapts AWS documentation. Full source list: `SOURCES.md`; licence texts: `LICENSE.apache-2.0`, `NOTICE.neuron-agentic-development`, `LICENSE-NOTES.md`.

**Scope.** Public AWS sources, each checked against the installed nki 0.7.0b1 on trn2. Where public material and the installed package disagree, the installed package and the device tests win. The sections below list the known differences.

| Item | Installed value |
|---|---|
| nki | `0.7.0b1+34883233359.g2aa6a82a` (pre-release; `nki.__version__`) |
| neuronx-cc | `2.0.404830.0a0+55f4bf89` |
| torch / torch-neuronx | `2.14.0+cpu` / `2.14.3.0.0+unknown.dev` (native PyTorch, device `"neuron"`; torch-xla is not installed; kernels are called eagerly. `torch.compile` was not tested) |
| Hardware | trn2.3xlarge: 1 Trainium2 chip, 8 physical NeuronCore-v3 (gen3), `NEURON_LOGICAL_NC_CONFIG=2` (4 logical cores of 2 physical cores each); one process uses one logical core |
| Compiler target | `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, `NEURON_CC_FLAGS="--target trn2 --lnc 2"` |

**Source tags.** `[pkg]` = installed package source, signature or docstring (`evidence/introspection.json`). `[doc]` = AWS Neuron documentation or neuron-agentic-development@ee25e45. `[dev:x]` = device test `evidence/smoke_tests/x_*.py` run on this trn2 host (results in `evidence/smoke_results.json`). Untagged statements are definitions or plain consequences of tagged ones.

---

## 0. Ten rules that prevent most failures

1. Import `import nki`, `import nki.language as nl`, `import nki.isa as nisa`. Decorate the top-level kernel with `@nki.jit`. Call it with `torch` tensors that live on `torch.device("neuron")` [dev:a].
2. Allocate every output with `nl.ndarray(shape, dtype=..., buffer=nl.shared_hbm)` and **return** it (a tuple for several outputs). A kernel that returns nothing fails in this PyTorch integration [dev:m].
3. In this process, a bare call `kernel(...)` already runs with **LNC=2**: two programs, one per physical core, `nl.num_programs() == 2`. Without a `program_id` split, both cores do the full work. The result is still correct [dev:a, dev:b].
4. On-chip tiles: axis 0 is the partition axis and must be ≤ 128. SBUF/PSUM tensors need at least 2 dimensions [pkg]. HBM tensors are tiled by slicing them in Python loops [dev:a].
5. Compute instructions are destination-passing: `nisa.<op>(dst=..., ...)`. Allocate `dst` first with `nl.ndarray`. Never write `x = nisa.nc_matmul(a, b)` [dev:c].
6. `nisa.nc_matmul(dst, stationary, moving)` computes `stationary.T @ moving`. Both operands are SBUF tiles with the contraction on the partition axis (K ≤ 128). `dst` is a **float32 PSUM** tile with M ≤ 128 rows and N ≤ 512 columns [pkg, dev:c, dev:d].
7. PSUM cannot be DMA'd. Copy PSUM → SBUF with `nisa.tensor_copy` (which can also cast), then `nisa.dma_copy` SBUF → HBM [dev:m, dev:c].
8. A compute-engine operand that starts at a partition offset must start at partition 0, 32, 64 or 96. DMA may start at any partition [dev:m].
9. The kernel body is a Python subset. List and generator comprehensions, lambdas, `try`, keyword-only parameters, closures over enclosing-function locals, and direct calls of nested functions are rejected. Module-level helpers, globals, `range`, `if`, `while`, `math` and `dict` work [dev:l, dev:k].
10. `nl.arange`, `nl.mgrid`, `nl.par_dim`, `mask=`, `nl.spmd_dim`, `nl.nc` and `nisa.rsqrt` do not exist in 0.7.0b1 [pkg, dev:d].

---

## 1. Writing, launching and specialising a kernel

### 1.1 `@nki.jit`

```text
nki.jit(fn=None, *, compile_config=None, specialization_config=None, **kwargs)   [pkg]
```

- Use it as `@nki.jit` or as `@nki.jit(compile_config=nki.compiler.CompileConfig(...))`. Any other keyword raises an error. `lnc=` raises `TypeError` ("Set LNC at the callsite with bracket syntax"). `platform_target=` raises `ValueError` (use `NEURON_PLATFORM_TARGET_OVERRIDE`). `mode=` is ignored with a `DeprecationWarning` [pkg].
- `CompileConfig` fields and defaults are `address_rotation=AddressRotation.AUTO`, `non_ssa_legalization=False`, `dead_code_elim=True`, `barrier_check=False`, `prefetch_auto=False` and `nki_alloc_fallback=True`. `SpecializationConfig` has one field, `allow_nested_config_override=False` [pkg]. Kernels do not need any of them.
- The decorator returns an `nki.Kernel`. `kernel[1](...)` / `kernel[2](...)` select the LNC degree (§2).
- A `@nki.jit` kernel called from inside another kernel is inlined into the caller's trace [pkg, dev:l]. A plain module-level function works too.

### 1.2 Calling from PyTorch

```python
import torch, torch_neuronx            # registers the "neuron" device (torch also autoloads it via an entry point)
x = torch.randn(256, 512, dtype=torch.float16).to("neuron")
y = torch.randn(256, 512, dtype=torch.float16).to("neuron")
out = add_kernel(x, y)                 # returns a torch tensor on neuron:0
torch.neuron.synchronize()             # or torch_neuronx.synchronize()
```

- Each `torch.Tensor` argument becomes an `NkiTensor` in `nl.shared_hbm` with the tensor's shape and dtype [pkg]. Inside the kernel, `x.shape` (a tuple of Python ints) and `x.dtype` are compile-time values.
- Non-tensor arguments are **compile-time constants**: Python numbers, `str`, `bool`, tuples of these, and nl dtypes (for example `out_dtype=nl.float32`). They are part of the cache key. Each new value, shape or dtype triggers a new trace and compile, and results stay correct [dev:k, dev:p]. Positional and keyword arguments both work at the call site [dev:k].
- Configuration must reach the kernel as an argument or a module-level global. A closure over a local variable of an enclosing function (a "kernel factory") fails: `error: unbound variable 'c'` [dev:k].
- Returned tensors keep the dtype and shape of the `nl.ndarray` you returned [dev:a, dev:i]. Returning a tuple returns a tuple of torch tensors [dev:l].
- Writing into an input tensor (`nisa.dma_copy(dst=x, ...)`) modifies the caller's tensor [dev:m]. Do not do this unless the in-place update is intended.
- A non-contiguous input view (for example `x.t()`) is handled. The kernel sees the logical values [dev:p].
- Dtype mapping at the boundary: `torch.int64` becomes `int32` and the output dtype is `int32`; `torch.bool` becomes `uint8`; `torch.float64` becomes `float32` [pkg, dev:o]. On trn2, passing a `torch.float8_e4m3fn` tensor raises `RuntimeError: float8_e4m3fn is not supported on this target (TRN2 and earlier) ...` [dev:d]. `torch.float8_e5m2` passes through bit-exactly [dev:d].
- The first call with a new signature traces and compiles. Repeated calls reuse the compiled kernel [dev:a]. `torch_neuronx` enables a persistent NKI trace cache by default (`NKI_ENABLE_TRACE_CACHE=1`) and warns that it "may not capture all kernel environment dependencies or non-argument values" [pkg]. This is one more reason to pass every configuration value as a kernel argument.

### 1.3 Minimal complete kernel (element-wise add)

```python
# example: add_kernel
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def add_kernel(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    P = nl.tile_size.pmax                      # 128
    rows, cols = a.shape
    for i in range(rows // P):                 # Python range: unrolled at trace time
        a_t = nl.ndarray((P, cols), dtype=a.dtype, buffer=nl.sbuf)
        b_t = nl.ndarray((P, cols), dtype=b.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_t, src=a[i * P:(i + 1) * P, :])
        nisa.dma_copy(dst=b_t, src=b[i * P:(i + 1) * P, :])
        c_t = nl.ndarray((P, cols), dtype=a.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=c_t, data1=a_t, data2=b_t, op=nl.add)
        nisa.dma_copy(dst=out[i * P:(i + 1) * P, :], src=c_t)
    return out
```

This assumes `rows` is a multiple of 128. §4.6 shows remainder tiles. [dev:a, dev:r]

### 1.4 TileArena module interface (generic)

The task prompt defines the exact contract. In general you write a Python module with:

- `run(...)`: takes neuron torch tensors (and Python scalars), calls one or more `@nki.jit` kernels and returns neuron torch tensors;
- `get_last_config()`: returns a JSON-serialisable `dict` describing the configuration used by the last `run` call (for example tile sizes or the chosen code path).

Keep the kernels at module level, so that the default frontend can resolve their source, helpers and globals. Pass shape-dependent choices to the kernel as arguments, never as closures.

### 1.5 What the kernel body may contain (default frontend)

`NKI_COMPILER_FRONTEND` is unset, which selects `custom` (= the "parser"). This frontend accepts a subset of Python and evaluates it with NKI's own partial evaluator. The other public value is `standard` (the tracer) [pkg `nki/_config.py`, `nki/compiler/frontend.py`]. Everything below was observed with the default frontend [dev:l, dev:k]:

| Construct inside `@nki.jit` | Result |
|---|---|
| Module-level helper functions (default args; keyword arguments at the call site) | works |
| Module-level globals (for example `SCALE = 3.0`) | works |
| `range`, `if/elif/else`, Python `while` on trace-time ints | works |
| `max`, `min`, `abs`, `int`, `float`, `len`, `math.ceil`, `math.sqrt`, `dict` literals, f-strings, `assert cond, f"..."` | works |
| `x is None`, `x is not None`, `x != None` | works (library code that uses `is not` on other objects prints a harmless `warning: the 'is not' operator is not supported in NKI, use '!='`) |
| `print(...)` | runs at trace time, once per program (twice under LNC=2) |
| `t[...] = 9.0` (memset), `out[...] = sbuf_tile` (DMA) | works |
| `return a, b` | works (a tuple is returned to torch) |
| list comprehension, generator expression | `error: unsupported expression` |
| `lambda` | `error: unsupported expression` |
| `try/except` | `error: NKI does not support 'try' statements; use 'if/else' control flow within kernels` |
| helper with keyword-only parameter `def f(a, *, c=3)` | `warning: keyword-only arguments are not supported in NKI`, then an error |
| nested `def` called directly | `error: inner functions can only be used as fori_loop/while_loop body arguments; direct calls are not supported` |
| closure over an enclosing function's local | `error: unbound variable '<name>'` |

The migration guide in the 0.6.0 docs says the tracer becomes the default in 0.7.0 [doc]. In this installed 0.7.0b1 the default is still the parser [pkg].

### 1.6 How errors appear

Trace and compile errors are raised from the torch call as `RuntimeError` [dev:c, dev:d]:

```text
RuntimeError: [NkiDispatch] Python lower_func failed for cache_key=...: AssertionError[WITH __notes__]:
error: failed to compile NKI kernel: Collected 1 different diagnostics: - [x2] error: <message> note: <file>:<line>: ...
```

`[x2]` means the same diagnostic was collected twice: the kernel is traced once per program under LNC=2, which is also why a trace-time `print` appears twice. The `note:` lines point to your kernel line. Some limits are only detected by the backend compiler (`neuronx-cc`). These show as `... exit code 70: [INTERNAL_ERROR] [NCC_xxxx] ...` [dev:n, dev:o]. Set `NKI_TRACEBACK_FILTERING=off` to keep NKI's internal frames [pkg].

---

## 2. LNC=2 on this host

### 2.1 Launch forms (verified)

| Call | `nl.num_programs()` | `nl.program_id(0)` | `nl.program_ndim()` | Notes |
|---|---|---|---|---|
| `kernel(...)` | 2 | 0 on core 0, 1 on core 1 | 1 | a following `kernel[2]` call hit the compile cache |
| `kernel[2](...)` | 2 | 0 / 1 | 1 | explicit |
| `kernel[1](...)` | 1 | 0 | 0 | runs on one physical core; **no** runtime error here |

[dev:b]

- The `nki.jit` docstring says "The default is LNC=1" and that a mismatch with `NEURON_LOGICAL_NC_CONFIG` causes a runtime error. In this environment, an unset LNC is resolved from `NEURON_LOGICAL_NC_CONFIG` (= 2) [pkg `nki/framework/_lnc.py`], and `kernel[1]` ran correctly [dev:a, dev:b].
- `nl.program_id(0)` and `nl.num_programs()` are trace-time Python ints. Use them in arithmetic, slicing, `range` and Python `if` [dev:b]. `kernel[n]` accepts only 1 or 2 [pkg].
- A kernel without a split runs its whole body on both physical cores. Both cores write identical values to the same output locations, so the result is correct but the work is done twice [dev:a, dev:b].

### 2.2 Splitting work between the two physical cores

```python
# example: split_add
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def split_add(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    P = 128
    n_prog = nl.num_programs()            # 2 for kernel(...) / kernel[2](...), 1 for kernel[1](...)
    pid = nl.program_id(0)                # 0 or 1
    rows_per_prog = a.shape[0] // n_prog
    for i in range(rows_per_prog // P):
        r0 = pid * rows_per_prog + i * P
        a_t = nl.ndarray((P, a.shape[1]), dtype=a.dtype, buffer=nl.sbuf)
        b_t = nl.ndarray((P, a.shape[1]), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_t, src=a[r0:r0 + P, :])
        nisa.dma_copy(dst=b_t, src=b[r0:r0 + P, :])
        nisa.tensor_tensor(dst=a_t, data1=a_t, data2=b_t, op=nl.add)   # dst may alias an input
        nisa.dma_copy(dst=out[r0:r0 + P, :], src=a_t)
    return out
```

The same kernel is correct under `kernel[1]`, where one program covers all rows [dev:b, dev:r]. Both programs must run the same control flow. Divergent dynamic control flow between the cores is undefined [doc lnc.md].

### 2.3 Memory visibility

| Buffer | Visibility | Source |
|---|---|---|
| `nl.sbuf`, `nl.psum` | private to each program (physical core). The same `nl.ndarray` allocation holds different data on core 0 and core 1 | [pkg buffers.py], [dev:b] |
| `nl.shared_hbm` | visible to all programs. Kernel inputs and outputs live here | [pkg], [dev:h] |
| `nl.private_hbm` (`nl.hbm` is an alias) | per program (per-instance HBM scratch) | [pkg] |

On Trainium2 the chip has four HBM stacks, and each is shared by two physical NeuronCore-v3. With LNC=2 the two cores share one address space [doc LNC config].

### 2.4 Inter-core synchronisation and exchange (NeuronCore-v3+)

```text
nisa.core_barrier(data, cores, engine=nisa.engine.gpsimd, name=None)          # cores must be (0, 1)
nisa.sendrecv(src, dst, send_to_rank, recv_from_rank, pipe_id, dma_engine=nisa.dma_engine.dma, name=None)
nisa.sendrecv_compute(src, dst, send_to_rank, recv_from_rank, reduce_op, name=None)  # reduce_op: nl.add / nl.maximum / nl.minimum
```

[pkg] Verified uses [dev:h]:

```python
# example: barrier_exchange
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def barrier_exchange(dummy):
    # each core writes its own row of a shared_hbm scratch, waits, then reads both rows
    out = nl.ndarray((2, 2, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    scratch = nl.ndarray((2, 64), dtype=nl.float32, buffer=nl.shared_hbm)   # internal scratch, not returned
    pid = nl.program_id(0)
    mine = nl.ndarray((1, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=mine, value=float(pid + 1))
    nisa.dma_copy(dst=scratch[pid:pid + 1, :], src=mine)
    nisa.core_barrier(data=scratch, cores=(0, 1))
    both = nl.ndarray((2, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=both, src=scratch)
    nisa.dma_copy(dst=out[pid], src=both)      # out[0] == out[1] == [[1...], [2...]]
    return out
```

```python
# SBUF tile exchange between the two cores (inside an LNC=2 kernel)
other = 1 - nl.program_id(0)
nisa.sendrecv(src=send, dst=recv, send_to_rank=other, recv_from_rank=other, pipe_id=0)   # recv = peer's send
nisa.sendrecv_compute(src=send, dst=res, send_to_rank=other, recv_from_rank=other, reduce_op=nl.add)  # res = send + peer's send
```

`sendrecv` needs `src` and `dst` in SBUF with the same shape and dtype. `sendrecv_compute` needs both in SBUF or both in private HBM [pkg]. The `nki.collectives` APIs (`all_reduce`, `all_gather`, ...) exist [pkg] but were not examined or tested for this reference.

---

## 3. Memory, buffers and tiles

### 3.1 Allocation

```text
nl.ndarray(shape, dtype, buffer=nl.sbuf, name='', address=None)                    [pkg]
nl.zeros(shape, dtype, buffer=nl.sbuf, name='')      nl.ones(...)      nl.full(shape, fill_value, dtype, buffer=nl.sbuf, name='')
nl.zeros_like(x, dtype=None, buffer=None, name='')
nl.empty_like(x, dtype=None, buffer=None, name='', *, address=None)
```

- Buffers: `nl.sbuf`, `nl.psum`, `nl.shared_hbm`, `nl.private_hbm`, and `nl.hbm` (= `nl.private_hbm`) [pkg]. Use `nl.shared_hbm` for outputs.
- `nl.ndarray` does not initialise memory. `nl.zeros/ones/full` allocate (SBUF by default) and fill with `nisa.memset`. `nl.zeros(..., buffer=nl.shared_hbm)` fails: `Creating HBM tensor with init value is not supported.` [pkg, dev:o]. `nl.empty_like(t)` gives an uninitialised tile with `t`'s shape, dtype and buffer [dev:o].
- SBUF and PSUM tensors must have ≥ 2 dimensions (partition + free) [pkg]. Axis 0 is the partition axis (≤ 128). The remaining axes are free axes and may be several (for example `(128, 4, 64)`) [dev:g].
- The PSUM `address=(partition_offset, free_offset)` must be bank-aligned. You normally leave `address=None` [pkg].
- Each `nl.ndarray` call creates a new tensor, also inside an unrolled loop body. With `address=None` the compiler assigns on-chip addresses [pkg].

### 3.2 Tile-size constants (`nl.tile_size`)

| Attribute | Value on trn2 | Meaning [pkg `_tile_size.py`] |
|---|---|---|
| `pmax` | 128 | max partition dimension |
| `psum_bank_fmax` | 512 | fp32 elements per PSUM bank per partition (2 KiB) |
| `psum_bank_fmax_bytes` | 2048 | bytes per PSUM bank per partition |
| `psum_num_banks` | 8 | PSUM banks per partition (property; 7 when DMA transpose is lowered to PE) |
| `gemm_stationary_fmax` | 128 | max free size of `nc_matmul` stationary (= M) |
| `gemm_moving_fmax` | 512 | max free size of `nc_matmul` moving (= N) |
| `bn_stats_fmax` | 512 | max free size for `bn_stats` |
| `sbuf_fmax_bytes` | 212984 | usable SBUF bytes per partition (224 KiB − 16 KiB reserved − 8 B) |
| `sbuf_fmax` | 53246 | the same in fp32 elements |
| `sbuf_size_bytes` | 29360128 | total SBUF (128 × 224 KiB) |
| `psum_min_align`, `sbuf_min_align` | 4 | byte alignment of free-dim addresses |
| `psum_fmax`, `total_available_sbuf_size` | 512, = `sbuf_fmax_bytes` | deprecated names |

The version-dependent properties (`sbuf_*`, `psum_num_banks`) work only inside a kernel. Outside one they raise `RuntimeError: No backend set` [pkg]. Inside a kernel on trn2 they returned the values above [dev:l]. `nl.tile_size.psum_fmax_bytes`, used by an older public guide, does **not** exist [pkg].

### 3.3 Capacity per physical NeuronCore-v3

- SBUF: **28 MiB** = 128 partitions × 224 KiB (NeuronCore-v2 had 24 MiB). Usable: 212,984 B per partition [doc trainium2_arch, pkg].
- PSUM: **2 MiB** = 128 partitions × 16 KiB = 8 banks × 2 KiB per partition [doc, pkg].
- With LNC=2 each of the two programs has its own SBUF and PSUM of this size (§2.3).
- Over-allocation is reported by the backend: `[NCC_INLA001] ... Allocated memory out of bound {_mem_0}@PSUM-1<0,0>(128x32768)` for a `(128, 8192)` fp32 PSUM tile, and the same `@SB<0,0>(128x240000)` message for a `(128, 60000)` fp32 SBUF tile [dev:o].

### 3.4 Views, slicing and partition offsets

- Slices: Python ints, `a:b`, `a:b:step`, negative indices, `...`, and `nl.ds(start, size)` (= `start:start+size`) [pkg, dev:m]. An integer index on an HBM tensor removes the axis. On SBUF/PSUM, axis 0 is kept (size 1) [pkg].
- `NkiTensor` view methods (no copy) [pkg]: `reshape(shape)`, `permute(dims)`, `broadcast(dim, size)`, `expand_dim(dim)`, `squeeze_dim(dim)`, `reshape_dim(dim, shape)`, `flatten_dims(start, end)`, `slice(dim, start, end, step=1)`, `select(dim, index)`, `view(dtype)` (bit reinterpret; last-dim size scales by the dtype ratio), and `ap(...)` (§4.4). On-chip `reshape` must keep the partition size.
- HBM views as DMA sources were verified [dev:m]: `x.reshape((2, 128, F))[1, :, 0:F//2]` == `x[128:, :F//2]`; `x.reshape((128, 2, F))[:, 1, 0:F:2]` == `x[1::2, ::2]`; `x[-128:, F//2:]`.
- **Partition-offset rule for compute engines.** An operand that is a partition slice of an SBUF tile must start at partition 0, 32, 64 or 96 (observed with `nisa.tensor_scalar`; the MLIR verifier names the instruction). `t[32:64]`, `t[64:96]`, `t[96:128]` and `t[0:100]` work. `t[8:24]`, `t[40:72]`, `t[5:6]` and `t[33:34]` fail with `'nisa.tensor_scalar_arith' op Expected 'dst' partition start in {0, 32, 64, 96} for a 16-partition access, but got 8` [dev:m]. DMA has no such restriction: `nisa.dma_copy(dst=t[5:6, :], ...)` and `t[37:40, :]` work [dev:m]. To place data at an arbitrary partition, use DMA.
- Assignment shortcuts: `t[...] = 1.0` (memset), `sbuf_t[...] = other_sbuf_t` (tensor_copy), and `hbm[...] = sbuf_t` (dma_copy) [pkg tensor.py, dev:l].

---

## 4. Data movement

### 4.1 `nisa.dma_copy`

```text
nisa.dma_copy(dst, src, priority=None, oob_mode=nisa.oob_mode.error, dge_mode=nisa.dge_mode.unknown,
              engine=nisa.engine.unknown, name=None)                                  [pkg]
```

- `src` and `dst` may each be in HBM or SBUF, including SBUF → SBUF [pkg, dev:m]. **PSUM is not allowed**: `dma_copy requires HBM or SBUF tensors, got src=...psum...` [dev:m].
- The total number of elements of `src` and `dst` must match [pkg].
- dtype conversion is automatic (via fp32). fp32 → bf16 by DMA matched torch's round-to-nearest-even bit for bit [pkg, dev:i].
- `priority=` is NeuronCore-v4 only: `dma_copy priority is only supported for NeuronCore-v4 or newer` [dev:n].
- `dge_mode` is `none`, `swdge`, `hwdge` or `unknown` (the default lets the compiler choose). Leave the default unless you need hwdge or swdge features [pkg].

### 4.2 `nl.load` / `nl.store` (exist in 0.7.0b1)

```text
nl.load(src, dtype=None)  -> new SBUF tile of src's shape (dtype converted if given)    [pkg]
nl.store(dst, value)                                                                     [pkg]
nl.load_transpose2d(src, dtype=None) -> new SBUF tile of shape (src.shape[1], src.shape[0])
```

These are thin wrappers marked experimental: `nl.load` = `nl.ndarray(src.shape, ..., nl.sbuf)` + `nisa.dma_copy`; `nl.store` = `nisa.dma_copy` [pkg]. They work on trn2 [dev:j], including `dtype=nl.float32` on a bf16 input (exact) and `load_transpose2d` for bf16 and fp32. `nl.load` of a slice with more than 128 rows fails: `dma_copy dst partition dimension 256 exceeds maximum 128` [dev:j]. The Beta-1 keyword `mask=` is not accepted: `load() got an unexpected keyword argument 'mask'` [dev:d]. Older guides say "never use nl.load/nl.store". That applied to the Beta-1 semantics, not to these wrappers.

### 4.3 Transposes by DMA

```text
nisa.dma_transpose(dst, src, axes=None, priority=None, dge_mode=nisa.dge_mode.unknown, oob_mode=nisa.oob_mode.error, name=None)   [pkg]
```

- `dst` is SBUF. `src` is HBM or SBUF, with `src.dtype == dst.dtype` and `dst.shape` equal to the transposed `src.shape` [pkg]. 2-D permutation `(1, 0)`. 3-D `(2, 1, 0)`. 4-D `(3, 1, 2, 0)`.
- Default `dge_mode`: bf16 and fp32 `[128, 64]` HBM → `[64, 128]` SBUF was bit-exact [dev:m, dev:j].
- `dge_mode=nisa.dge_mode.hwdge`: requires a 2-byte dtype (`dma_transpose hwdge requires 2-byte dtype, got float32`) and `src.shape[-1] <= 128`. For 2-D, `src.shape[0] == 16` (bf16 `[16, 128]` worked). For 3-D/4-D, `src.shape[0]` must be in {1, 2, 4, 8, 16} [pkg, dev:m].
- TensorE alternative: `nisa.nc_transpose` (§5.1).

### 4.4 Access patterns: `.ap()`

```text
t.ap(pattern, offset=None, scalar_offset=None, vector_offset=None, indirect_dim=0, dtype=None)   [pkg]
```

- `pattern` is required: a list of `[stride, count]` pairs in elements, outermost first. Omitting it fails with `ap() missing value for required argument 'pattern'` [dev:m]. `offset` is an element offset added to the view's base offset.
- Example: rows 128..255 of a `[256, F]` HBM tensor: `x.ap(pattern=[[F, 128], [1, F]], offset=128 * F)` [dev:m].
- A zero partition stride replicates one HBM row into all partitions: `row.ap(pattern=[[0, 128], [1, F]], offset=0)` (`row` is `[1, F]` in HBM) [dev:q].
- `scalar_offset` (a runtime value: `fori_loop` index, register or 1×1 SBUF tile) adds `scalar_offset * stride(indirect_dim)`. `vector_offset` (an SBUF `[P, 1]` index tile on an HBM tensor) gives per-partition gather/scatter with `dma_copy` [pkg]. §6.2 shows the runtime-offset form.

### 4.5 DMA compute

```text
nisa.dma_compute(dst, srcs, reduce_op, scales=None, unique_indices=True, oob_mode=nisa.oob_mode.error, name=None)   [pkg]
```

The result is `dst = srcs[0] + srcs[1] + ...` computed in fp32 by the DMA engines (up to 16 sources, equal element counts). Only `reduce_op=nl.add` and all-1.0 `scales` are supported (`dma_compute only supports nl.add for reduce_op`) [pkg, dev:h]. Verified SBUF → SBUF and HBM → HBM (`nisa.dma_compute(dst=out, srcs=[a, b], reduce_op=nl.add)` with `a`, `b`, `out` in HBM) [dev:h]. With a `vector_offset` source it performs scatter-add / gather-add read-modify-write [pkg].

### 4.6 Tiling a 2-D HBM tensor, including remainders

```python
# example: add_one_any_shape
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def add_one_any_shape(x):
    R, F = x.shape
    P = 128
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    for i in range((R + P - 1) // P):
        r0 = i * P
        rows = min(P, R - r0)                     # Python int at trace time
        t = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:rows, :], src=x[r0:r0 + rows, :])
        nisa.tensor_scalar(dst=t[0:rows, :], data=t[0:rows, :], op0=nl.add, operand0=1.0)
        nisa.dma_copy(dst=out[r0:r0 + rows, :], src=t[0:rows, :])
    return out
```

Partial tiles are slices that start at partition 0. This replaces the Beta-1 `mask=`. Verified for (200, 300) and (77, 513) [dev:p, dev:r]. Free sizes such as 300 and 513 worked for DMA and `tensor_scalar`. The free size only has to fit in a partition (§3.2).

---

## 5. Compute engines and their instructions

All compute instructions are destination-passing (`dst=` first) and return an `NkiInstruction` handle, not a tile [pkg]. `engine=` takes `nisa.engine.{tensor, vector, scalar, gpsimd, unknown}`. `unknown` (the default) lets the compiler choose [pkg]. The GpSimd engine cannot access PSUM [pkg].

### 5.1 TensorE: `nc_matmul`, `nc_transpose`

```text
nisa.nc_matmul(dst, stationary, moving, is_stationary_onezero=False, is_moving_onezero=False, is_transpose=False,
               accumulate=None, tile_position=(), tile_size=(), perf_mode=nisa.matmul_perf_mode.none, name=None)   [pkg]
```

- Semantics: `dst[M, N] = stationary[K, M].T @ moving[K, N]`. `stationary` and `moving` must be **SBUF** tiles and `dst` a **PSUM** tile. The errors are `nc_matmul moving must be in [sbuf], got psum` and `nc_matmul stationary must be in [sbuf], got shared_hbm` [pkg, dev:d].
- Limits on NeuronCore-v3: K (partition dim of both inputs) ≤ 128, M ≤ 128, N ≤ 512 per instruction. `N = 1024` fails with `Matmul moving free dimension 1024 exceeds max 512 for nc_version=...gen3` [pkg, dev:d]. A PSUM tile may be wider than a bank. Write it with several instructions, each to a ≤ 512-column slice (a `(128, 1024)` fp32 PSUM tile written as two N=512 slices worked) [dev:d].
- `dst` dtype must be **float32** on trn2 (`nc_matmul dst dtype must be float32 on gen3, got bfloat16`) [pkg, dev:c]. Accumulation is fp32. Cast while copying PSUM → SBUF with `nisa.tensor_copy` [dev:c].
- Input dtypes: `float8_e4m3`, `float8_e5m2`, `bfloat16`, `float16`, `tfloat32`, `float32`. If one input is fp32/tf32 the other must be too (`nc_matmul: if one input is tfloat32/float32, both must be`). There is **no int8 matmul** (`nc_matmul stationary dtype int8 not supported. Valid: (float8_e4m3, float8_e5m2, bfloat16, float16, tfloat32, float32)`) and **no OCP `float8_e4m3fn`** on trn2 (same message) [pkg, dev:d]. `float8_e4m3fn` needs NeuronCore-v4 [pkg].
- `accumulate`: `False` overwrites the PSUM region and `True` adds to it. With `None` (default) the compiler overwrites on the first write to a PSUM location and accumulates after that [pkg]. Explicit `accumulate=(k > 0)` and the default both gave correct K-tiled sums [dev:c]. Do not `memset` PSUM and then accumulate. On NeuronCore-v2/v3, matmul accumulation onto PSUM initialised by a non-matmul instruction is undefined [pkg].
- `perf_mode=nisa.matmul_perf_mode.double_row` (fp8 only, trn2+): the contraction is split over the partition axis and a first free axis of size 2. Operands are `[128, 2, M]` and `[128, 2, N]`, so one instruction contracts K = 256. Use the same k → (partition, j) mapping for both operands. fp8_e4m3 and fp8_e5m2 (cast on chip with `tensor_copy`) worked. With bf16 operands the backend failed (`[NCC_IBIR322] Unspecified`) [pkg, doc, dev:n]. `double_row` cannot be combined with column tiling [pkg].
- `tile_position=(row, col)` / `tile_size=(rows, cols)` select a sub-array of the 128×128 PE grid (row tiles 32/64) [pkg]. Leave them unset unless you need that.
- `is_transpose=True` is the bit-exact transpose mode (used by `nc_transpose`) [pkg].

```text
nisa.nc_transpose(dst, data, engine=nisa.engine.unknown, name=None)                       [pkg]
```

- TensorE (`engine=nisa.engine.tensor`): input SBUF ≤ `[128, 128]`, output in **PSUM with the same dtype as the input** on trn2. A bf16 `[128, 64]` → `[64, 128]` transpose was bit-exact. A fp32 PSUM `dst` for bf16 input fails with `nc_matmul (transpose mode) dst dtype must match input dtype on gen3+` [pkg, dev:g].
- VectorE (`engine=nisa.engine.vector`): ≤ `[32, 32]`, SBUF or PSUM. `[64, 64]` fails with `Vector engine transpose requires shape <= [32, 32]` [pkg, dev:g].
- `nl.matmul(x, y, transpose_x=False)` and `nl.transpose(x)` are convenience wrappers that allocate their PSUM result with the **input dtype**. On trn2 this works only for fp32 inputs. With bf16 inputs, `nl.matmul` fails with the "dst dtype must be float32 on gen3" error [pkg, dev:c]. Use `nisa.nc_matmul` with an explicit fp32 PSUM tile.

### 5.2 VectorE / ScalarE / GpSimd instructions

Arithmetic is performed in fp32 internally, and the result is cast to `dst.dtype` at no extra cost [pkg]. Bitwise ops treat integers as bit patterns [pkg].

```text
nisa.tensor_tensor(dst, data1, data2, op, engine=nisa.engine.unknown, name=None)
nisa.tensor_scalar(dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False,
                   engine=nisa.engine.unknown, name=None)
nisa.activation(dst, op, data, bias=None, scale=1.0, reduce_op=None, reduce_res=None,
                reduce_cmd=nisa.reduce_cmd.idle, engine=nisa.engine.scalar, name=None)
nisa.activation_reduce(dst, op, data, reduce_op, reduce_res, bias=None, scale=1.0, name=None)
nisa.tensor_reduce(dst, op, data, axis, negate=False, keepdims=False, name=None)
nisa.tensor_copy(dst, src, engine=nisa.engine.unknown, name=None)
nisa.memset(dst, value, engine=nisa.engine.unknown, name=None)
nisa.iota(dst, pattern, offset=0, channel_multiplier=0, name=None)
nisa.reciprocal(dst, data, name=None)
nisa.tensor_tensor_scan(dst, data0, data1, initial, op0, op1, reverse0=False, reverse1=False, name=None)
```

[pkg]

- **`tensor_tensor`**: element-wise op on two tiles with the same partition size and the same number of elements per partition. `data1` and `data2` cannot both be in PSUM (`Both data1 and data2 cannot be in PSUM simultaneously`). `dst` must be in SBUF or PSUM (`tensor_tensor dst must be in [sbuf, psum], got shared_hbm`) [pkg, dev:g]. When all operands are int32/uint32, the default engine is GpSimd with exact 32-bit integer math. Forced onto VectorE (`engine=nisa.engine.vector`), an int32 add of values around 2^30 was inexact in 8011 of 8192 elements, because of the fp32 internal math [pkg, dev:o].
- **`tensor_scalar`**: `(data op0 operand0) op1 operand1`, where each operand is a compile-time scalar or a `[P, 1]` tile broadcast along the free axis. **A tile operand used with an arithmetic op must be float32.** An fp16 `[P, 1]` operand fails with `'nisa.tensor_scalar_arith' op 'operand0' must be float32, got 'f16'` [pkg, dev:g]. Cast it first with `tensor_copy`. Use `reverse0/1=True` to swap operand order for non-commutative ops. On ScalarE (trn2) only `(multiply, add)`, `(multiply, None)` and `(add, None)` combinations exist [pkg].
- **`activation`** (ScalarE): `dst = op(data * scale + bias)`. `scale` is an fp32 scalar or `[P, 1]` fp32 tile. `bias` is a scalar (NeuronCore-v3+) or a `[P, 1]` tile [pkg]. The optional fused reduction sums `op(...)` along the free axis: `reduce_op=nl.add` (the only allowed op), `reduce_res=[P, 1]` tile, `reduce_cmd=nisa.reduce_cmd.reset_reduce`. `exp(2x + b)` with a vector bias and its row sum were verified [dev:g]. Activation functions on ScalarE: `nl.exp, log, sqrt, rsqrt, tanh, sigmoid, relu, gelu, gelu_apprx_sigmoid, gelu_apprx_tanh, gelu_dx, gelu_apprx_sigmoid_dx, silu, silu_dx, softplus, mish, erf, erf_dx, reciprocal, copy, arctan, square, sign, sin, abs` (`prelu` only via `activate2` on NeuronCore-v4) [pkg]. `sin`, `arctan`, `log`, `sqrt`, `rsqrt` and `reciprocal` have limited valid input ranges [pkg]. `engine=nisa.engine.gpsimd` is NeuronCore-v4 only (`'activation with engine=gpsimd' is supported for ...gen4+`) [dev:g].
- **`tensor_reduce`** (VectorE): reduces the **last contiguous free axes**, ending at the final axis. Axis 0 (partition) cannot be reduced [pkg]. With `data (P, 4, 64)`, `axis=2` gives `dst (P, 4)`, and `axis=(1, 2), negate=True` gives `dst (P, 1)` = −sum [dev:g]. Ops: `nl.add, nl.multiply, nl.maximum, nl.minimum` and bitwise ops on integers [pkg].
- **`tensor_copy`**: SBUF/PSUM → SBUF/PSUM copy and dtype cast. fp32 → bf16 and fp32 → fp16 matched torch bit for bit (round to nearest even), and bf16 → fp32 is exact [dev:i]. `nisa.activation(op=nl.copy)` casts the same way [dev:i]. The GpSimd engine cannot read or write PSUM [pkg].
- **`memset`**: fills with a compile-time constant (`value=3` into int32, `value=1.0` into float; an integral float such as `2.0` into an int32 tile also stored 2) [dev:e, dev:l]. **`iota`**: `dst[p, ...] = offset + p * channel_multiplier + Σ step_i * idx_i` (int32 math, cast to `dst.dtype`, SBUF only). `pattern=[[1, 64]], offset=1, channel_multiplier=64` gave `1 + 64p + j` [pkg, dev:g].
- **`reciprocal`**: 1/x [dev:g]. There is no `nisa.rsqrt` in 0.7.0b1. Use `nisa.activation(op=nl.rsqrt, ...)` or `nl.rsqrt(x)` [pkg].
- **`tensor_tensor_scan`**: per partition, `res[:, i] = op1(op0(data0[:, i], res[:, i-1]), data1[:, i])`, with `res[:, -1] := initial`. `op0=nl.multiply` on a ones tile with `op1=nl.add` gives a running sum along the free axis [pkg, dev:g].
- Comparisons (`nl.greater`, `nl.equal`, ...) in `tensor_tensor` write 1/0 into a uint8 or fp32 `dst` [dev:o]. `nl.right_shift` on int32 is a **logical** shift (it matched an unsigned shift, not torch's arithmetic `>>`). `nl.bitwise_and` with a constant is bit-exact [dev:o].
- `engine=` selection: `tensor_tensor` on VectorE and on GpSimd, and `tensor_copy` on ScalarE, all verified [dev:g].

Other instructions that exist on trn2 [pkg] (signatures in `COMPATIBILITY.md`):

- `select_reduce`: select between `on_true` and `on_false` by an integer predicate, optional max-reduction.
- `range_select`: select by comparing the free index against per-partition bounds; `on_false_value` must be FP32_MIN.
- `affine_select`: select by an affine expression of the partition and free indices (GpSimd).
- `tensor_copy_predicated`, `scalar_tensor_tensor`, `tensor_scalar_reduce`, `tensor_scalar_cumulative`, `tensor_partition_reduce`.
- `bn_stats` / `bn_aggr`, `max8`, `nc_find_index8`, `nc_match_replace8`, `local_gather`, `nc_n_gather`, `nc_stream_shuffle`, `nonzero_with_count`, `sequence_bounds`.
- `rng`, `dropout`, `set_rng_seed`, `rand_set_state` / `rand_get_state` (GpSimd on NeuronCore-v3). `rand2` is NeuronCore-v4 only [pkg].
- `topk` (GpSimd, NeuronCore-v3+; BF16 input in a 16-partition "snake" layout; a minimal call ran on trn2 [dev:n]).

Read the docstring (`help(nisa.<name>)`) before using one of these: each has layout and dtype rules.

### 5.3 `nl.*` convenience operations

`nl.add, subtract, multiply, divide, maximum, minimum, power, exp, log, sqrt, rsqrt, tanh, sigmoid, relu, gelu, silu, abs, negative, square, reciprocal, sin, cos, ...` each **allocate and return a new tile** [pkg]:

- binary ops on two full tiles → `tensor_tensor`; with a scalar or a `[P, 1]` tile operand → `tensor_scalar`. The result is allocated in **the buffer of the full-size data tile** (SBUF stays SBUF; a PSUM data tile gives a PSUM result) with its dtype unless `dtype=` is given [pkg].
- most unary ops → `nisa.activation` (some, such as `floor`/`ceil`/`sin`, are composed of several instructions), with the result in the input's buffer [pkg].
- reductions `nl.sum / nl.max / nl.min / nl.mean / nl.prod / nl.var / nl.all (x, axis, dtype=None, keepdims=False)` → a new SBUF tile [pkg]. **`nl.max(x, axis)` is a reduction. `nl.maximum(a, b)` is element-wise.** Both exist [pkg].
- also `nl.where, nl.broadcast_to, nl.expand_dims, nl.copy, nl.zeros_like` [pkg].

Verified: `nl.load` → `nl.add`, `nl.exp`, `nl.multiply(t, 2.0)`, `nl.sum(t, axis=1, keepdims=True)`, `nl.maximum(t, rowvec)` → `nl.store` [dev:g]. `nl.broadcast_to(tile[1, F], (128, F))` [dev:q]. They are marked experimental [pkg]. Their internal temporaries count against SBUF, so prefer explicit `nisa.*` calls when you need control over dtypes, engines or buffers.

### 5.4 Operator names (`nl.<op>`, used as `op=` / `op0=`)

Arithmetic: `add, subtract, multiply, divide, maximum, minimum, average, power, fmod, mod` (`abs_max`, `abs_min` exist but are NeuronCore-v4 only [pkg]). Comparison: `equal, not_equal, less, less_equal, greater, greater_equal`. Bitwise (integer): `bitwise_and, bitwise_or, bitwise_xor, invert, left_shift, right_shift`. Logical: `logical_and, logical_or, logical_xor, logical_not`. Unary: `abs, negative, sign, square, copy, bypass, floor, ceil, trunc`. Activation: see §5.2. `nl.power` in `tensor_tensor` runs on GpSimd (no PSUM operands) [pkg].

---

## 6. Loops and control flow

### 6.1 Compile-time (unrolled) loops

- `range(...)`, `nl.range(...)` (= `nl.affine_range`), `nl.affine_range`, `nl.sequential_range` and `nl.static_range` are **all fully unrolled at trace time**. Their 0.7.0b1 docstrings call the `nl.*_range` functions aliases of Python `range` and mark them deprecated ("Prefer using `range()` directly") [pkg]. A loop-carried update (accumulating into one SBUF tile) gave the correct result with all five, with no warnings [dev:f]. Older guidance about choosing `affine_range` vs `sequential_range` for correctness does not apply to 0.7.0b1. The loop index is a Python int: use it freely in slices and arithmetic.
- Python `if` on trace-time values (shapes, constants, `program_id`) selects code at trace time [dev:l, dev:b]. Every unrolled iteration emits its own instructions. For a very large or runtime-dependent trip count, use `nl.fori_loop`.

### 6.2 On-device loops: `nl.fori_loop`

```text
nl.fori_loop(lower, upper, body_fun, step=1)      # step: positive compile-time int; returns None   [pkg]
```

- A plain rolled loop on the device. `body_fun(i)` is a nested function. It may read variables of the enclosing kernel (tensors and constants), and it returns nothing [pkg, doc].
- `i` is an `NkiScalar` when the bounds are ints, or a `VirtualRegister` when a bound is a register [pkg]. **Under the default frontend, Python arithmetic on `i` fails**: `x[nl.ds(i * P, P), :]` gives `error: 'mul' expected (int, int) or (float, float) ... got (object, int)` [dev:e]. Use `i` directly as a runtime offset. Three verified ways to index row tile `i` [dev:e]:

```python
# example: fori_rows
import nki
import nki.language as nl
import nki.isa as nisa

P = 128


@nki.jit
def fori_rows(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    N = x.shape[1]
    T = x.shape[0] // P
    x3 = x.reshape((T, P, N))                  # HBM view: tile index on axis 0
    o3 = out.reshape((T, P, N))

    def body(i):                               # (a) i = 0, P, 2P, ... thanks to step=P
        t = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x[nl.ds(i, P), :])
        nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=1.0)
        nisa.dma_copy(dst=out[nl.ds(i, P), :], src=t)

    nl.fori_loop(0, x.shape[0], body, step=P)

    def body2(j):                              # (b) runtime index on a reshaped view
        t = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=o3[j])
        nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=2.0)
        nisa.dma_copy(dst=o3[j], src=t)

    nl.fori_loop(0, T, body2)

    def body3(j):                              # (c) explicit access pattern + scalar_offset
        src = o3.ap(pattern=[[N, P], [1, N]], offset=0, scalar_offset=j, indirect_dim=0)
        t = nl.ndarray((P, N), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=src)
        nisa.tensor_scalar(dst=t, data=t, op0=nl.subtract, operand0=1.0)
        nisa.dma_copy(dst=o3.ap(pattern=[[N, P], [1, N]], offset=0, scalar_offset=j, indirect_dim=0), src=t)

    nl.fori_loop(0, T, body3)
    return out                                 # = (x + 1) * 2 - 1
```

- Rules [doc nki-060-dynamic-loop-migration-guide, pkg]: carry state across iterations in SBUF/HBM tiles allocated **before** the loop and updated in place. A tile allocated inside the body cannot be used after the loop. The loop variable is read-only. `fori_loop` and `while_loop` cannot sit inside `nl.no_reorder()`. Under LNC=2 both programs must run the same trip count.
- Runtime trip count: load an int32 value into SBUF, then `reg = nisa.register_alloc(); nisa.register_load(reg, cnt_tile); nl.fori_loop(0, reg, body)`. With 2 from an input tensor, exactly 2 tiles were processed [dev:e].

### 6.3 `nl.while_loop`

```text
nl.while_loop(init, body_fun)        # init: VirtualRegister; body_fun(r) -> next condition register   [pkg]
```

This is a real `while` (zero iterations if `init` is 0). Verified pattern [dev:e]:

```python
cnt = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
nisa.memset(dst=cnt, value=3)
reg = nisa.register_alloc()
nisa.register_load(reg, cnt)

def body(r):
    nisa.tensor_tensor(dst=acc, data1=acc, data2=xt, op=nl.add)    # acc, xt allocated before the loop
    nisa.tensor_scalar(dst=cnt, data=cnt, op0=nl.add, operand0=-1)
    nisa.register_load(r, cnt)
    return r

nl.while_loop(reg, body)             # acc += xt three times
```

### 6.4 `nl.dynamic_range`, `nl.no_reorder`

- `for i in nl.dynamic_range(T):` (with a reshaped-view index `x3[i]`) compiled and ran correctly under the default frontend, with **no** warning [dev:f]. The 0.6.0 release notes deprecate it and say NKI warns. Under the tracer frontend it is removed [doc]. Prefer `nl.fori_loop`.
- `with nl.no_reorder():` keeps the enclosed instructions in program order [pkg, dev:f].

---

## 7. Generic patterns (all verified)

### 7.1 Row reduction along the free axis

```python
m = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.tensor_reduce(dst=m, op=nl.maximum, data=x_tile, axis=1)        # row max (x_tile: [P, F])
s = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
nisa.tensor_reduce(dst=s, op=nl.add, data=x_tile, axis=1)            # row sum
```

A sum of an activation's output can be fused into `nisa.activation(..., reduce_op=nl.add, reduce_res=s, reduce_cmd=nisa.reduce_cmd.reset_reduce)` (§5.2) [dev:g].

### 7.2 Reduction across partitions

`tensor_reduce` cannot reduce axis 0. Two options [pkg]:

1. Multiply by a ones vector on TensorE: `ones[P, 1].T @ x[P, N]` gives the column sums `[1, N]` in PSUM [dev:g].
2. Transpose (`nisa.nc_transpose`, ≤ 128×128 per instruction) and then `tensor_reduce` along the free axis.

```python
# example: colsum
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def colsum(x):                                 # x: [128, N] fp32, N <= 512
    N = x.shape[1]
    out = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    ones = nl.ndarray((x.shape[0], 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    acc = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=acc, stationary=ones, moving=xt)
    r = nl.ndarray((1, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r, src=acc)
    nisa.dma_copy(dst=out, src=r)
    return out
```

Broadcasting the other way (one row to all partitions): a zero-stride `.ap` DMA (§4.4), `nl.broadcast_to`, or `nc_matmul(stationary=ones[1, 128], moving=row[1, F])` (K = 1) [dev:q].

### 7.3 K-tiled matmul with PSUM accumulation

```python
# example: matmul_lhsT
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def matmul_lhsT(lhsT, rhs, out_dtype):
    # out[M, N] = lhsT[K, M].T @ rhs[K, N]; K % 128 == 0, M <= 128, N <= 512
    K, M = lhsT.shape
    _, N = rhs.shape
    out = nl.ndarray((M, N), dtype=out_dtype, buffer=nl.shared_hbm)
    acc = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)      # fp32 PSUM on trn2
    KT = 128
    for k in range(K // KT):
        s = nl.ndarray((KT, M), dtype=lhsT.dtype, buffer=nl.sbuf)
        m = nl.ndarray((KT, N), dtype=rhs.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=s, src=lhsT[k * KT:(k + 1) * KT, :])
        nisa.dma_copy(dst=m, src=rhs[k * KT:(k + 1) * KT, :])
        nisa.nc_matmul(dst=acc, stationary=s, moving=m, accumulate=(k > 0))
    res = nl.ndarray((M, N), dtype=out_dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=res, src=acc)          # PSUM -> SBUF, cast to out_dtype
    nisa.dma_copy(dst=out, src=res)
    return out
```

Verified with bf16 and fp16 inputs (K = 256), fp32 inputs, and output dtypes fp32 and fp16 [dev:c, dev:r]. Larger M or N need more output tiles (each ≤ 128 × 512 per instruction). Larger K needs more accumulation steps. If the left operand is stored as `[M, K]`, obtain `[K, M]` tiles on chip with `nisa.nc_transpose` (TensorE, PSUM result in the input dtype, then `tensor_copy` to SBUF) or `nisa.dma_transpose` (§4.3) [dev:g, dev:m].

### 7.4 Transposing a tile on TensorE

```python
# example: transpose_tile
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def transpose_tile(x):                         # x: [128, 64] bf16 -> out: [64, 128]
    out = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.shared_hbm)
    xt = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x)
    tp = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.psum)   # same dtype as input
    nisa.nc_transpose(dst=tp, data=xt, engine=nisa.engine.tensor)
    ts = nl.ndarray((x.shape[1], x.shape[0]), dtype=x.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ts, src=tp)
    nisa.dma_copy(dst=out, src=ts)
    return out
```

[dev:g, dev:r]

### 7.5 Per-row scale and bias (free-axis broadcast)

```python
v = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)            # MUST be fp32 for arithmetic ops
nisa.dma_copy(dst=v, src=v_hbm)                                       # DMA converts dtype if v_hbm is fp16/bf16
nisa.tensor_scalar(dst=r, data=x_tile, op0=nl.multiply, operand0=v, op1=nl.add, operand1=1.0)
```

[dev:g, dev:i]

---

## 8. Data types on trn2 (NeuronCore-v3)

| nl dtype | torch dtype at the kernel boundary | Notes |
|---|---|---|
| `nl.float32`, `nl.float16`, `nl.bfloat16` | `torch.float32 / float16 / bfloat16` | all engines |
| `nl.tfloat32` | (none) | TensorE input type |
| `nl.float8_e4m3` (legacy) | (none on trn2; see note) | TensorE input, double_row; produce on chip with `tensor_copy` [dev:d] |
| `nl.float8_e5m2` | `torch.float8_e5m2` | TensorE input, double_row [dev:d] |
| `nl.float8_e4m3fn` (OCP) | `torch.float8_e4m3fn` → **rejected on trn2** | NeuronCore-v4 only for matmul [dev:d] |
| `nl.int8, int16, int32, uint8, uint16, uint32` | same names; `torch.int64` → int32; `torch.bool` → uint8 | Vector/GpSimd/DMA; **no TensorE** [dev:o, dev:d] |
| `nl.bool_` | (via uint8) | predicates |
| `nl.float8_e5m2_x4, float8_e4m3fn_x4, float4_e2m1fn_x4, float8_e8m0fnu` | — | MX formats for NeuronCore-v4 instructions |

- `UNSAFE_FP8FNCAST=1` makes torch `float8_e4m3fn` tensors map to legacy `nl.float8_e4m3` on trn2. The two formats differ in maximum and saturation, so this is a reinterpretation [pkg].
- PSUM `dst` of `nc_matmul`: fp32 only on NeuronCore-v3. `nc_transpose` PSUM `dst`: the input dtype [pkg, dev:c, dev:g].
- Float → narrower float casts round to nearest even [dev:i, doc data types]. Vector/Scalar engine arithmetic is fp32 internally [pkg]. An int32 add forced onto VectorE was inexact for values near 2^30, while the default (GpSimd) int32 add was exact [dev:o].
- **NeuronCore-v4-only APIs (unavailable on trn2)** [pkg docstrings, dev:n]: `nisa.exponential` (`exponential requires NeuronCore-v4 or newer`), `nisa.activate2` (`'activate2' is supported for ...gen4+`), `nisa.quantize_mx` (backend: `[NCC_IBIR530] QuantizeMx is not supported on arch Trn2, must be Trn3 or greater`), `nisa.nc_matmul_mx` (docstring: v4+), `dma_copy(priority=...)`, `activation(engine=nisa.engine.gpsimd)`, tensor indirection `t.indirect(index)`, fp8_e4m3fn matmul, bf16 matmul `dst`, and N > 512 per matmul.

---

## 9. Common errors and fixes (exact messages observed on this host)

| Message (inside the `RuntimeError` text) | Cause | Fix |
|---|---|---|
| `dma_copy dst partition dimension 129 exceeds maximum 128` | partition axis > 128 (the `nl.ndarray` call itself is accepted) | tile rows in chunks of ≤ 128 |
| `Matmul moving free dimension 1024 exceeds max 512 for nc_version=...gen3` | N > 512 in one `nc_matmul` | split N into ≤ 512 slices (a wider PSUM tile is fine) |
| `nc_matmul contraction dimension ... exceeds pmax=128` / `stationary free dimension ... exceeds gemm_stationary_fmax=128` | K or M > 128 [pkg] | tile K (accumulate) / tile M |
| `nc_matmul moving must be in [sbuf], got psum` / `stationary must be in [sbuf], got shared_hbm` | operand not in SBUF | `dma_copy` HBM → SBUF; `tensor_copy` PSUM → SBUF |
| `nc_matmul dst dtype must be float32 on gen3, got bfloat16` | bf16 PSUM matmul output (also from `nl.matmul` on bf16) | fp32 PSUM tile; cast on `tensor_copy` |
| `nc_matmul stationary dtype int8 not supported. Valid: (...)` | int8 / float8_e4m3fn matmul on trn2 | convert to bf16/fp16 (or legacy fp8_e4m3 / fp8_e5m2) on chip |
| `nc_matmul: if one input is tfloat32/float32, both must be` | fp32 × bf16 | use the same precision class for both |
| `float8_e4m3fn is not supported on this target (TRN2 and earlier)` | torch `float8_e4m3fn` argument | pass bf16/fp16 and convert on chip |
| `nc_matmul (transpose mode) dst dtype must match input dtype on gen3+` | `nc_transpose` into a different-dtype PSUM tile | PSUM tile with the input dtype |
| `Vector engine transpose requires shape <= [32, 32]` | VectorE transpose too large | `engine=nisa.engine.tensor` (≤ 128×128) |
| `'nisa.tensor_scalar_arith' op 'operand0' must be float32, got 'f16'` | non-fp32 `[P, 1]` operand | `tensor_copy` it to fp32 |
| `... Expected 'dst' partition start in {0, 32, 64, 96} for a 16-partition access, but got 8` | compute on a misaligned partition slice | align to a multiple of 32, or move rows with DMA |
| `Both data1 and data2 cannot be in PSUM simultaneously` | two PSUM inputs | copy one to SBUF |
| `tensor_tensor dst must be in [sbuf, psum], got shared_hbm` | compute writing HBM | compute into SBUF, then `dma_copy` |
| `dma_copy requires HBM or SBUF tensors, got src=...psum` | DMA from PSUM | `tensor_copy` PSUM → SBUF first |
| `[NCC_INLA001] ... Allocated memory out of bound ...@PSUM` / `...@SB` | on-chip over-allocation | smaller tiles; ≤ 16 KiB PSUM and ≤ 212,984 B SBUF per partition |
| `failed to resolve name 'nki.language.arange'` / `'nki.language.par_dim'` | removed Beta-1 API | slices / `nl.ds`; plain partition sizes |
| `load() got an unexpected keyword argument 'mask'` | Beta-1 masking | slice partial tiles (§4.6) |
| `nc_matmul() missing value for required argument 'moving'` | Beta-1 call `nisa.nc_matmul(a, b)` | `nisa.nc_matmul(dst=acc, stationary=a, moving=b)` |
| `'mul' expected (int, int) ... got (object, int)` | Python arithmetic on a `fori_loop` index | `step=P` + `nl.ds(i, P)`, reshaped view `[i]`, or `.ap(scalar_offset=i)` |
| `unsupported expression` | comprehension / generator / lambda | explicit loops and helpers |
| `unbound variable '<name>'` | closure over an enclosing function's local | pass it as a kernel argument or use a module global |
| `Creating HBM tensor with init value is not supported.` | `nl.zeros(..., buffer=nl.shared_hbm)` | memset an SBUF tile and DMA it |
| `ValueError: Cannot call .result on operation torch.operator which has 0 results` | kernel returns nothing | allocate the output in `nl.shared_hbm` and return it |
| `exponential requires NeuronCore-v4 or newer` (likewise `activate2`, `priority`, `engine=gpsimd` activation) | gen4-only API | NeuronCore-v3 alternatives (§5) |
| `[NCC_IBIR322] Unspecified` | `perf_mode=double_row` with non-fp8 operands | double_row only for fp8 |

[dev:d, dev:c, dev:g, dev:h, dev:j, dev:k, dev:l, dev:m, dev:n, dev:o, dev:e]


---

## 10. Compatibility summary (installed 0.7.0b1 vs older public material)

Status: **S** supported, **R** supported with restriction, **D** deprecated (still works), **A** absent, **G4** NeuronCore-v4 only (fails on trn2). The full matrix with evidence links is in `COMPATIBILITY.md`.

| API / pattern | Status | Note |
|---|---|---|
| `@nki.jit`, `kernel(...)`, `kernel[2](...)` | S | bare = LNC 2 here [dev:b] |
| `kernel[1](...)` | S | 1 program, no runtime error here [dev:b] |
| `@nki.jit(lnc=..., platform_target=...)` | A | error; `mode=` ignored with a warning [pkg] |
| `nl.ndarray / zeros / ones / full / zeros_like / empty_like` | S | `zeros` etc. not in `shared_hbm` [dev:o] |
| `nl.load / nl.store / nl.load_transpose2d` | S | experimental wrappers of DMA; no `mask=` [dev:j] |
| `nisa.dma_copy`, `dma_compute` | S / R | `dma_compute`: `nl.add` only [dev:h] |
| `nisa.dma_transpose` | R | hwdge: 2-byte dtypes [dev:m] |
| `nisa.nc_matmul` | R | fp32 PSUM, K/M ≤ 128, N ≤ 512, no int8/e4m3fn [dev:c, dev:d] |
| `accumulate=None` (auto) | S | first write overwrites [dev:c] |
| `psum[...] += ...` implicit accumulation (Beta 1) | A | ISA calls need `dst=` and return an instruction handle; use `accumulate=` [pkg, dev:c] |
| `perf_mode=double_row` | R | fp8 `[128, 2, X]` operands [dev:n] |
| `nisa.nc_transpose` | R | TensorE PSUM dtype = input; VectorE ≤ 32×32 [dev:g] |
| `nl.matmul`, `nl.transpose` | R | PSUM result in the input dtype → fp32 inputs only on trn2 [dev:c] |
| `nisa.tensor_tensor / tensor_scalar / activation / tensor_reduce / tensor_copy / memset / iota / reciprocal / tensor_tensor_scan` | S/R | `tensor_scalar` tile operands fp32 [dev:g] |
| `nisa.select_reduce / range_select / affine_select / topk / bn_stats / max8 / ...` | S | [pkg]; `topk` ran [dev:n] |
| `nisa.core_barrier / sendrecv / sendrecv_compute` | S | LNC=2 [dev:h] |
| `range`, `nl.range` | S | unrolled [dev:f] |
| `nl.affine_range / sequential_range / static_range` | D | unrolled aliases of `range` [pkg, dev:f] |
| `nl.fori_loop`, `nl.while_loop` | R | no Python arithmetic on the index [dev:e] |
| `nl.dynamic_range` | D | works, no warning observed [dev:f]; removed under the tracer [doc] |
| `nl.no_reorder` | S | [dev:f] |
| `nl.arange`, `nl.mgrid`, `nl.par_dim`, `nl.spmd_dim`, `nl.nc`, `nl.atomic_rmw`, `mask=` | A | [pkg, dev:d] |
| `nisa.rsqrt`, `nisa.tensor_copy_dynamic_src/dst`, `nisa.activate` | A | [pkg] |
| `nl.tile_size.psum_fmax_bytes` | A | use `psum_bank_fmax_bytes` [pkg] |
| `nl.tile_size.psum_fmax`, `total_available_sbuf_size` | D | [pkg] |
| `nisa.exponential / activate2 / quantize_mx / nc_matmul_mx`, `t.indirect()`, DMA `priority` | G4 | [dev:n, pkg] |
| `torch_xla` / `xm.xla_device()` drivers, `torch_neuronx.trace` | A | this stack: native `"neuron"` device |

---

## 11. Stale patterns in public material: do not copy

| Seen in older docs / samples | Use in 0.7.0b1 |
|---|---|
| `import neuronxcc.nki`, `neuronxcc.nki.language` | `import nki`, `nki.language`, `nki.isa` |
| `nl.arange(128)[:, None]`, `nl.mgrid[0:128, 0:512]`, `mask=` | slices `x[0:128, 0:512]`, `nl.ds(start, size)`, partial-tile slices |
| `nl.par_dim(128)` in shapes | plain `128` (axis 0 is the partition axis) |
| `res = nisa.nc_matmul(a, b)` / `acc += ...` | `nisa.nc_matmul(dst=acc, stationary=a, moving=b, accumulate=...)` |
| `nisa.constants.engine.tensor` | `nisa.engine.tensor` |
| `kernel[nl.spmd_dim(n, nl.nc(2))](...)` | `kernel(...)` / `kernel[2](...)` with `nl.program_id(0)` |
| `@nki.jit(platform_target=...)`, `@nki.jit(mode=...)` | env `NEURON_PLATFORM_TARGET_OVERRIDE`; no mode |
| `xm.xla_device()`, `torch_neuronx.trace(...)` | `tensor.to("neuron")`, call the kernel directly |
| "tracer is the default in 0.7.0" | installed default is the parser (`custom`) [pkg] |
| "affine_range for accumulation, not sequential_range" | all `*_range` are unrolled aliases [pkg, dev:f] |
| "never use nl.load / nl.store" | they exist and work as DMA wrappers [dev:j] |
| `nisa.rsqrt(...)` | `nisa.activation(op=nl.rsqrt, ...)` |
| `tensor_reduce` without `negate=` | `negate=` exists and works [dev:g] |
| "LNC default is 1" (jit docstring) | here the unset LNC resolves to `NEURON_LOGICAL_NC_CONFIG` = 2 [pkg, dev:b] |
