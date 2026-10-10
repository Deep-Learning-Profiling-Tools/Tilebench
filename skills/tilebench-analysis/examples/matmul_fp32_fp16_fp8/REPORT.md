# matmul_fp32_fp16_fp8, fp32, B200, K=20480: why TileLang is 3.8x slower than cuTile and 2.5x slower than Triton

All paths are relative to `/scratch/arustagi/tb_skill_trials/main_r2/matmul_fp32_fp16_fp8/` unless they start with `tilebench/` (the checkout `/scratch/arustagi/tb_main_skill_wt`) or `tilelang/src` (the installed TileLang 0.1.11 package under `/scratch/arustagi/tilebench_pdf_env/lib/python3.11/site-packages/`). "SASS" means `bundle/<backend>/annotated_sass_r0a0.txt`. Nothing was run on a GPU; the only new artefacts are compiler outputs in `ir/`, `variant_tmem_tf32/` and `contrast_fp16/`.

## 1. Verdict

Case: C = A x B with M = N = 4096, K = 20480, fp32 inputs multiplied as TF32 (`bundle/brief.md`, "Benchmark row"; source `results/B200/csv/matmul_fp32_fp16_fp8_autotune.csv`).

| Backend | CSV latency (ms) | TileLang / backend |
|---|---:|---:|
| cuTile | 0.9145 | 3.83 |
| torch | 0.9466 | 3.70 |
| Triton | 1.4130 | 2.48 |
| TileLang | 3.4981 | 1 |

This is a divergence, far outside the 10% band.

Cause: TileLang 0.1.11 cannot lower a TF32 `T.gemm` to the Blackwell tcgen05 tensor-core instruction, so its kernel uses the legacy register MMA (`mma.sync m16n8k8`, SASS `HMMA.1688.F32.TF32`). It has to issue 335,544,320 of them, the tensor pipe is the binding resource, and it spends about 3.5x as much tensor-pipe time as Triton and cuTile need to do the same arithmetic with `UTCHMMA`.

## 2. Is this point representative?

Yes. The gap is a property of the fp32 path, not of this shape.

- **Scale** (`bundle/brief.md`, "Scale trend"): over all 20 shapes K = 1024..20480, TileLang/Triton is 2.48 to 2.87 and TileLang/cuTile is 3.27 to 4.05. TileLang's winner is the same at every shape (128x128x64, 128 threads, 2 stages). There is no winner flip to test against.
- **Dtype** (`bundle/brief.md`, "Dtype trend"): at the same shape the gap collapses for the dtypes where TileLang does use tensor memory: fp16 is 1.25 (vs Triton) and 1.40 (vs cuTile); fp8 is 1.40 and 1.70.
- **Hardware** (`bundle/brief.md`, "Other hardware"; `contrast_gh200/bundle/brief.md`): on GH200 TileLang takes 3.7091 ms, almost the same as its 3.4981 ms on B200. Triton goes from 1.7811 ms (GH200) to 1.4130 ms (B200) and cuTile from 2.566 ms to 0.9145 ms. TileLang did not gain from Blackwell; the peers did.

## 3. NCU diagnosis per backend

The three reports are the same problem: grid x tile is 32x32 x 128x128 (TileLang), 1024 x 128x128 (Triton) and 256 x 256x256 (cuTile), each 4096x4096 (`launch__grid_dim_*`, `launch__grid_size`; winners in `bundle/case.json`).

| Dimension | TileLang | Triton | cuTile |
|---|---|---|---|
| Launch: threads, registers, shared bytes | 128, 208, 132,096 | 128, 135, 99,376 | 256, 255, 230,764 |
| CTAs per SM and what limits it | 1 (shared memory) | 2 (shared memory) | 1 (shared memory and registers) |
| Occupancy, theoretical / achieved % | 6.25 / 6.25 | 12.5 / 11.48 | 12.5 / 10.89 |
| Waves per SM | 6.92 | 3.46 | 1.73 |
| `sm__cycles_active` min / avg / max | 5.51M / 6.36M / 6.45M | 1.63M / 2.05M / 2.26M | 0.68M / 1.18M / 1.38M |
| Warp instructions `smsp__inst_executed.sum` | 7.704e8 | 1.782e8 | 2.088e7 |
| Issue active per cycle | 0.2047 | 0.1452 | 0.0299 |
| Tensor pipe `sm__pipe_tensor_cycles_active...elapsed` % | 69.19 | 52.36 | 81.17 |
| tcgen05 pipe `sm__pipe_tc_cycles_active...elapsed` % | 0 | 59.93 | 81.61 |
| MMA instruction rate `sm__inst_executed_pipe_tensor_subpipe_hmma.avg.pct_of_peak_sustained_elapsed` % | 69.19 | 1.64 | 1.27 |
| DRAM throughput % | 7.73 | 18.48 | 18.49 |
| L2-to-L1 read bytes | 2.147e10 | 2.147e10 | 1.074e10 |
| PC samples (total; largest reason) | 229,602; `wait` 138,828 | 74,121; `long_scoreboard` 45,852 | 42,592; `long_scoreboard` 38,019 |
| Samples at mbarrier/sleep sites | 0 | 42,931 | 38,222 |

Metric names are those in `bundle/brief.md` ("NCU side-by-side") unless written out; the `sm__cycles_active` and `subpipe_hmma` rows are from `bundle/<backend>/ncu.json`.

**TileLang: bound by the legacy tensor (HMMA) pipe.**
- The hot loop is SASS 0x01660 to 0x03ae0: 585 instructions, each executed 1,306,624 times (4096 warps x 319 iterations). Per warp per iteration it holds 256 `HMMA.1688.F32.TF32`, 131 `LDS`, 32 `LDSM.16.M88.4`, 32 `LDGSTS.E.BYPASS.128`, 2 `BAR.SYNC` (counted from SASS by opcode).
- 61.0% of PC samples are on `HMMA`, with `wait` as the stall; `LDS` holds another 13.7% (`bundle/brief.md`, "PC samples by opcode"). `wait` is 138,828 of 229,602 samples. Typical run: 0x02160 to 0x02180, three consecutive `HMMA` with 519 to 559 `wait` samples each.
- The HMMA rate counter reads 69.19% of its sustained peak, and the tensor pipe is active 69.19% of elapsed cycles: when the pipe is active it issues HMMA at its maximum rate.
- Memory is not the limit: DRAM throughput 7.73%, `long_scoreboard` 20,649 samples (9%), `lg_throttle` 4,172.
- Shared loads conflict: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` is 1.678e8, equal to the 1.678e8 shared-load instructions (`smsp__sass_inst_executed_op_shared_ld.sum`). `short_scoreboard` holds 11,600 samples (5%), so this is secondary.
- NCU's rules "High Compute Throughput", "Wait Stalls" and "Shared Load Bank Conflicts" all agree with the SASS.

**Triton: one warp per CTA drives tcgen05 and waits for TMA tiles.**
- Hot loop SASS 0x013e0 to 0x019d0, 2,617,344 executions per instruction (4096 warps x 639 iterations). Per CTA per iteration: 4 `UTCHMMA` (0x016a0 to 0x01700, 654,336 executions each), 1 `UTCBAR` (0x01710), 2 `UTMALDG.2D` (0x01950, 0x01980), 4 `BAR.SYNC`.
- 56.5% of samples sit on the branch after the mbarrier `TRYWAIT` at 0x014b0 (41,686 `long_scoreboard`) and 15.1% on 0x01830 (`barrier` 11,145). 42,931 of 74,121 samples are at signal-wait sites.
- tcgen05 pipe 59.93% busy, DRAM 18.48%. Binding resource: synchronization on TMA arrival and block barriers around a tcgen05 pipe that is busy about 60% of the time.

**cuTile: tcgen05 pipe close to saturated.**
- Per CTA per iteration (81,920 executions = 256 CTAs x 320): 16 `UTCHMMA` (first at 0x1954003360), 4 `UTCBAR`, 12 `UTMALDG.2D`.
- 71.6% of samples are on two `NANOSLEEP.SYNCS` sites (0x1954004f50, 0x1954005640; 30,489 `long_scoreboard`). These are warps asleep until the epilogue (`impl_cutile.py:68`), not warps waiting for loads: only one warp per CTA issues the MMAs.
- tcgen05 pipe 81.61% busy; NCU's "Very High Utilization: TC" rule is consistent with SASS. Binding resource: the tcgen05 pipe.
- 256 CTAs on 148 SMs is 1.73 waves, so the second wave leaves SMs idle (`sm__cycles_active.min` is 57% of the average). That is a tail cost cuTile pays and still leads.

**Measured differences that could explain the gap**

1. MMA form: `HMMA.1688.F32.TF32` (TileLang) versus `UTCHMMA` (Triton, cuTile); tcgen05 pipe 0 versus 59.93 and 81.61.
2. Instruction count: 36.9x cuTile and 4.3x Triton (`smsp__inst_executed.sum`).
3. Where warps wait: fixed-latency `wait` on HMMA (TileLang) versus mbarrier waits (peers).
4. Operand staging: 1.717e8 `LDS` + 4.19e7 `LDSM` + 4.19e7 `LDGSTS` (TileLang) versus 1.31e6 and 9.83e5 `UTMALDG` (Triton, cuTile).
5. Shared-load bank conflicts: 1.678e8 versus 6,422 and 5,167.
6. Residency: 1 CTA per SM for TileLang, 2 for Triton.
7. Accumulator location: 208 registers per thread (TileLang) versus 135 (Triton).

## 4. Selected configurations and work ledger

| | TileLang | Triton | cuTile |
|---|---|---|---|
| Winner (`bundle/case.json`) | M128 N128 K64, 128 threads, 2 stages | M128 N128 K32, 4 warps, 3 stages | 256x256x64, occupancy hint 8 |
| Check against the capture | block 128; loop bound 319 at SASS 0x01e10 (`ISETP.NE ... 0x13f`); shared 132,096 | block 128; 639 loop trips; shared 99,376 | name ends `I20480_I256_I256_I64_I8`; block 256 |
| K iterations per CTA | 320 | 640 | 320 |
| MMA instructions executed | 335,544,320 `HMMA` | 2,621,440 `UTCHMMA` | 1,310,720 `UTCHMMA` |
| Multiply-adds per MMA instruction | 1,024 (m16n8k8) | 131,072 (derived) | 262,144 (derived) |
| MMA instructions per CTA per iteration | 1,024 (256 per warp x 4 warps) | 4 | 16 |
| Warp instructions per 1000 units of problem size | 1.121 | 0.2593 | 0.0304 |

The total is M x N x K = 3.436e11 multiply-adds for all three. 335,544,320 x 1,024 equals that exactly, so TileLang's HMMA count is fixed by the problem and the m16n8k8 shape, not by the tile configuration. The Triton and cuTile per-instruction figures are the total divided by their UTCHMMA counts. MMA counts are `bundle/brief.md` "Dynamic warp-level opcode executions" (TileLang, cuTile) and per-PC counts in the Triton SASS (4 x 654,336 + 4 x 1,024).

## 5. Lowering decision matrix (differing rows)

Intermediate code: `ir/tilelang_fp32.cu` (generated CUDA), `ir/triton/matmul_kernel.ttgir` and `.ptx`, `ir/cutile_fp32.tileir`. All three were checked against the capture: TileLang's line numbers match the `tvm_kernels.cu:<line>` column of the SASS (line 64 is the `LDS` group, 69/70 the `HMMA`), the loop bound and per-iteration opcode counts match; Triton's 4 MMAs and 2 TMA copies per iteration match; cuTile's constants match the mangled name.

| # | Decision | TileLang | Triton | cuTile |
|---|---|---|---|---|
| 7 | Tensor-core path, accumulator | `tl::mma_sync<kTensorFloat32, kTensorFloat32, kFloat32, 16, 8, 8, ...>` (`.cu`:69-70); accumulator `float acc[128]` per thread (`.cu`:25) | `ttng.tmem_alloc` 128x128 f32 (`.ttgir`:47) and `ttng.tc_gen5_mma` (`.ttgir`:99); PTX `tcgen05.mma.cta_group::1.kind::tf32` (`.ptx`:408-423) | one `tile_mma` on `Tile[tfloat32,(256,64)]` x `(64,256)` (`.tileir`:89); `UTCHMMA` and `LDTM` in SASS (back-end choice) |
| 6 | Operand staging | swizzled shared tiles read back into registers: 4 `ptx_ldmatrix_x4` per k-step for A (`.cu`:60), 16 scalar reads per k-step for B (`.cu`:64) | `nvmma_shared` buffers handed to the MMA by descriptor (`.ttgir`:54-55); no register read-back | not visible in Tile IR; SASS has 2,048 shared loads in total |
| 8 | Async and sync per iteration | 32 `tl::cp_async_gs<16>` per thread (`.cu`:45-51), `cp_async_wait<1>`, 2 `__syncthreads()` (`.cu`:43, 54); every thread copies and computes | 2 `ttng.async_tma_copy_global_to_local` (`.ttgir`:114, 116), mbarrier waits (`.ttgir`:94, 100) | `tile_load` with `allow_tma=None` (`.tileir`:84, 87); `UTMALDG` and `SYNCS`/`NANOSLEEP` in SASS |
| 5 | Loop body | 8 k-steps x 32 `mma_sync`, fully unrolled by nvcc into 256 `HMMA` | 4 MMAs per iteration | 16 MMAs per iteration (SASS) |
| 9 | Resource footprint | `__launch_bounds__(128, 1)` (`.cu`:21); 2 stages x (128x64 + 64x128) x 4 B = 131,072 B shared; 208 registers | `"shared": 98352`, `"tmem_size": 128` (`ir/triton/matmul_kernel.json`); 135 registers | 230,764 B shared, 255 registers; the hint `occupancy=8` is inert (1 CTA per SM) |
| 10 | Data footprint | 128x128 tiles: 2.147e10 B from L2 | same | 256x256 tiles: 1.074e10 B |
| 12 | Tuning surface | no config changes row 7 | larger tiles are in the space and lost | larger tile won |

Rows 1 to 4 and 11 do not differ in a way that matters: all three use whole-tile loads with constant tile sizes, no bounds guards in the loop, and one accumulation pass.

## 6. Mechanisms

### TileLang

**M1. TF32 GEMM is lowered to legacy `mma.sync`, not tcgen05 (primary).**

- Source: `tilebench/benchmarks/operators/matmul_fp32_fp16_fp8/impl_tilelang.py:94` sets `use_tmem = dtype != T.tfloat32`, so for fp32 line 106 calls `T.gemm(a_tile, b_tile, acc)` with a register fragment (`T.alloc_fragment`, line 93).
- Why the author wrote that: TileLang's instruction selection refuses the alternative. `tilelang/src/cuda/op/gemm.cc:77-85` (`AllowTcgen5Mma`) defers to `GetTCGEN5MMAMeta`, and `tilelang/src/op/tcgen5_meta.h` has branches only for fp16/bf16 (line 42), fp8/fp6/fp4 (line 79) and int8 (line 126); the file contains no `tfloat32` case, so the query fails and `SelectInst` falls through to `return kCudaMMA` (`gemm.cc:280-286`). Checked by compiling an edited copy with `use_tmem = True` (`variant_tmem_tf32/impl_tilelang.py`) at 128x128x64 and 256x256x64: both stop with `AssertionError: local_buf acc_tmem must be a fragment, but got shared.tmem` from `mma_macro_generator.py:822`, that is, the MMA emitter was chosen even with a tensor-memory accumulator. The instruction template itself exists (`tilelang/src/tl_templates/cuda/instruction/tcgen05mma.h:98-108`, `tcgen05.mma.cta_group::1.kind::tf32`); only the selection table lacks the dtype.
- Generated code (`ir/tilelang_fp32.cu`:67-70):
  ```c
  for (int i_7 = 0; i_7 < 4; ++i_7) { for (int j_1 = 0; j_1 < 4; ++j_1) {
    tl::mma_sync<tl::DataType::kTensorFloat32, tl::DataType::kTensorFloat32, tl::DataType::kFloat32, 16, 8, 8, false, true>(acc + ((i_7 * 32) + (j_1 * 8)), A_local + (i_7 * 4), B_local + (j_1 * 4));
    tl::mma_sync<... 16, 8, 8, false, true>(acc + (((i_7 * 32) + (j_1 * 8)) + 4), A_local + (i_7 * 4), B_local + ((j_1 * 4) + 2));
  ```
  inside `for (int ki = 0; ki < 8; ++ki)` (line 58): 32 calls x 8 = 256 per thread per K iteration. The instruction is `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` (`bundle/tilelang/embedded_source/r0a0/mma_sm80.hpp:305`).
- SASS: 256 `HMMA.1688.F32.TF32` from 0x01fd0 up to the loop branch at 0x03ae0, 1,306,624 executions each; total 335,544,320.
- Counters and samples: `sm__pipe_tc_cycles_active` 0 against 59.93 (Triton) and 81.61 (cuTile); HMMA rate 69.19% of sustained peak; 61.0% of samples on `HMMA` with `wait`; the excess over cuTile is +187,010 samples, of which +138,274 are `wait` and +46,504 have no stall recorded (`bundle/brief.md`, "Where the extra time sits").
- What it costs: the same 3.436e11 multiply-adds take 0.6919 x 3,630 us = 2,512 us of tensor-pipe time in TileLang, against 0.5236 x 1,367 us = 716 us in Triton and 0.8117 x 868 us = 704 us in cuTile (tensor-pipe percentage x `gpu__time_duration.sum`). That is 3.5x the pipe time.
- A floor, derived from NCU's own normalisation and therefore an inference: 335,544,320 HMMA over 969,921,506 SM-cycles (`sm__cycles_elapsed.sum`) is 0.346 per SM-cycle, which NCU reports as 69.19% of peak, so the peak is 0.5 HMMA per SM-cycle. The same peak reproduces the Triton and cuTile readings of this counter from their UTCHMMA counts (1.64% and 1.27%). At 100% of that rate on 148 SMs the HMMA alone need 4.53M cycles, 2.51 ms at the 1.805e9 cycles/s of this capture. That is above Triton's and cuTile's whole measured latency, so no choice of tile, stage count or occupancy closes the gap while the path is `mma.sync`.
- What the design buys: nothing on Blackwell. Register accumulators also cost 128 floats per thread (208 registers).

**M2. The same threads stage every operand through registers (secondary; fills the 31% of time the tensor pipe is idle).**

- Source: `T.copy(a[...], a_tile)` and `T.copy(b[...], b_tile)` (`impl_tilelang.py:101-102`) inside `T.Pipelined`, with warp specialization disabled.
- Generated code: 16 + 16 `tl::cp_async_gs<16>` per thread (`.cu`:45-51), `tl::cp_async_wait<1>()` and two `__syncthreads()` per iteration (`.cu`:43, 53-54); then per k-step 4 `tl::ptx_ldmatrix_x4` for A (`.cu`:60) and 16 scalar reads `B_local[(i_6 * 4) + j] = ((tfloat32_t*)b_tile)[...]` for B (`.cu`:64).
- SASS: 32 `LDGSTS.E.BYPASS.128` (0x01840 to 0x01e80), `LDGDEPBAR`/`DEPBAR.LE SB0, 0x1` (0x01e90, 0x01ea0), `BAR.SYNC` (0x01670, 0x01eb0), 32 `LDSM.16.M88.4`, 131 `LDS` (line 64).
- The B reads conflict two ways. In `.cu`:64 the only terms that vary across a warp and are not multiples of 32 elements are `8 * ((j>>1) + bit1(tid)) & 1`, `4 * (bit4(tid) + bit0(tid)) & 1` and `(tid & 15) >> 2`, which give at most 16 distinct banks for 32 lanes; lanes that differ in bit 0 and bit 4 together land in the same bank at addresses 32 elements apart. The counter agrees: 1.678e8 conflicts for 1.678e8 shared-load instructions.
- Samples: `LDS` 13.7%, `IADD3` (address arithmetic for the copies, carrying `long_scoreboard`) 7.0%, `LDSM` 5.3%, `LDGSTS` 2.9%; `short_scoreboard` 11,600 and `barrier` 3,534 in total.
- Cost: this is why the HMMA pipe runs at 69% and not higher. It is bounded: removing all of it cannot go below the M1 floor.

**Differences that do not matter here**

- One CTA per SM (shared-memory limit). The fp16 TileLang kernel has the same 6.25% occupancy (`contrast_fp16/bundle/brief.md`) and is within 1.40x of cuTile, and cuTile itself runs one CTA per SM.
- DRAM traffic: TileLang reads 2.03e9 B against Triton's 1.82e9 B at 7.73% DRAM throughput; not a bound.

### Triton (why it is 1.55x behind cuTile, which sets the 2.5x versus 3.8x split)

- `tl.dot(a, b.T, accumulator, input_precision="tf32")` (`impl_triton.py:76`) becomes `ttng.tc_gen5_mma` on a tensor-memory accumulator; loads through `TensorDescriptor` become TMA copies. This is the same tensor path as cuTile.
- Its winner uses a 128x128x32 tile, so it runs 640 iterations per CTA and moves 2.147e10 B from L2 against cuTile's 1.074e10 B. Each iteration waits on the TMA mbarrier (SASS 0x014b0, 56.5% of samples) and passes four block barriers (0x01830, 15.1%). Its tcgen05 pipe is busy 59.93% against cuTile's 81.61%.

### cuTile

- `ct.mma` on `tfloat32` tiles (`impl_cutile.py:59, 66`) is one `tile_mma`; the closed back end emits `UTCHMMA` with tensor memory, TMA loads and one driving warp per CTA (read from SASS). With 256x256x64 tiles it keeps the tcgen05 pipe 81.61% busy. No mechanism to fix.

## 7. Why the net gap is this size

NCU durations are 3,630 us (TileLang), 1,367 us (Triton) and 868 us (cuTile); they track the CSV (3,498, 1,413, 914.5 us) and are used here only to apportion.

- Against cuTile the NCU gap is 2,762 us. Tensor-pipe busy time accounts for 2,512 - 704 = 1,808 us of it (65%): the cost of the HMMA form (M1). The rest, about 955 us, is the difference in time the tensor pipe is idle (1,118 us in TileLang, 163 us in cuTile): operand staging and synchronization by the compute warps (M2).
- Against Triton the NCU gap is 2,263 us: 1,796 us from pipe-busy time (79%) and 467 us from idle time. Triton's own pipe is idle 651 us waiting on TMA and barriers, which is why TileLang is 2.5x behind Triton but 3.8x behind cuTile.
- What the resident warps are doing: in TileLang all four warps of the single CTA per SM are in the HMMA sequence, stalled on `wait` between consecutive HMMA. In the peers one warp per CTA issues a handful of `UTCHMMA` and then waits on an mbarrier while the tensor core works.

**Trying to break it.** The strongest alternative is that TileLang is latency- or occupancy-bound: one CTA per SM, lane-issued `cp_async` copies and a barrier per iteration. Three measurements separate the two:

1. Stall reasons: `wait` 138,828 samples against `long_scoreboard` 20,649, `lg_throttle` 4,172 and `barrier` 3,534. A load-latency bound would put the samples on `long_scoreboard`.
2. The fp16 sibling keeps the same lane-issued `cp_async_gs<16>` copies, `__syncthreads()` and 6.25% occupancy (`contrast_fp16/tilelang_fp16.cu`:63-79) but calls `tl::tcgen05mma_ss<tl::DataType::kFloat16, false>` (line 91). Its gap is 1.40x to cuTile and 1.25x to Triton. Same staging, different MMA form, and most of the gap is gone.
3. GH200 (`contrast_gh200/bundle/brief.md`): TileLang executes the identical 335,544,320 `HMMA` there and takes 3.7091 ms, with `wait` again the largest reason (220,041 of 411,898 samples) and 50.7% of samples on `HMMA`. A different memory system and SM count, the same HMMA count, the same latency to within 6%.

The explanation survives all three.

## 8. Classification

- TileLang versus both peers: **capability gap in the compiler** (no tcgen05 lowering for TF32 in TileLang 0.1.11), with a secondary **codegen/lowering** cost (register staging of operands with a two-way bank conflict on B).
- Not an autotune choice: the HMMA count is independent of every tuned parameter, and the winner is the same at all 20 shapes.
- Not an author error: the source guard at `impl_tilelang.py:94` is the only form that compiles.
- Not a capture artifact: NCU and CSV ratios agree (4.18 and 3.83 against cuTile; 2.65 and 2.48 against Triton).

## 9. Ranked directions

**TileLang**

1. Compiler: add a `tfloat32` case to `GetTCGEN5MMAMeta` (`tilelang/src/op/tcgen5_meta.h`) so that `T.gemm` with a `T.alloc_tmem` accumulator selects `cuda.tcgen05` and emits the existing `kind::tf32` template. The operator can then drop the guard at `impl_tilelang.py:94` and use the fp16 structure for fp32. Confirming measurement: a new capture showing `UTCHMMA` in SASS and `sm__pipe_tc_cycles_active` above zero, with the CSV ratio to cuTile compared against the fp16 ratio of 1.40.
2. There is no source-level change in the DSL today that reaches tcgen05 for TF32; the variant in `variant_tmem_tf32/` is rejected at compile time.
3. Lower priority, and only worth doing after 1: move operand copies off the compute threads or onto TMA, and read the B operand without the two-way bank conflict. Confirming measurement: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and the tensor-pipe percentage in a new capture.

**Triton**: nothing indicated by this case. The 256x256x64 tile that cuTile wins with is in Triton's search space (`impl_triton.py:96-101`) and was not selected.

**cuTile**: nothing indicated.

## 10. Limits

- No TileLang TF32 tcgen05 kernel exists to measure. The size of the recoverable gap is inferred from the fp16 sibling, whose B200 capture is reduced (no SASS or PC samples); its tcgen05 path is taken from the generated CUDA, not from captured SASS.
- The 2.51 ms HMMA floor rests on NCU's `pct_of_peak_sustained` normalisation for the HMMA sub-pipe, cross-checked only against the three reports in this bundle.
- The Triton `ttgir` was served from Triton's compile cache: its `#loc` lines name a different checkout path. The line numbers agree with the checkout's `impl_triton.py` and with the report, and the structure matches the SASS, but the file was not freshly compiled from the checkout.
- `launch__shared_mem_per_block` is 1,024 B larger than the compiled size for both TileLang (132,096 against 131,072) and Triton (99,376 against 98,352). The cause was not determined; it does not change the residency of either kernel.
