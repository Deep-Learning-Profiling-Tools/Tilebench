# weight_dequant fp16 on B200: why cuTile is 2.9x slower than TileLang and Triton

Case: `weight_dequant`, dtype `fp16`, `M = N = 10240`, `TILE_SIZE = 128` (104,857,600 elements, scale table 80 x 80), NVIDIA B200.
All paths below are relative to `/scratch/arustagi/tb_skill_trials/main_r2/weight_dequant/` unless they start with `tilebench/` or `results/` (TileBench checkout `/scratch/arustagi/tb_main_skill_wt`).
Evidence is saved benchmark CSVs, autotune winner logs, saved Nsight Compute reports and compiler intermediate code generated without a GPU. Nothing was run on a GPU.

## 1. Verdict

| Backend | CSV latency (ms) | vs TileLang | Winner config |
|---|---:|---:|---|
| TileLang | 0.0635 | 1.00x | `BLOCK_SIZE=2048, threads=128` |
| Triton | 0.0679 | 1.07x | `BLOCK_SIZE=2048, num_warps=4` |
| cuTile | 0.1864 | 2.94x (2.75x vs Triton) | `tile=4096, occupancy=4` |

Source: `results/B200/csv/weight_dequant_autotune.csv`, row `M=10240, fp16` (`bundle/brief.md:8-15`, `bundle/case.json`).

TileLang and Triton are at parity (7%, inside the 10% band). cuTile diverges.

Cause: the cuTile kernel computes the scale-table index (two signed floor divisions, one modulo with sign fix-up, two bounds tests, a 64-bit address) and issues one 16-bit gather load separately for each of the 4096 elements of its tile, 1300 warp instructions per warp against 66 (TileLang) and 105 (Triton). That makes it bound by instruction issue on the integer (ALU) pipe, while the two peers are bound by DRAM bandwidth and fold the same index to one or two divisions per thread.

## 2. Is this point representative?

Yes for the direction, and it is the large end of a steady trend, not an anomaly.

- **Scale** (`bundle/brief.md:23-44`; per-element figures computed from the CSV). cuTile costs about 1.8 ps per element at every non-power-of-two shape from M=5120 up (1.86 at 5120, 1.82 at 7168, 1.79 at 9216, 1.78 at 10240). TileLang falls from 0.74 to 0.61 ps per element over the same range. The ratio cuTile/TileLang therefore rises from about 2.0-2.8 at mid sizes to 2.94 here. The brief flags this shape as an outlier against the median (0.34 vs 0.48 for tilelang/cutile, `bundle/brief.md:311`); the cause is the peers getting faster per element, not cuTile getting slower.
- **Winner flips do not move the gap.** cuTile's winner changes between `tile` 1024/2048/4096 and `occupancy` 4/8/16 across shapes (`bundle/brief.md:29-44`) while its per-element time stays near 1.8 ps. The gap is not a property of one config.
- **Power-of-two shapes are cheaper for cuTile only.** M=4096 costs 1.48 ps/element and M=8192 costs 1.37, against 2.01/1.91 and 1.84/1.78 at their neighbours (3584/4608 and 7680/8704; CSV). `N` is a compile-time constant in all three kernels, so at these shapes the division by `N` is a shift. This is consistent with section 6 but is an inference: no profile exists for those shapes.
- **Dtype.** bf16 is identical to fp16 (0.1865 / 0.0641 / 0.0677 ms). At fp32 the gap shrinks to 1.54x: cuTile 0.1909 ms, TileLang 0.1236, Triton 0.1225 (`bundle/brief.md:50-52`). cuTile's time barely changes when the bytes double; the peers' time doubles. Section 6 uses this as the separating test.
- **Hardware.** GH200: cuTile 0.1717 ms, TileLang 0.1185, Triton 0.1201 (1.45x; `bundle/brief.md:58`). Same direction, smaller gap. No GH200 profile was analysed.

## 3. NCU diagnosis per backend

NCU durations (66.6 / 69.7 / 188.6 us, `gpu__time_duration.sum`) agree with the CSV to within 5% and are used only for diagnosis. All three reports read the same data: `dram__bytes_read.sum` is 2.097e8 / 2.098e8 / 2.098e8 B. Each is a single kernel.

| Dimension (metric) | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| Grid x block (`launch__grid_size`, `launch__block_size`) | 51200 x 128 | 51200 x 128 | 25600 x 128 |
| Registers/thread (`launch__registers_per_thread`) | 26 | 32 | 124 |
| Shared memory/block, B (`launch__shared_mem_per_block`) | 1024 | 1024 | 17420 |
| CTAs/SM limit (`launch__occupancy_limit_registers`) | 16 | 16 | 4 |
| Theoretical / achieved occupancy % (`sm__maximum_warps_per_active_cycle_pct`, `sm__warps_active.avg.pct_of_peak_sustained_active`) | 100 / 83.4 | 100 / 85.0 | 25 / 23.4 |
| Warp instructions (`smsp__inst_executed.sum`) | 1.352e7 | 2.150e7 | 1.331e8 |
| Issue active per cycle (`smsp__issue_active.avg.per_cycle_active`) | 0.194 | 0.288 | 0.638 |
| Eligible warps per cycle (`smsp__warps_eligible.avg.per_cycle_active`) | 0.26 | 0.81 | 1.44 |
| ALU pipe % (`sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active`) | 10.6 | 30.3 | 89.5 |
| DRAM throughput % (`gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`) | 82.6 | 80.2 | 30.0 |
| SMSP active cycles (`smsp__cycles_active.avg`) | 117,732 | 126,030 | 352,625 |
| Global load requests (`l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`) | 819,200 | 3,686,400 | 3,686,400 |
| L1 miss sectors (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum`) | 6.589e6 | 6.586e6 | 6.593e6 |
| Shared loads / stores (`smsp__sass_inst_executed_op_shared_{ld,st}.sum`) | 0 / 0 | 0 / 0 | 1,228,800 / 1,228,800 |
| PC samples (`smsp__pcsamp_sample_count`) | 4,317 | 4,556 | 12,735 |

Values from `bundle/brief.md:65-124` and `bundle/<backend>/ncu.json`. Block balance: `sm__cycles_active` min/max is within 3% of the mean in all three (cuTile 347,959 / 357,680), and waves per SM are 21.6 / 21.6 / 43.2, so there is no tail effect. `smsp__thread_inst_executed_per_inst_executed.ratio` is 32.0 in all three (no divergence). Tensor pipes are zero everywhere, as expected.

**TileLang: DRAM-bandwidth bound.** DRAM at 82.6% of peak, issue slots used 19% of the time. 3,880 of 4,317 samples (90%) are `long_scoreboard`, on the first consumers of the loads: `0x00190 LEA` (1,998 samples, consumer of the scale load at `0x00130`), `0x001a0 HADD2.F32` (1,333, consumer of the X load at `0x00140`), `0x002b0 HADD2.F32` (512, consumer of the second X load at `0x00160`) (`bundle/tilelang/annotated_sass_r0a0.txt`). All four loads are issued before the first wait.

**Triton: DRAM-bandwidth bound.** DRAM at 80.2%. 2,545 of 4,556 samples are `long_scoreboard` on `0x00570 PRMT` (1,889) and `0x00530 PRMT` (560), the first consumers of the scale loads; 633 are `lg_throttle` on the 16 `LDG.E.U16` scale loads at `0x00300`-`0x00510` (`bundle/triton/annotated_sass_r0a0.txt`). The extra narrow loads cost 59% more instructions than TileLang but both kernels sit above 80% DRAM, so the difference is hidden: 7% in latency.

**cuTile: instruction-issue bound on the ALU pipe.** ALU pipe at 89.5% of peak, issue slots used 64% of the time, more than one warp eligible every cycle, DRAM at 30%. Samples by stall reason (`bundle/brief.md:272`): `not_selected` 2,631, `math_pipe_throttle` 2,592, `long_scoreboard` 2,276, `wait` 1,457, `dispatch_stall` 656, `barrier` 426, none recorded 2,205. The five execution-side classes hold 9,541 of 12,735 samples (75%). Hot PCs (`bundle/cutile/annotated_sass_r0a0.txt`):

- `0x02e30 PRMT` 1,798 samples and `0x02eb0 PRMT` 376, `long_scoreboard`: the first consumers of the four `LDG.E.128` X loads at `0x02d70`-`0x02e20`. This is the only memory wait in the kernel.
- `0x00040 R2UR` 1,292 samples, `not_selected` 626 / `math_pipe_throttle` 603: the kernel's fifth instruction. Warps queue here for an issue slot; it marks a saturated scheduler, not a slow instruction.
- `0x002f0`, `0x004b0`, `0x00580` `LEA.HI.SX32` (811 / 250 / 234) and `0x00920 ISETP.NE.U32.AND` (509), all `not_selected` / `math_pipe_throttle`, inside the index arithmetic of source lines 31-34.
- `0x03bd0 LDG.E.U16` 339 samples, `barrier` 335: the first instruction after `BAR.SYNC` at `0x03b20`.

**Issue-slot floor.** `smsp__inst_executed.sum` / (148 SMs x 4 schedulers, `device__attribute_multiprocessor_count` x `device__attribute_num_schedulers_per_multiprocessor`) = 133,120,000 / 592 = 224,865 cycles for cuTile. That is already 1.9x TileLang's whole measured run (117,732 cycles) and 1.8x Triton's (126,030). No change to occupancy or memory behaviour can bring this kernel to the peers' time; it executes too much.

**NCU rules checked against SASS.** "Very High Utilization: ALU 87.5%" (cuTile) is confirmed. "Low Utilization" (TileLang) is the usual misfire on a DRAM-bound kernel. "L1TEX Global Load Access Pattern, 22.0 of 32 bytes per sector" (cuTile, and 22.7 for Triton) is real but harmless: it comes from the 16-bit scale loads, whose sectors hit L1 (`l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct` 32.9 / 33.0) and add no DRAM traffic (L1 miss sectors are equal across backends).

**Where the gap sits in time** (`bundle/brief.md:178-184`). cuTile has 8,418 more samples than TileLang. `not_selected` +2,618, `math_pipe_throttle` +2,580, none recorded +2,131, `wait` +1,390, `dispatch_stall` +648 sum to +9,367; `long_scoreboard` is -1,604. Against Triton the same five classes sum to +9,070 and `long_scoreboard` is -269. The whole excess is spent executing or queueing to execute. cuTile waits on memory less than either peer.

## 4. Selected configurations and work ledger

Join check: 51200 x 2048 = 25600 x 4096 = 104,857,600 = 10240 x 10240. The cuTile kernel name in the report ends `I10240_I128_I4096` (N, TILE_SIZE, tile), matching the winner.

| | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| Elements per CTA | 2048 | 2048 | 4096 |
| Elements per thread | 16 | 16 | 32 |
| Warp instructions per warp (`bundle/brief.md:130-134`) | 66 | 105 | 1300 |
| Warp instructions per 1000 elements (`bundle/brief.md:145`) | 128.9 | 205.1 | 1270 |
| X loads per thread | 2 x `LDG.E.128` | 2 x `LDG.E.128` | 4 x `LDG.E.128` |
| Scale loads per thread (`LDG.E.U16`) | 2 | 16 | 32 |
| Scale loads per element | 1/8 | 1 | 1 |
| Divide-by-10240 sequences per thread (`IMAD.HI ..., 0x66666667`) | 1 | 2 | 32 |
| Bounds compares on the scale index per thread (`ISETP.GE.U32` vs `UR10`/`UR11`) | 0 | 0 | 64 |
| Shared stores + loads per thread | 0 | 0 | 12 + 12 |
| `BAR.SYNC` per thread | 0 | 0 | 2 |
| Output stores per thread | 2 x `STG.E.128` | 2 x `STG.E.128` | 4 x `STG.E.128` |

Per-thread counts are static instruction counts from each `annotated_sass_r0a0.txt`; every counted instruction executes once per warp (all three kernels are straight-line: 204,800 executions per instruction for TileLang and Triton, 102,400 for cuTile). Per element, cuTile executes 9.85x TileLang's instructions and 6.19x Triton's.

cuTile's 1300 instructions per warp, by source line of `tilebench/benchmarks/operators/weight_dequant/impl_cutile.py` (executions / 102,400, from the annotated SASS; totals in `bundle/brief.md:295`):

| Source line | Instructions per warp | Share | PC samples |
|---|---:|---:|---:|
| 31-34 `row`, `col`, `s_row`, `s_col` | 707 | 54.4% | 5,477 (43.0%) |
| 37 `ct.gather` | 287 | 22.1% | 1,257 (9.9%) |
| 28 `ct.load` of X, incl. shared-memory re-layout | 110 | 8.5% | 2,982 (23.4%) |
| 41 `ct.store`, incl. shared-memory re-layout | 85 | 6.5% | 901 (7.1%) |
| 39 multiply | 48 | 3.7% | 340 (2.7%) |
| 25 `offsets` | 34 | 2.6% | 201 (1.6%) |
| 21-24 entry, block id | 29 | 2.2% | 1,577 (12.4%) |

Index arithmetic and gather together are 994 of 1300 instructions (76.5%). The remaining 306 per warp (9.6 per element) total 31.3 million warp instructions, 1.5x Triton's 21.5 million and 2.3x TileLang's 13.5 million, against 6.2x and 9.85x for the whole kernel. This is arithmetic on the counts, not a predicted latency.

## 5. Lowering decision matrix (differing rows)

Intermediate code: `ir/tilelang_fp16.cu` and `ir/tilelang_fp16.lowered.tir` (TileLang 0.1.11, `sm_100`), `ir/triton_ir/dequant_kernel.{ttgir,ptx}` (Triton 3.6.0, `cuda:100`), `ir/cutile_fp16.tileir` (cuda-tile 1.3.0, `sm_100`). Each matches its capture: TileLang's `tvm_kernels.cu` line numbers in the SASS (26 X load, 27 scale load, 45 store) are the same lines of the regenerated file; Triton's PTX has 2 `ld.global.v4.b32` + 16 `ld.global.b16` + 2 `st.global.v4.b32` (`dequant_kernel.ptx:83-205`), the same as the SASS; the cuTile symbol printed by `cutile_ir.py` is identical to the kernel name in the report.

| # | Decision | TileLang | Triton | cuTile |
|---|---|---|---|---|
| 1 | Element ownership | 8 consecutive fp16 per thread, two runs 1024 apart: `X + blockIdx.x*2048 + i*1024 + threadIdx.x*8` (`tilelang_fp16.cu:26`) | Same: `sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4]` (`dequant_kernel.ttgir:1`), one layout for every tensor | Not in Tile IR. SASS shows two ownerships in one kernel: X load and output store take 8 consecutive fp16 per thread at 0x800-byte steps (`0x02d70`-`0x02e20`, `0x050b0`-`0x05120`); the index tile takes one element per thread, 128 apart (`VIADD Rn, R11, 0x80 ... 0xf80` at `0x000d0`-`0x007b0`, `R11 = bid*4096 + tid`) |
| 2 | Access width of the scale read | One scalar read per 8-element vector, broadcast: `S_local_cast_2[0:8] = T.Broadcast(S_1[...], 8)` (`tilelang_fp16.lowered.tir:21`) | One `ld.global.b16` per element; 8 in a row read the same address (`dequant_kernel.ptx:104-164`) | One pointer load per element: `load_pointer(pointer=$177, mask=$173, ...)` on `Tile[float16,(4096)]` (`cutile_fp16.tileir:73`) |
| 3 | Bounds handling | None. The source guard `idx < n_elements` is proved away; no guard in the generated code | One mask per thread, `arith.cmpi slt, %offsets_5, 104857600` (`ttgir:18`), one `ISETP` at `0x00270` | Per element on the gather: two `raw_cmp ... fn="lt"` and an `and_` on 4096-wide tiles (`tileir:55,64,65`), 64 `ISETP.GE.U32` per thread. Plus zero-padding tests on the X tile (`tileir:29`, `0x02d30`-`0x02de0`) and on the store (`0x04f70`-`0x04fe0`) |
| 4 | Specialization | Scale-table width is the literal 80 (`cu:27`) | `S_COLS` is a `tl.constexpr`, 80 (`ttgir:9`) | `N`, `TILE_SIZE`, `TILE` are `typed_const` (`tileir:20-22`); the scale table's extents and row stride are runtime arguments (`tileir:14-16`, read at `0x02f90 LDCU.128` and `0x030b0 LDCU`), so each address is a 64-bit `IMAD.WIDE Rn, Rm, UR5, ...` |
| 5/11 | Index formulation | Folded by the compiler to one expression per vector with no per-element division: `S[blockIdx.x/640*80 + blockIdx.x%5*16 + i*8 + (threadIdx.x >> 4)]` (`cu:27`). One `IMAD.HI ..., 0x66666667` per thread (`0x00060`) | `arith.divsi`/`remsi` on the 2048-wide tensor in `ttgir:19-22`, but the PTX has two `mul.hi.s32 ..., 1717986919` (`ptx:47,53`): one division per 8-element run | Four per-element ops on `Tile[int32,(4096)]`: `fn="floordiv"` by N (`tileir:33`), `fn="c_mod"` by N plus a sign fix-up chain `raw_cmp`/`xor`/`ne`/`and_`/`add`/`raw_where` (`tileir:36-44`), two `fn="floordiv"` by TILE_SIZE (`tileir:47,50`). 32 `IMAD.HI ..., 0x66666667` per thread: nothing is folded |
| 6 | Staging through shared memory | None | None (`"shared": 0` in `dequant_kernel.json`, 0 `convert_layout`) | Two round trips, a back-end decision read from SASS: X tile `STS.128` x4 (`0x03250`-`0x033f0`), `BAR.SYNC` (`0x03b20`), `LDS.64` x8 (`0x047a0`-`0x048d0`); result `STS.64` x8 (`0x04b20`-`0x04da0`), `BAR.SYNC` (`0x04e20`), `LDS.128` x4 (`0x04e80`-`0x04f30`) |
| 7 | Arithmetic path | Widen to fp32, multiply, narrow (`cu:33-42`; 20 `HADD2.F32`, 8 `FMUL2`, 8 `F2FP` per thread), as the source's `T.cast(..., "float32")` asks | Packed fp16 multiply (`arith.mulf` on f16, `ttgir:31`; 4 `HFMA2` + 4 `HMUL2`) | Packed fp16 multiply (`tileir:74`; 8 `HFMA2` + 8 `HMUL2`) |
| 9 | Resource footprint | 26 registers, 16 CTAs/SM | 32 registers, 16 CTAs/SM | 124 registers (32 elements of index state live at once), 17,420 B shared, 4 CTAs/SM |
| 12 | Tuning surface | `BLOCK_SIZE`, `threads` | `BLOCK_SIZE`, `num_warps` | `tile`, `occupancy`; neither changes rows 2-6 (the fp32 winner uses `tile=2048` and has the same per-element structure, section 6) |

Rows that differ but do not matter here, one line each:

- Row 7: TileLang's fp32 round trip costs 36 instructions per thread and no time, since the kernel is DRAM-bound.
- Row 2, Triton vs TileLang: Triton's 16 scale loads against 2 show up as 633 `lg_throttle` samples and 59% more instructions, hidden under the DRAM bound (7% in latency, parity).
- Row 9: cuTile's 25% occupancy is not the loss. The ALU pipe is at 89.5% and more than one warp is eligible each cycle with only 15 warps resident per SM (`sm__warps_active.avg.per_cycle_active` 15.0); more resident warps would have no free issue slots to use. The fp32 capture runs at 43.75% theoretical occupancy and takes the same time (section 6).

## 6. Mechanisms

### cuTile

**M1. Per-element signed index arithmetic (primary; about 54% of instructions).**
Source: `impl_cutile.py:31-34`, `row = offsets // N; col = offsets % N; s_row = row // TILE_SIZE; s_col = col // TILE_SIZE` on a 4096-element `int32` tile.
Lowering: integer `//` and `%` on a tile become one `raw_binary_arith` per element, and Python floor semantics on a signed value add a correction:

```
$94:  Tile[int32,(4096)] = raw_binary_arith(lhs=$47, rhs=$93,  fn="floordiv")     # tileir:33
$105: Tile[int32,(4096)] = raw_binary_arith(lhs=$47, rhs=$104, fn="c_mod")        # tileir:36
$113: Tile[int32,(4096)] = raw_where(cond=$111, x=$112, y=$105)                   # tileir:44, sign fix-up
$124: Tile[int32,(4096)] = raw_binary_arith(lhs=$94,  rhs=$123, fn="floordiv")    # tileir:47
$135: Tile[int32,(4096)] = raw_binary_arith(lhs=$113, rhs=$134, fn="floordiv")    # tileir:50
```

`offsets` carries no non-negativity fact (the `assume_bounded` facts at `tileir:11-17` cover array extents only), so every division keeps its negative-value path.
SASS: for each of a thread's 32 elements, a magic-number divide (`IMAD.HI Rn, Rm, 0x66666667`, `SHF.R.U32.HI ..., 0x1f`, `LEA.HI.SX32 ..., 0x14`), the remainder (`IMAD Rn, Rq, -0x2800, Rm`) with its fix-up (`ISETP.GE`, `@!P VIADD Rn, Rn, 0x2800`), and two floor divisions by 128, each `LEA.HI`, `LOP3.LUT ..., 0xffffff80`, `ISETP.NE.U32`, `ISETP.LT`, `SHF.R.S32.HI ..., 0x7`, `LEA.HI.SX32 ..., 0xffffffff, 0x19`, `@!P IMAD.MOV` (offsets `0x000f0`-`0x02b90` and `0x04390`-`0x04580`). Static counts per thread: 32 `IMAD.HI 0x66666667`, 32 `VIADD ..., 0x2800`, 64 `LOP3 ..., 0xffffff80`, 64 `LEA.HI.SX32 ..., 0xffffffff, 0x19`.
Measurement: lines 31-34 execute 72,396,800 warp instructions (707 per warp), 5.4x TileLang's entire kernel, and hold 5,477 PC samples, of which `math_pipe_throttle` and `not_selected` are the dominant reasons (1,675 and 1,606 for lines 25 and 31-34 together).
Peers, same source formulation: Triton's source is the same four lines (`impl_triton.py:17-21`) and its PTX has two divisions per thread; TileLang's generated code has no per-element division at all. Elements `8k .. 8k+7` share a scale entry because 8 divides 128 and 128 divides 10240; both peer compilers use that, cuTile's does not.
Buys: nothing here. Costs: 707 of 1300 instructions.

**M2. `ct.gather` is one bounds-checked pointer load per element (about 22% of instructions).**
Source: `impl_cutile.py:37`, `ct.gather(s_ptr, (s_row, s_col), padding_value=0)`.
Lowering (`tileir:51-73`): each index is widened to `uint64`, compared against the runtime extent, multiplied by the runtime row stride and added to the base pointer:

```
$163 = raw_cmp(lhs=$159, rhs=$162, fn="lt")      # s_row < rows     tileir:55
$167 = raw_binary_arith(lhs=$159, rhs=$166, fn="mul")              # tileir:59
$172 = raw_cmp(lhs=$168, rhs=$171, fn="lt")      # s_col < cols     tileir:64
$177 = pointer_offset(pointer=$176, offset=$174)                   # tileir:69
$181, $182 = load_pointer(pointer=$177, mask=$173, padding_value=$180, ...)   # tileir:73
```

SASS per element: two `ISETP.GE.U32` against `UR10`/`UR11`, a sign extension, `IMAD.WIDE Rn, Rm, UR5, ...`, `LEA` + `LEA.HI.X`, a zero for the padding value, and `@!P LDG.E.U16` (`0x03300`-`0x046b0`). Static counts per thread: 64 bounds compares, 32 `IMAD.WIDE`, 32 `LDG.E.U16`.
Measurement: line 37 executes 29,388,800 warp instructions (287 per warp) and 3,276,800 of the kernel's 3,686,400 global load requests. The scale loads add no DRAM traffic: L1 miss sectors equal the peers' (6.593e6 vs 6.589e6).
Buys: safety for arbitrary indices. Costs: 9 instructions per element for a value that is the same for 128 consecutive elements.

**M3. Two layouts in one kernel, reconciled through shared memory (about 15% of instructions; small in time).**
Source: `ct.load`/`ct.store` on a partition view (`impl_cutile.py:28,41`) mixed with the gathered tile.
Lowering: `tile_load` and `tile_store` (`tileir:30,76`) are whole-tile operations and get 8 consecutive elements per thread; the index tile gets one element per thread at stride 128. The back end moves the X tile to the gather's ownership and the result back (matrix row 6). This is back-end behaviour read from SASS; the exact read-side mapping of the swizzled shared addresses was not decoded.
Measurement: 12 shared stores and 12 shared loads per thread (`smsp__sass_inst_executed_op_shared_{ld,st}.sum` = 1,228,800 each), 2 barriers, 17,420 B shared per CTA. `barrier` holds 426 samples (3.3%) and `short_scoreboard` 219. Bank conflicts are negligible (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` = 2,681 and 7,938 against 1.2 million accesses each). Lines 28 and 41 together are 195 instructions per warp; most of line 28's 2,982 samples (2,126) are the DRAM wait on X, which every backend has.

### TileLang (reference)

Source `impl_tilelang.py:34-42`, the same per-element formulation inside `T.Parallel(BLOCK_SIZE)`. The compiler vectorises to 8 fp16, proves the scale index constant across each vector and simplifies it to `blockIdx.x/640*80 + blockIdx.x%5*16 + i*8 + (threadIdx.x >> 4)` (`tilelang_fp16.cu:27`), and drops the tail guard. Result: 4 loads issued before the first wait (`0x00130`-`0x00160`), 66 instructions per warp, DRAM at 82.6%.

### Triton (reference)

`sizePerThread = [8]` makes X and Y one 128-bit access per run; the scale `tt.load` (`ttgir:30`) stays per element, 16 `LDG.E.U16` per thread with 8 on each of two addresses, and the divisions fold to two per thread in the back end. 105 instructions per warp, DRAM at 80.2%.

### Trying to break the explanation

Strongest alternative: cuTile is slow because it runs at 25% occupancy with 124 registers and two barriers, that is, it is short of resident warps and waiting, and the instruction count is incidental. Three measurements separate the two:

1. Where the samples are. If cuTile were waiting, its excess would sit in `long_scoreboard` or `barrier`. `long_scoreboard` is 1,604 samples below TileLang's, `barrier` is 426, and the execution-side classes carry more than the whole excess (section 3).
2. The issue-slot floor (224,865 cycles) exceeds the peers' measured time. Occupancy cannot remove instructions.
3. The fp32 contrast (`contrast_fp32/brief.md`). Doubling the bytes doubles the peers' time and leaves cuTile's unchanged, with a different tile and a different occupancy:

| | fp16 | fp32 |
|---|---:|---:|
| cuTile winner | `tile=4096, occupancy=4` | `tile=2048, occupancy=4` |
| cuTile registers / theoretical occupancy % | 124 / 25 | 70 / 43.75 |
| cuTile `smsp__inst_executed.sum` | 1.331e8 | 1.513e8 |
| cuTile `smsp__cycles_active.avg` | 352,625 | 363,700 |
| cuTile DRAM throughput % | 30.0 | 56.6 |
| cuTile CSV ms | 0.1864 | 0.1909 |
| TileLang `smsp__cycles_active.avg` | 117,732 | 234,000 |
| TileLang CSV ms | 0.0635 | 0.1236 |
| `dram__bytes_read.sum` (all backends) | 2.10e8 | 4.19e8 |

cuTile's time follows the element count, not the bytes and not the occupancy. In fp32 its index and gather lines execute the same counts as in fp16 (lines 32/33/34/37: 20.9 / 22.5 / 25.2 / 28.3 million against 20.7 / 23.0 / 25.0 / 29.4 million; `contrast_fp32/brief.md:297`, `bundle/brief.md:295`), and the same stall classes dominate (`not_selected` 2,960, `math_pipe_throttle` 2,682).

The catalogue entries "Per-element gather/scatter (cuTile)" and "Signed integer division and modulo per element (cuTile)" (`references/mechanisms.md`) describe this fingerprint, including "gap shrinking as the element widens". They were read after the analysis and are consistent with it; the evidence is the bundle above.

## 7. Why the net gap is 2.9x and not 10x

cuTile executes 9.85x TileLang's instructions per element but takes 2.94x the time. Time is roughly instructions / issue rate: TileLang issues on 19.4% of cycles because it spends the rest waiting for DRAM, cuTile issues on 63.8% (`smsp__issue_active.avg.per_cycle_active`). 9.85 / (0.638 / 0.194) = 2.99, against 2.83 in NCU duration and 2.94 in the CSV (`bundle/brief.md:166-170`).

So the two sides of the ratio are set by different resources. The peers' 0.064-0.068 ms is the time to move 210 MB in and 210 MB out at about 80% of DRAM peak; their resident warps sit at the first consumer of a load. cuTile's 0.186 ms is the time to push 133 million warp instructions through an ALU pipe running at 89.5%; its resident warps are queued for an issue slot. cuTile overlaps its instruction work with its own memory wait (its `long_scoreboard` samples are fewer than TileLang's), which is why the gap is 2.9x and not larger. The gap grows with M because the peers' per-element time keeps falling toward the DRAM limit while cuTile's stays fixed, and it halves at fp32 because the peers' DRAM time doubles.

## 8. Classification

**Codegen/lowering (cuTile), with a programming-model component.** The source formulation is the same in all three implementations. TileLang's compiler and Triton's back end fold the per-element index to one computation per 8-element run; cuTile lowers `//`, `%` and `ct.gather` literally, per element, with signed-floor corrections and bounds tests (attribution: compiler, for M1 and M2). The model contributes: `ct.gather` hands the back end a tile of independent pointers, so the fact that 128 consecutive elements share an index cannot be expressed through it, and mixing it with `ct.load`/`ct.store` forces the re-layout of M3. An author-level workaround exists (section 9), so the cost is not unavoidable in cuTile.

Not an autotune choice (the gap is the same across cuTile's winners), not a capture artifact (NCU and CSV agree within 5%), not a capability gap in the tensor-core sense.

## 9. Ranked directions

**cuTile**

1. Compute the scale index once per run of `TILE_SIZE` elements instead of once per element. With `TILE` a multiple of `TILE_SIZE` and `N` a multiple of `TILE_SIZE` (both hold for every benchmarked case: M is a multiple of 512, `config.yaml:13`, and every tile in the search space is a multiple of 128, `impl_cutile.py:16`), build the index on a `TILE // TILE_SIZE`-element tile, gather that, and broadcast it over the X tile reshaped to `(TILE // TILE_SIZE, TILE_SIZE)`. A sketch of this was compiled through the cuTile front end only (`variant/impl_cutile_variant.py` -> `variant/cutile_variant_fp16.tileir`): the `floordiv`/`c_mod` ops and the `load_pointer` shrink from `Tile[...,(4096)]` to `Tile[...,(32)]` (`cutile_variant_fp16.tileir:36-53,76`). What the closed back end emits for it, and its latency, are unknown.
   Confirming measurement: `smsp__inst_executed.sum` per element and `sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active` of the variant; the mechanism predicts DRAM throughput rising toward the peers' 80%.
2. Remove the gather altogether: index X as a 2-D array and let one CTA cover whole scale cells, so the scale is a scalar or a small `ct.load` tile and no division or pointer load remains. This also removes the second layout and with it the shared-memory round trips (M3). Not compiled here.
   Confirming measurement: `smsp__sass_inst_executed_op_shared_st.sum` and `launch__shared_mem_per_block` dropping to the peers' values.
3. If the per-element form is kept, make the index unsigned or otherwise known non-negative so the floor-division corrections disappear. Whether cuTile 1.3.0 then drops the fix-up chain was not tested.
   Confirming measurement: `ISETP` executions per element (244 per 1000 elements now, `bundle/brief.md:152`).

**Triton**: at parity; one scale load per 8-element run instead of per element would remove 14 of its 18 loads per thread and the `lg_throttle` samples, but the kernel is DRAM-bound, so no latency change is expected at this shape. Confirming measurement: `lg_throttle` samples and CSV latency at a small M where DRAM is not the bound.

**TileLang**: nothing indicated; it is at the DRAM bound with 66 instructions per warp.

## 10. Limits

- cuTile's back end is closed. The Tile IR shows what the front end asked for; the shared-memory re-layout, register count and absence of index folding are read from SASS of cuda-tile 1.3.0 and may differ in another version.
- Direction 1 is supported only by front-end IR. No SASS, counter or latency exists for it.
- The power-of-two observation in section 2 and the GH200 ratio rest on CSV rows alone; no profile was read for them.
- PC samples count sampled warps. cuTile has about 15 resident warps per SM against about 53 for the peers, so sample shares are compared as warp-time by stall reason, not converted to wall-clock shares.
