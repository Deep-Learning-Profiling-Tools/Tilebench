# TileBench Mechanism Catalogue

What the TileBench study has already established about how Triton, cuTile and
TileLang kernels win and lose on NVIDIA B200/GH200. Use it as a set of priors:
each entry names a mechanism, the fingerprint that identifies it in a case
bundle, and the check that confirms or rejects it. A prior becomes a finding
only when this case's own SASS, counters and PC samples show the fingerprint.
A case that matches nothing here is new information; say so.

## Blackwell SASS Glossary

Read `annotated_sass_*.txt` with this table. Execution counts and stall samples
on the same line tell you whether an instruction family matters.

| SASS | Meaning | Why it matters |
|---|---|---|
| `UTCHMMA` | tcgen05 tensor-core MMA accumulating in tensor memory (TMEM) | The Blackwell GEMM path. Accumulators cost no registers. |
| `HMMA.*` (e.g. `HMMA.1688.F32.TF32`) | Legacy MMA with register accumulators | Older path; accumulators inflate registers/thread and cut residency. |
| `LDTM` / `STTM` | TMEM load/store to registers | Epilogue drain of TMEM accumulators, or intermediates round-tripping through TMEM. |
| `UTCBAR` | tcgen05 commit signalling an mbarrier | Completion of MMA; pairs with a wait before buffer reuse. |
| `UTMALDG` / `UTMASTG` | TMA descriptor (bulk tensor) load/store | One issuing thread moves a whole tile; no per-lane address math. |
| `LDGSTS` (often `.E.BYPASS.128`) | Lane-issued async global-to-shared copy (`cp.async`) | Every participating lane computes addresses and issues copies. |
| `LDGDEPBAR` / `DEPBAR` | Commit / wait on an async-copy group | Where a lane-issued pipeline blocks for operands. |
| `SYNCS.PHASECHK…TRYWAIT`, `SYNCS.ARRIVE` | mbarrier wait / arrive | Producer-consumer handshake. Samples here are waiting, not computing. |
| `NANOSLEEP` | Sleep inside a wait loop | Idle role warps; large sample counts do not mean useful work is stalled. Some wait loops retry with no sleep: the samples then sit on the branch after `TRYWAIT`. |
| `BAR.SYNC` | Whole-CTA barrier (`__syncthreads`) | Serializes every warp; see barrier stall samples. |
| `ATOMS` / `ATOMG` | Shared / global atomic | CTA-private versus global contended updates. |
| `LDS` / `STS` (`.64`, `.128`) | Shared-memory load/store | Staging and layout conversion; check bank conflicts. |
| `LDG` / `STG` (`.E`, `.64`, `.128`) | Global load/store; suffix is access width | Width and lane stride set bytes/sector and L1 sector traffic. |
| `LDL` / `STL` | Local (stack) load/store | Real spills or local arrays; zero count means no spill regardless of registers. |
| `FFMA` / `FFMA2`, `HFMA2` | Scalar FMA / paired FMA (two values per instruction) | Paired forms halve issue count for the same arithmetic. NCU's "non-fused FP" rule can miss them. |
| `ISETP`, `IMAD`, `LEA`, `LOP3`, `PLOP3` | Compares, integer multiply-add, address math, predicates | Bounds checks and index arithmetic; compare dynamic counts per output element. |
| `BRA` backward with hot counts | Retained runtime loop | Loop not unrolled/specialized. |

## Reading Rules

- **Time is roughly instructions executed / issue rate.** `brief.md` computes
  this per backend. A gap carried by the instruction ratio is extra work
  (bounds checks, conversions, retained loops, scalar instead of packed ops). A
  gap carried by the issue-rate ratio is waiting (too few resident warps, or
  resident warps blocked on data or barriers). Diagnose the two differently.
- **A stall is charged to the consumer.** `long_scoreboard` samples land on the
  instruction that needs a pending load's value (a store, `PRMT`, `HADD2`,
  `FFMA`), not on the `LDG` that issued it. Find the producing load by walking
  back in the annotated SASS.
- **Memory-level parallelism.** Loads in flight per SM is about resident warps x
  loads a warp issues before its first dependent wait. Count the latter from
  SASS order. A kernel with full occupancy but two loads per wait can have far
  fewer misses overlapping than a 30%-occupancy kernel that issues 64.
  Signature of too little: `long_scoreboard` dominant, `lg_throttle` and
  `mio_throttle` near zero, low eligible warps, low DRAM %. Signature of enough:
  the load/store unit pushes back (`lg_throttle`, `mio_throttle`).
- **Half-precision paths differ by backend.** `HADD2.F32` is an fp16->fp32
  convert per loaded value; `HMUL2`/`HFMA2` operate on packed fp16 pairs. A
  source that multiplies before casting gets packed fp16 arithmetic and fewer
  converts (and different rounding) than one that casts first. When a gap exists
  for fp16 but not fp32, compare convert counts, load forms (`LDG.E.U16` vs
  wider) and where the stalls land.
- **Multi-kernel operators: apportion first.** Use the per-kernel roll-up to
  split each gap by kernel before diagnosing anything; the kernel that carries
  one backend's gap is often not the one that carries another's.
- **Small grid plus serial loop is a latency chain.** When a kernel has fewer
  CTAs than SMs and loops many times per CTA, its time is iterations x time per
  iteration, not throughput. Compare ns per iteration; occupancy and DRAM % say
  little. Check which shape parameter latency tracks in the scale trend.
- **Throttle and `not_selected` samples mark contention, not a slow
  instruction.** A hot PC at kernel entry with those stalls means the scheduler
  is saturated, which is the instruction-bound regime.
- **An outlier shape is two findings.** If the profiled shape's ratio is far
  from the median of the scale trend, report the steady gap and the
  shape-specific excess separately; the profile explains only what it captured.

## Fingerprints Seen in TileBench

Each entry: mechanism, where it was seen, bundle fingerprint, confirming check.

### GEMM-like kernels (matmul, batched matmul, attention, stream-K)

- **Lane-issued copies plus collective waits (TileLang).** Warp specialization
  is disabled in TileBench TileLang kernels (`TL_DISABLE_WARP_SPECIALIZED`), so
  the same threads issue `LDGSTS` copies, run `UTCHMMA`, then wait
  (`UTCBAR`, `SYNCS…TRYWAIT`, `BAR.SYNC`) inside the K loop. Triton and cuTile
  move tiles with `UTMALDG`/`UTMASTG`. Fingerprint: same register/shared
  residency limits as Triton but lower achieved occupancy and less tensor-pipe
  work per unit time. Confirm: wait/barrier PCs between MMA and the next copy carry the
  samples; tensor pipe % lower at equal or fewer instructions.
- **TF32 falls off the tcgen05 path (TileLang fp32).** fp32 GEMM emits
  `HMMA.1688.F32.TF32` with register accumulators while peers emit `UTCHMMA`.
  Fingerprint: 200+ registers/thread, theoretical occupancy 6 to 13%, the
  tcgen05 pipe (`sm__pipe_tc_cycles_active`) at zero while the general tensor
  pipe is busy, fp16 much closer than fp32 in the dtype trend.
- **2D TMEM lowering blocks pipelining (TileLang attention).** Outer loop runs
  serially; intermediates move TMEM -> fragment -> shared repeatedly
  (`LDTM`/`STTM`/`STS` hot). Fingerprint: similar DRAM bytes across backends,
  DRAM throughput low, tensor pipe underused.
- **K-tile size sets loop count and wait frequency.** A backend with a larger
  reduction tile can win with lower occupancy because it halves iterations.
  Compare winners' K tile and stage count before reading occupancy.
- **Pretransposed operand cached outside the timed region (Triton matmul).**
  Check `impl_triton.py` for a cached transpose before comparing memory traffic.
- **TMA/mbarrier machinery on short serial loops (cuTile, Blackwell).** Tiny
  grids with a load-wait-MMA round trip per iteration, sometimes with the
  accumulator moved between TMEM and registers each iteration (`LDTM`/`STTM`
  hot). Few instructions, long waits; the same backend can lead on GH200.
- **Role warps sleeping (cuTile).** `NANOSLEEP`/`SYNCS` samples and very high
  long-scoreboard ratios come from producer/consumer role warps; cuTile can be
  at parity with Triton despite them. Do not rank by stall ratio alone.

### Stencils, pooling, direct convolution

- **Full unrolling trades registers for work (TileLang).** Taps unrolled into a
  register fragment, several outputs per thread: fewer dynamic instructions and
  L1 sectors, but very high registers/thread and low theoretical occupancy with
  no `LDL`/`STL` spills. The net gain is smaller than the instruction saving
  because few resident warps hide load latency (long-scoreboard samples on the
  arithmetic that consumes the loads).
- **Lane mapping wastes sectors (Triton).** Strided lane layouts and repeated
  coefficient loads give low bytes/sector, L1 throughput near peak and
  `lg_throttle` samples on the offending `LDG`.
- **Gather bounds and retained loops (cuTile).** Hot `ISETP.GE.U32`, `LEA`,
  `R2UR`, a backward `BRA`, high ALU pipe %, math-pipe throttle. The arithmetic
  itself may be efficient (`FFMA2`).
- **Store-layout conversion through shared memory (Triton, cuTile).** Epilogue
  `STS…`/`BAR.SYNC`/`LDS.64`/`STG.E.64` converts compute layout to vector
  stores. Buys wide stores, costs shared traffic and a barrier. TileLang often
  stores directly.
- **Irregular gather with one CTA per SM (TileLang).** 140+ registers, low
  bytes/sector, long-scoreboard dominant, nothing resident to hide latency.

### Elementwise, copy and permutation kernels

- **Serialized load-then-store chains (TileLang `T.Parallel`).** Element-wise
  lowering can unroll into guarded load/store pairs, so only a couple of loads
  are outstanding per warp and each pair is a full round trip. See the
  memory-level-parallelism rule.
- **Scalar versus packed stores.** One narrow `STG` per element versus one
  `STG.E.128` per several elements: same sectors, many more store instructions.
  Whether a thread's elements are contiguous decides if packing is possible.
- **Strided lanes re-touch sectors (Triton).** Thread-contiguous layouts make a
  warp-level load span many sectors at a few bytes each; L1 absorbs the repeats
  (high hit rate, unchanged DRAM bytes) and the cost appears as `lg_throttle`.
- **Shared-memory re-layout before store (cuTile).** `STS`/`BAR.SYNC`/`LDS`
  between gather and store, with high registers and low occupancy that need not
  hurt if many loads are in flight per warp.

- **Per-element gather/scatter (cuTile).** Arbitrary-index `gather`/`scatter`
  lowers to one predicated narrow load or store per element with its own index
  math and bounds test, even when lanes share an index. Fingerprint: instruction
  count per element an order of magnitude above peers, ALU pipe high, identical
  DRAM bytes, gap shrinking as the element widens.
- **Signed integer division and modulo per element (cuTile).** `//` and `%`
  on an `int32` index tile are `floordiv` / `c_mod` ops in Tile IR and lower to
  a multiply, shift and compare sequence for every element. In an index-heavy
  kernel this can outweigh the gather itself. Peers can fold the same index
  into one expression per thread or per run of elements; count the divide
  sequences per thread in each backend's PTX or SASS.
- **Vector width set by lane count, not bytes (TileLang).** Narrow dtypes get
  proportionally narrower loads and stores than the same kernel in fp32.
- **In-kernel transposed element copies (TileLang GEMM operands).** Filling a
  shared tile element by element from a transposed global index gives one L1
  sector per lane (sectors per load request near 32, hit rate near 7/8,
  `lg_throttle` present) and starves the tensor pipe; peers load pre-transposed
  tiles by TMA with no `LDG` at all.

### Reductions, scans, histogram, sort, movement

- **CTA-private shared accumulation (TileLang win).** `ATOMS` on a shared
  buffer versus peers' `ATOMG` per element. Advantage grows with N.
- **Wider tile removes a reduction pass (TileLang win).** One wide tile halves
  load requests and drops a second-stage kernel/pass.
- **Single-warp scan (TileLang `CumSum1D`).** One warp scans while the rest
  wait at `BAR.SYNC`; barrier stall samples dominate at high occupancy.
- **Bytewise shared-to-shared movement (TileLang narrow dtypes).** Scalar byte
  `LDS`/`STS` through a second shared tile and a smaller winning tile, hence
  more CTAs. The wider dtype of the same operator is at parity. The rate limit
  is usually shared-store bank conflicts, not the byte count: compare
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` with the shared
  store count (peers also store bytes, without conflicts), and evaluate the
  generated destination index for one warp to count lanes per bank
  (bank = byte address / 4 mod 32). The wider dtype can carry the same
  conflicts hidden under its DRAM limit.

## Cross-Cutting Rules From the Study

- **Autotune is part of the system.** Winners change with scale. A gap at one
  shape can be a winner flip, not a codegen property; read the scale trend and
  winner column in `brief.md` before generalizing.
- **Dtype stratification separates capability gaps from movement problems.** A
  gap present for one dtype and absent for another points at instruction-family
  support (tensor path) or data-movement granularity (byte loads), not at the
  algorithm.
- **High occupancy is not throughput; low occupancy is not the loss.** Barrier-
  or dependency-stalled kernels reach 70-90% occupancy and lose; register-heavy
  kernels near 12% win. Always pair occupancy with what the resident warps are
  doing at the hot PCs.
- **DRAM % varies widely.** Across the B200 reports the dominant kernel's
  DRAM throughput has a median near 58%: about a quarter are under 25% and
  about a third are above 70%. Do not assume either regime; read it per case.
  Two backends above 80% are both bandwidth-bound and any other difference
  between them is hidden.
- **Parity band.** Differences within 10% are parity unless the trend across
  shapes is consistent in sign.
- **Torch is frequently the fastest reference** (vendor libraries) and has no
  saved profile; report the ratio, assign no mechanism.

## Capture Quirks

- TileLang reports embed `tvm_kernels.cu` with empty content, so the brief
  says the embedded source is empty. That is expected. Regenerate the CUDA
  with `tl_codegen.py`: the `tvm_kernels.cu:<line>` numbers in the annotated
  SASS line up with the regenerated file when the winner config is right, which
  is also the check that it is.
- Some TileLang captures are reduced (kernel replay, far fewer replay passes,
  no stall/eligible-warp counters). `brief.md` flags missing families; a
  missing counter is not zero, and DRAM byte totals from different replay modes
  are not comparable.
- cuTile mangled kernel names end in the specialization constants
  (`…I7_I7_I2_I128` = two 7s then tile 2x128); the occupancy hint appears only
  in the winner log.
- Triton/cuTile embedded sources carry the capture machine's path; compare the
  kernel body to the checkout's `impl_<backend>.py` rather than the whole file.
- The released profiles capture the largest configured shape per dtype. Verify
  with `launch__grid_dim_*` x winner tile = problem shape.
- NCU duration ratios often differ from CSV ratios (different warm state and
  profiler overhead). CSV is the benchmark; NCU explains.
