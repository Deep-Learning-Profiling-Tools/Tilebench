---
name: tilebench-analysis
description: Explain TileBench kernel performance (Triton, cuTile, TileLang on B200/GH200/MI300X) from saved benchmark results, autotune winners and released NCU profiles, with no GPU needed. Resolves a case into an evidence bundle, and applies TileBench-specific mechanism knowledge. Use for "why is X slower than Y on <operator>", single-backend diagnosis, and scale/dtype/hardware trend questions.
---

# TileBench Analysis

This skill is the TileBench layer: it knows where the evidence is, how the three
DSLs lower code, and which mechanisms the study has already found. It carries
its own Nsight Compute method ([ncu-analysis](references/ncu-analysis.md): six
analysis dimensions and diagnosis patterns for saved TileBench reports and
three-backend comparison). The NCU pass says
what is slow, from counters and SASS; the IR pass says why the DSL produced
it; the report joins the two.

| Job | Owner |
|---|---|
| Resolve case, CSV trends, winner configs, saved reports, evidence bundle | This skill (`scripts/tb_case.py`) |
| DSL/operator mechanisms, Blackwell SASS meanings, capture quirks | This skill ([mechanisms](references/mechanisms.md)) |
| Why a DSL emitted a pattern: construct, compiler rule, intermediate code, alternative | This skill: [TileLang](references/tilelang-lowering.md) (`scripts/tl_codegen.py`), [Triton](references/triton-lowering.md) (`scripts/triton_ir.py`), [cuTile](references/cutile-lowering.md) (`scripts/cutile_ir.py`) |
| NCU analysis of each kernel: six dimensions, patterns, metric names, report API (Pass A) | This skill ([ncu-analysis](references/ncu-analysis.md)) |
| Comparison verdict and report | This skill |

What each step needs:

| Step | Needs |
|---|---|
| Read saved reports, build the bundle | Nsight Compute's Python module (`<nsight-compute>/extras/python` on `PYTHONPATH`); no GPU |
| Triton IR (`triton_ir.py`) | Triton only; no GPU, explicit target |
| TileLang generated CUDA (`tl_codegen.py`) | TileLang only; no GPU and no `nvcc` (the target is forced with `--arch`, default `sm_100`); `--ptx` additionally needs `nvcc`. Operators whose source selects a kernel by device capability need the profiled GPU or that selection forced |
| cuTile Tile IR (`cutile_ir.py`) | cuda-tile only; no GPU, driver or `tileiras` (the signature is built from the shapes given) |

If a step's requirement is missing, say which, do the rest, and label what
could not be checked.

### Hardware targets

`--hardware` is the only target setting. Pass the same name to `tb_case.py` and to each IR script so the
IR is generated for the GPU the reports were captured on. The IR differs by
target: an operator can take a different tensor-core path or a different
kernel altogether.

| `--hardware` | Reports | Triton IR | TileLang CUDA / TIR | cuTile Tile IR | Status |
|---|---|---|---|---|---|
| `B200` | yes | `cuda:100` | `sm_100` | `sm_100` | Verified identical to IR generated on a B200 |
| `GH200` | yes | `cuda:90` | `sm_90` | `sm_90` | Scripts run; check the result against the GH200 report's SASS before citing it. Hopper has TMA but no tensor memory, so do not carry a Blackwell TMEM/tcgen05 explanation over |
| `MI300X` | rocprof-compute profiles (counters and sampled instructions, no execution counts) | `hip:gfx942` | not available | not available | Triton only, so there is no cross-DSL comparison: compare with PyTorch on MI300X or with Triton on B200/GH200. Bundle and IR tested; no full analysis run yet |

For a GPU outside TileBench, the IR scripts also accept its architecture in
place of a name (`--hardware sm_89`; for Triton also `backend:arch`). That
shows what the compilers would emit there; there are no reports to check it
against.

Winner configs are logged for B200. For another target, take the config from
that hardware's own evidence (the launch statistics and the constants in the
mangled kernel name of its report) and say where it came from; do not reuse
the B200 winner without checking it matches.

Saved evidence only: this skill grants no permission to run GPU work,
benchmark, autotune, install packages or modify kernels. Reading a compiler's
intermediate output with the `*_ir`/`tl_codegen` helpers is a compile, not a
run.

## 1. Build the Case Bundle

Run one command from the TileBench checkout. `<python>` is the workspace
interpreter that can import `ncu_report`; resolve the script relative to this
skill's directory.

```bash
<python> <skill-dir>/scripts/tb_case.py <B200|GH200|MI300X> <operator> <dtype> --out <output>/bundle
# optional: --params 'M=640'   --backends tilelang,triton   --reports-dir <dir>   --no-network
```

Reports already in the checkout (`evidence/`, `outputs/ncu/<hardware>/<operator>/`)
or under any `--reports-dir` are used as-is; otherwise they are downloaded.

It selects the CSV row (largest shape by default, which is what the released
profiles capture), finds winner configs (checkout, pinned git archive, then
GitHub), finds or downloads the saved reports, and writes:

- `brief.md`: benchmark row and ratios, scale/dtype/hardware trend tables with
  winners, NCU side-by-side with exact metric names, a per-element work ledger,
  the instructions / issue-rate time model (a first-order check, not a verdict),
  each slower backend's excess PC samples by stall reason (with the samples
  that sit on mbarrier/sleep waits counted separately), and per backend the
  order of global and TMA loads and load waits, the dynamic opcode
  mix, stall totals, PC samples by opcode and by source line, hottest PCs with
  SASS, NCU rule output and capture warnings. It ends with suggested contrast
  cases. **Read this first and in full.**
- `<backend>/annotated_sass_*.txt`: every instruction with its execution count,
  PC-sample share, source line and stall breakdown.
- `<backend>/detail_*.json`, `<backend>/ncu.json`, `<backend>/embedded_source/`,
  `case.json`: exact records for citation.

Then read the three `impl_<backend>.py` files for the operator, in
`benchmarks/operators/<op>/` or `tilebench/benchmarks/operators/<op>/`
depending on the branch ([layout](references/tilebench-layout.md)) and verify the join:
captured grid x winner tile should equal the problem shape. Do not rebuild this
extraction by hand; write extra code only for a question the bundle cannot answer.
If a backend's capture is reduced (the brief says no SASS or PC samples) and
another hardware's report for that backend exists, bundle it as a proxy for the
generated code and label what you take from it as inferred.
MI300X has no NCU reports. Its bundle carries the rocprof-compute profile
instead: headline counters, stall and instruction-type samples, and the sampled
instructions in `<backend>/sampled_asm_*.txt`, with the profile's own winner
config. Follow the AMD section of [ncu-analysis](references/ncu-analysis.md).

## 2. Diagnose

Three passes, in this order. Each pass has its own output; do not merge them
early. The first pass finds what is slow from the profiles alone, the second
finds what each compiler did, and the third joins them.

### Pass A: NCU analysis of every backend

Do this before reading the mechanism catalogue or any intermediate code, so the
diagnosis is not steered by what a DSL is expected to do.

1. Read [ncu-analysis](references/ncu-analysis.md) and follow it. For each
   backend's kernel (for multi-kernel operators, first apportion the gap with
   the per-kernel roll-up and take the kernels that carry it), go through all
   six dimensions and write the answers side by side: launch geometry and
   occupancy, block balance and tails, stall reasons with per-instruction
   hotspots, pipes and tensor cores, binding resource, memory access and cache
   behaviour.
2. Work from the raw extraction, not only the summary: read
   `<backend>/annotated_sass_*.txt` end to end for the hot region, and query
   the report through the NCU Python API for any counter the brief does not
   list (`ncu.json` holds every metric; its layout is in ncu-analysis). `brief.md` is a starting table,
   not the analysis. Its time model is a first-order check, never a verdict.
3. Match each kernel against the pattern table in ncu-analysis and state its
   binding resource: DRAM bandwidth, load latency, instruction issue, a specific pipe,
   shared memory, synchronization, occupancy.
4. Account for each gap in time: use the brief's "Where the extra time sits"
   sample table and the load-order line. The stall reasons holding the excess
   samples are where the gap is spent.
5. Treat every NCU rule as a hypothesis and confirm or refute it against SASS.

Output of Pass A: one short NCU diagnosis per backend (binding resource, hot
PCs, evidence), and a list of the measured differences between backends that
could explain each gap, each with its counter and SASS evidence. Include
differences you cannot yet explain.

### Pass B: IR analysis of every backend

1. Get each compiler's intermediate form for the winner config (use supplied
   files when present; the scripts compile, they never run a kernel):
   - TileLang: generated CUDA, and TIR on request, via `scripts/tl_codegen.py`
     ([tilelang-lowering](references/tilelang-lowering.md)).
   - Triton: `ttir`/`ttgir`/`llir`/`ptx` via `scripts/triton_ir.py`
     ([triton-lowering](references/triton-lowering.md)).
   - cuTile: front-end Tile IR via `scripts/cutile_ir.py`
     ([cutile-lowering](references/cutile-lowering.md)); its back end is
     closed, so the last step is read from SASS.

   **Filling in the script arguments.** The scripts decide how to compile
   (target, no GPU, no launch). You supply which kernel instance: the
   arguments the benchmark called it with. For each backend:

   1. Open `impl_<backend>.py`. Find the kernel (`@tilelang.jit`,
      `@triton.jit`, `@ct.kernel`) and read its parameter list.
   2. Read `run()` in the same file down to the kernel call. It shows what
      each parameter receives: reshaped views, computed output sizes, strides,
      rounded block counts. Use the non-autotune call as the template.
   3. Take values from three places: the case parameters in `brief.md` (the
      winner log's `params`), defaults the benchmark fills in
      (`config.yaml` `case_defaults`, and the operator's input generator in
      `data/tensors.py`, under `tilebench/` in the package layout, for anything still missing), and that
      backend's own winner config. Each backend has its own winner; do not
      reuse another backend's block sizes.
   4. Write one argument per kernel parameter:

      | | Arrays | Shape scalars | Config |
      |---|---|---|---|
      | `tl_codegen.py` | `--tensor SHAPE:dtype`, in order | `--kw NAME=VALUE` | `--kw NAME=VALUE` (block sizes, `threads`, `num_stages`) |
      | `triton_ir.py` | `--arg NAME=ptr:<dtype>` (`desc:<dtype>:<AxB>` for a `TensorDescriptor`) | `--arg NAME=int:VALUE` or `float:VALUE` | `--arg NAME=const:VALUE`, plus `--num-warps` and `--num-stages` |
      | `cutile_ir.py` | `--arg SHAPE:dtype`, in parameter order | `--arg VALUE`, in order | `--arg VALUE`, in order |

      For Triton, a parameter annotated `tl.constexpr` is `const:`; anything
      else is runtime (`ptr:`, `int:`, `float:`). Give runtime integers their
      real value. Dtypes may be written either way (`fp16` or `float16`,
      `i32` or `int32`). A TileLang kernel that takes only scalars needs no
      `--tensor`.

   Example, `2d_max_pooling` fp16 with `N=4, C=128, H=640`, kernel 3, stride
   2, padding 1. `run()` views the input as `(N*C, H, W)`, the generator sets
   `W = H`, and the output side is `(640 + 2 - 3) // 2 + 1 = 320`:

   ```bash
   tl_codegen.py .../impl_tilelang.py max_pool2d_kernel --tensor 512x640x640:float16 --tensor 512x320x320:float16 \
       --kw N=4 --kw C=128 --kw H=640 --kw W=640 --kw kernel_size=3 --kw stride=2 --kw padding=1 --kw dtype=float16 \
       --kw BLOCK_R=4 --kw BLOCK_C=128 --kw threads=128 --out <output>/tilelang_fp16.cu
   triton_ir.py .../impl_triton.py max_pool2d_kernel --arg input_ptr=ptr:fp16 --arg output_ptr=ptr:fp16 \
       --arg H=int:640 --arg W=int:640 --arg H_out=int:320 --arg W_out=int:320 \
       --arg kernel_size=const:3 --arg stride=const:2 --arg padding=const:1 --arg BLOCK_R=const:1 --arg BLOCK_C=const:512 \
       --num-warps 4 --out <output>/triton_ir
   cutile_ir.py .../impl_cutile.py max_pool2d_kernel --arg 512x640x640:float16 --arg 512x320x320:float16 \
       --arg 3 --arg 2 --arg 1 --arg 4 --arg 128 --out <output>/cutile_fp16.tileir
   ```

   A wrong argument does not fail; it compiles a different specialisation.
   Each script prints what it resolved (signature, target, kernel symbol).
   Confirm the result is the profiled kernel before citing it: threads per
   block and shared memory against the report's launch statistics, load and
   store widths and loop counts against the captured SASS, and for cuTile the
   constants at the end of the mangled kernel name in the report.
2. Fill the [lowering decision matrix](references/lowering-matrix.md): the same
   twelve decisions answered for every backend from its intermediate code.
3. Check each intermediate form against the captured SASS (load widths, loop
   counts, constants) and report any version mismatch. If a toolchain is
   unavailable, apply the rules to the source and label the result inferred.

Output of Pass B: the matrix rows where the backends differ, each with the IR
line that shows it.

### Pass C: Join, test, attribute

1. **Join.** Every Pass A difference needs a Pass B row that produces it
   (source construct -> IR op -> SASS offset -> counter and samples), and every
   differing Pass B row needs a Pass A measurement showing whether it matters.
   - A measured difference with no IR cause: say so and look again in the IR;
     it may be a back-end decision (state that) or a missed row.
   - An IR difference with no measured consequence: report it in one line as
     not mattering here. Do not promote it to a mechanism.
   The gap is explained only when the joined pairs account for where the excess
   samples sit.
2. **Try to break the leading explanation.** Name the strongest alternative
   and the measurement that separates the two, then make it. Equal bytes or
   sectors do not rule out memory waiting; more instructions do not prove an
   instruction-bound gap.
3. **Test representativeness and chase the contrast.** Use the trend tables:
   does the gap hold across scale, dtype and hardware, and does a winner-config
   change line up with a change in the gap? When the gap flips or shrinks in a
   sibling case, bundle that case for the affected backend into a second output
   directory and diff it: same backend, what changed? If you attribute a gap to
   a config-specific effect, the gap must move with the config across shapes.
4. **Cross-check the catalogue.** Only now read the relevant family in the
   [mechanism catalogue](references/mechanisms.md). Use it to name what you
   found and to catch a known fingerprint you missed, and verify any match
   against this bundle. A catalogue entry is not evidence.
5. **Attribute** each joined mechanism to the author, the compiler, the
   autotuner or a missing capability, and derive the source-level change from
   the matrix row.
6. **Reconcile the net result.** Explain why the measured latency ratio is what
   it is: which saving is real, what cost offsets it, what the resident warps
   are doing at the hot PCs.

For harder cases use the [execution model](references/execution-model.md),
[DSL pathways](references/dsl-pathways.md) and
[operator pathways](references/operator-pathways.md).

## 3. Report

Write `REPORT.md` in the output directory (or return it as text where file
output is unavailable), findings first:

1. **Verdict:** case, CSV latencies and ratios, parity or divergence (10% band),
   and the one-sentence cause.
2. **Is this point representative?** Scale, dtype and hardware trend; winner flips.
3. **NCU diagnosis per backend:** binding resource, hot PCs and the evidence,
   from Pass A.
4. **Selected configurations and work ledger:** winners, grid/block, work per
   CTA/thread, per-output normalized counts.
5. **Lowering decision matrix:** the differing rows only, one column per
   backend, each cell with its intermediate-code evidence.
6. **Mechanisms, per backend:** the joined pairs from Pass C, each with its full chain and exact citations
   (`file:line`, SASS offset, metric name, sample counts). The chain starts at the DSL construct and lowering rule and quotes the
   intermediate code (generated CUDA, `ttgir` or Tile IR). State what each design choice buys and what it costs.
7. **Why the net gap is this size:** the reconciliation.
8. **Classification:** algorithm/programming model, autotune choice,
   codegen/lowering, capability gap, or capture artifact.
9. **Ranked directions:** per backend, the source-level change the evidence
   points to (what to write differently in the DSL) and the single measurement
   that would confirm it. No invented speedup numbers.
10. **Limits:** only those that could change a conclusion above, in a few lines.

Commit to the best-supported explanation. Hard rules, and the only ones:

- Benchmark latency comes from the CSV; NCU duration is never substituted for it.
- A missing counter is not zero; a reduced capture is stated once, where it bites.
- Stall ratios `per_issue_active` are not elapsed-time shares. PC samples count
  sampled warps: compare them between backends as warp-time, not wall-clock share.
- Label inference as inference; do not present an unverified prior as a finding.
- Cite exact metric names and record paths so every number can be re-derived.
