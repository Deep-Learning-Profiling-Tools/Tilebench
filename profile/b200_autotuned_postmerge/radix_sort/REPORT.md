# radix_sort — decomposing TileLang's gap into radix width vs kernel cost

int32 keys, N=1M and N=10M. B200 sm_100, 148 SMs, NCU 2026.1.1, idle GPU pinned
by UUID. All three backends verified against `torch.sort` (`sorted_correct=True`).
Method: `kernel-perf-differential` skill.

## Verdict

**Half the gap to Triton, and *all* of the gap to cuTile, is radix width — not
kernel quality.** TileLang sorts 1 bit per pass and needs **32 passes**; Triton
and cuTile pack 2 bits and need **16**. Normalizing that away:

| | headline | per pass |
|---|---:|---:|
| TL / Triton | **3.66x** | **1.83x** |
| TL / cuTile | **1.90x** | **0.95x** |

Per pass TileLang is *faster than cuTile*. The residual ~1.8x against Triton is
real and is dominated by the scatter kernel — which is where the two `T.cumsum`
calls live. So the cumsum answer is **yes, but it is now roughly half the story
it used to be**, and the other half is an algorithmic choice that has nothing to
do with TileLang codegen.

## Read this first: the comparison moved under TileLang

Commit `8909454` (2026-07-25) — *"ping-pong buffers + tuner budget"* — rebuilt
`impl_triton.py`, `impl_cutile.py`, `impl_torch.py` and `config.yaml`. It did
**not** touch `impl_tilelang.py`, whose last change was `6cab61b` on 2026-07-07.

That rebuild is where the 2-bit packing came from:

```python
# impl_triton.py / impl_cutile.py        # impl_tilelang.py
_RADIX_BITS = 2                          (no radix constant)
_RADIX = 1 << _RADIX_BITS
for shift in range(0, 32, _RADIX_BITS):  for bit in range(32):
```

The earlier profile in `profile/radix_sort_ncu_20260720/` predates that commit
and reported 2.13x / 1.61x. TileLang's absolute time is unchanged (3.942 ms at
N=20M, then and now); the ratio moved to 3.51x / 1.71x purely because the other
two got faster. **This is the same effort asymmetry as `streamk_matmul`.**

## Where the time goes

Full `run()` under NCU, all kernels captured and attributed by name and launch
geometry.

### N=10M — 32 vs 16 passes

| backend | total | passes | per pass |
|---|---:|---:|---:|
| TileLang | 2,673.3 us | 32 | **83.5 us** |
| Triton | 854.8 us | 16 | **53.4 us** |
| cuTile | 1,469.4 us | 16 | **91.8 us** |

Per-pass: TL/Triton **1.56x**, TL/cuTile **0.91x**.

| | share | grid | blk | per-pass us |
|---|---:|---:|---:|---:|
| **TileLang scatter** (`main_kernel`) | **69.3%** | 9,766 | **64** | 57.9 |
| TileLang count | 12.3% | 9,766 | 128 | 10.2 |
| TileLang scan/per-block | 8.0% | **10** | 128 | 6.6 |
| TileLang count-blocks | 5.4% | **10** | 128 | 4.5 |
| TileLang scan/bb | 5.1% | **1** | 128 | 4.3 |
| **Triton scatter** | 52.4% | 9,766 | 128 | **28.0** |
| **cuTile scatter** | 66.7% | 9,766 | 128 | 61.3 |

Scatter per pass: TileLang **2.07x** Triton, **0.94x** cuTile.

## The scatter kernel — where cumsum shows up

One invocation at N=1M, identical grid (977) for all three:

| | TileLang | Triton | cuTile |
|---|---:|---:|---:|
| duration | **13,504 ns** | 7,232 ns | 11,584 ns |
| block size | **64** | 128 | 128 |
| registers/thread | 76 | 64 | 108 |
| achieved occupancy | **18.58%** | 33.21% | 20.61% |
| issue / cycle | **0.22** | 0.40 | 0.44 |
| cycles | 17,216 | 6,603 | 14,289 |
| **stall: barrier** | **4.29** | 1.27 | 0.43 |
| stall: short scoreboard | 2.57 | 1.47 | 1.21 |
| stall: long scoreboard | 2.47 | 3.85 | 2.34 |
| DRAM % of peak | 3.90 | 7.30 | 4.56 |

Decomposing the 2.61x cycle gap to Triton:

```
instructions  ~1.43x        issue rate  0.55x        =>  2.61x cycles
```

**The issue rate dominates, not instruction count** — the opposite of the
`sigmoid` and `matmul` findings. And the reason the issue rate is low is visible
directly: **barrier stall 4.29 versus Triton's 1.27 and cuTile's 0.43** — 3.4x
and 10x higher. That is the signature of `T.cumsum` lowering to a block-wide
scan with CTA-wide barriers, of which the scatter has two:

```python
T.cumsum(ones_scan, dim=0)
T.cumsum(zeros_scan, dim=0)
```

DRAM sits at 3.90% of peak, so nothing here is bandwidth-limited. This is a
synchronization and occupancy problem, consistent with the earlier report.

The 64-thread block compounds it: TileLang's tuner picked `threads=64`, giving
half the competitors' block size and 18.58% occupancy against Triton's 33.21%.
Fewer warps per CTA means less to hide each barrier behind.

## A second, N-dependent problem: single-CTA scan kernels

Three of TileLang's five kernels have grids that collapse at small N, because
their grid is `ceildiv(K, BLOCK_SIZE)` where `K` is already `ceildiv(N, 1024)`:

| N | scan-hierarchy grids | share of TileLang time |
|---|---|---:|
| 1M | **1, 1, 1** | **45.1%** |
| 10M | 10, 10, 1 | 18.5% |

At N=1M, three kernels run on a **single CTA** — 1 of 148 SMs — and together
cost more than the scatter does. Triton's equivalent stage uses grid 39/39/1 at
N=10M and is only 25.3% of its time.

This is why the gap is worst at low N (3.22x at N=1M) and eases as N grows. It
is a distinct problem from the cumsum barrier cost and would need a separate fix.

## What is established, and what is not

**Established:** the 32-vs-16 pass-count difference and its exact arithmetic
effect; per-pass parity with cuTile (0.95x) and a 1.83x per-pass deficit to
Triton; the scatter kernel as 69.3% of TileLang's time; its barrier stall at
3.4x/10x the competitors with DRAM at 3.90%; the 64-thread/18.58%-occupancy
launch; and the single-CTA scan collapse at low N.

**Not established:** that `T.cumsum` *specifically* causes the barrier stall, as
opposed to other synchronization in the scatter. The evidence is
circumstantial-by-elimination — the scatter contains two block-wide scans, the
barrier stall is its dominant excess, and nothing else in the kernel is a
plausible CTA-wide sync source. **No intervention was run.**

**Not established:** that giving TileLang 2-bit packing would recover 2x. Pass
count halves, but per-pass cost would rise (a 4-way split needs 4 counters and
more scan state). Triton's own 2-bit scatter is only 28.0 us/pass against
TileLang's 57.9 us/pass at 1 bit, so the trade looks favourable — but that is an
inference, not a measurement.

## Recommended order

1. **Port 2-bit packing to `impl_tilelang.py`.** This is the parity fix the
   07-25 commit gave the other two backends and skipped for TileLang. Largest
   single lever, and it makes every subsequent comparison meaningful.
2. **Widen the scatter's thread count.** `threads=64` at 18.58% occupancy is
   half of Triton's; the tuner's search space should be revisited after (1).
3. **Fix the single-CTA scan collapse** so the hierarchy keeps more than one CTA
   busy at small N. Worth ~45% of TileLang's time at N=1M, ~18% at N=10M.
4. **Only then** attack the `T.cumsum` barrier cost, which is the genuine
   codegen item and the hardest.

## Reproducing

```bash
cd profile/b200_autotuned_postmerge/radix_sort
source harness/env.sh
$NCU --profile-from-start off --section SpeedOfLight --section LaunchStats \
  --target-processes all -o reports/tilelang_N10M \
  $PY harness/profile_radix.py --backend tilelang --N 10000000
```

## Artifacts

- `harness/profile_radix.py` — one full `run()`, correctness-checked against
  `torch.sort`; calls the module entry point so it survives backend rebuilds.
- `reports/{tilelang,triton,cutile}_N1M.ncu-rep` — 6 sections incl. SchedulerStats.
- `reports/{tilelang,triton,cutile}_N10M.ncu-rep` — SpeedOfLight + LaunchStats.
- `analysis/kernel_attribution.txt` — per-kernel time/grid/block at both N.

Supersedes `profile/radix_sort_ncu_20260720/`, whose measurements predate the
07-25 Triton/cuTile rebuild.
