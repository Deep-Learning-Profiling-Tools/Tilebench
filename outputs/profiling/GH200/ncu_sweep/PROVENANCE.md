# GH200 NCU reports: provenance

Device: NVIDIA GH200 480GB (sm_90, hopper), NCU 2025.4.0.0. Methodology (all backends):
`--set full --import-source on --replay-mode application --cache-control none
--app-replay-mode strict --profile-from-start off`, kernel-name regex + exact launch count from
`outputs/profiling/GH200/kernel_counts.json`; harness: 3 warmups and a 120 MiB (2x the 60 MiB L2)
eviction outside the profiler range, exactly one `impl.run()` inside it. Inputs: the sweep-max case of
each operator/dtype; configs: the formal GH200 autotune winner (`ncu_catalogue.json`, built from
`results/GH200/logs/autotune_logs/*_autotune_triton-cutile-tilelang.json`).

| reports | profiling source SHA |
|---|---|
| Triton 110, cuTile 110 | 638ea84974d742512932d4d042de840ce6bb47b0 |
| TileLang 110 | d6ddb62256ef4e5b1a02c70be6eac41064134e6b |

The formal performance CSVs (`results/GH200/csv/`) were measured on earlier sources:
c882fe5074a82618f44ce456aa390b25102cb509 (89 CSVs) and 3c5eccbfcba65b9683361e2d2d22a60151cfc23e
(batched_matmul_autotune.csv). Between those and the profiling sources only profiling infrastructure
changed, plus one operator change:

**destindex (TileLang).** The profiling source d6ddb622 splits the TileLang default config of
destindex into `_DEFAULT_NOPE_CONFIG` and `_DEFAULT_ROPE_CONFIG`, both with the previous values
{BLOCK_SIZE: 1024, threads: 128}, so that the nope and rope launches of the one kernel can each replay
their own autotune winner (GH200 fp32: nope BLOCK_SIZE=512, rope 1024). The kernel, its search space,
the autotune algorithm and the default-mode behaviour are unchanged. Exact winner replay was verified
(strict replay 110/110; destindex fp32 NCU launches: block 64 / grid 122880 for nope and block 64 /
grid 2560 for rope), and correctness was verified against the torch reference for every dtype with
and without the winner.
