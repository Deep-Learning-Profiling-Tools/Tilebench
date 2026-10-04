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
| Triton 108, cuTile 110 | 638ea84974d742512932d4d042de840ce6bb47b0 |
| Triton 2 (bitonic_sort fp16, fp32; replaced, see below) | 5610f18f1cc695168c85712e1ecd5bfcec5439bb |
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

**bitonic_sort (Triton), replaced reports.** `pad_kernel` is not tuned, but read its BLOCK from
`_DEFAULT_CONFIG`, into which the profiling replay installs the autotune winner of
`bitonic_step_kernel`. The fp32 report (winner BLOCK=512, num_warps=8) therefore launched `pad_kernel`
with BLOCK=512 / grid 32768, while the formal autotune path pads with BLOCK=1024 / grid 16384 (both
block 128); the 300 `bitonic_step_kernel` launches were identical. fp16 (winner BLOCK=1024) was not
affected. Fix 5610f18f (identical to exp/mi300x 124fdc94): `pad_kernel` always uses `_PAD_BLOCK = 1024`;
search space, candidate order, warmup/rep, algorithm, step kernel and default step config are unchanged,
and the default and autotune launch sequences are the same as before, so the formal CSVs stand.

Both Triton bitonic_sort reports were re-profiled from 5610f18f with the methodology above
(validated, 301/301 launches; pad grid 16384 / block 128; steps grid 16384 / block 256 for fp16 and
grid 32768 / block 256 for fp32, i.e. the formal winners; the full launch sequence equals the formal
autotune path; `--page details` readable) and replaced on Hugging Face in one commit; no other report
changed, locally or remotely.

| report | old sha256 | new sha256 |
|---|---|---|
| bitonic_sort/triton_fp16.ncu-rep | 400048fa3f2c86bd70144d6f8a552e22fca9a23208160fd5493a90682d5cd74b | bfd52382c6d5c3f145ff972bcce7a112135d473e4629c09a0954c6fa71b0efa7 |
| bitonic_sort/triton_fp32.ncu-rep | d7208a62eb5dcbe3b5259a452a61c132740d81d2bbb73d9c8fe9770aaf6fc9a5 | c3aa63fdfe0e1b05a5889b10d826ab75d06b7c1800496abab7dffa1844b66eac |

HF `bcui2/NCU_report`: previous GH200 upload 45ee6451ee770133202d8ed2ba3af41884b17dcb; dataset head
before the replacement 645591bbb808b4ceeecae2c4ec61b78151ad6249; replacement commit
7cb810502bc6f668647c4c0d67caa491999b3563. Evidence: `bitonic_triton_replacement.json`,
`hf_remote_audit_bitonic.txt`, `hf_upload_bitonic.log`.
