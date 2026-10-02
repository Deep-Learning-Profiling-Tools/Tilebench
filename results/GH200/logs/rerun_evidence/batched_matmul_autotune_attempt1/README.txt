First formal attempt of the GH200 batched_matmul autotune run (source c882fe50, warmup=1/repeat=3,
autotune candidate timing warmup=1/rep=3), kept unmodified as failure evidence.
fp32 BATCH=32 M=160: a cuTile exhaustive_search candidate (tile_m=128, tile_n=64, tile_k=64,
occupancy=4) raised CUDA "illegal instruction"; the sticky error poisoned the process, TileLang's
tuning of the same case failed, and the remaining 45 cases were skipped by the engine
("not supported for input generation (AcceleratorError ...)"), exit 0, 15/60 rows.
The operator was rerun once in a fresh process with the same source and protocol (user decision).
