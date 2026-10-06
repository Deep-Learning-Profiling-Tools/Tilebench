"""The versioned calibration protocol. Every parameter that can change a
calibrated value is here and enters the protocol hash; the run records the
full protocol next to the samples."""
from __future__ import annotations

from tilebench.llm.v2.calibration.schema import sha256_json

PROTOCOL_ID = "tilebench-empirical-roofline/2"

DEFAULT = {
    "protocol": PROTOCOL_ID,
    "batches": 3,
    "warmup": 30,
    "repeat": 100,
    "target_sample_ms": 2.0,          # inner launches per timed sample are calibrated to reach this
    "statistic": "per point: median of samples within each batch, then median of the batch medians; "
                 "mode value: maximum robust throughput over the registered valid points",
    "device_share_min": 0.95,         # device-bound test: profiled busy share OR device/event time ratio must reach this
    "saturation_tolerance": 0.02,     # best point > this above the runner-up at the edge of the range -> flagged
    "telemetry_interval_ms": 250,     # clock/power/temperature sampling during every mode (nvidia-smi -lms)
    "batch_spread_flag": 0.03,        # flagged (not invalid) above this relative spread of batch medians
    "gemm": {"sizes": [4096, 8192, 12288, 16384],
             "check_block": 128},
    "hbm": {"primary_probes": ["d2d_copy", "sm_copy"],
            "diagnostic_probes": ["sm_read", "fill_write"],
            "sizes_mib": [16, 32, 64, 512, 1024, 2048, 4096],
            "min_working_set_over_llc": 4.0,
            "plateau_tolerance": 0.05},
    "vector": {"chains": 8, "block": 1024, "num_warps": 4,
               "programs_per_sm": [4, 8, 16, 32],
               "iters": 16384,            # the point is timed at `iters`; 2*iters is the scaling check
               "scaling_tolerance": 0.10,
               "a": 0.999, "b": 0.001},
}

QUICK = {  # smoke test of the code path only; a quick profile can never be activated
    **DEFAULT,
    "protocol": PROTOCOL_ID + "+quick",
    "batches": 1, "warmup": 2, "repeat": 3, "target_sample_ms": 0.5,
    "gemm": {"sizes": [1024], "check_block": 64},
    "hbm": {**DEFAULT["hbm"], "sizes_mib": [16, 512]},
    "vector": {**DEFAULT["vector"], "programs_per_sm": [4], "iters": 512},
}


def protocol_hash(p: dict) -> str:
    return sha256_json(p)


def is_quick(p: dict) -> bool:
    return str(p.get("protocol", "")).endswith("+quick")
