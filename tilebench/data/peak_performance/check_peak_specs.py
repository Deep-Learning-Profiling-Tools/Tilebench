"""Quick sanity-check: print B200 roofline numbers for mul2.

Peak values are read from B200.json next to this file (measured peak
bandwidth + per-dtype peak TFLOPS), the same file the metrics code uses.

Run from Tilebench/:
    python data/peak_performance/check_peak_specs.py
"""

import json
from pathlib import Path

# ── dtype element sizes ──────────────────────────────────────────────────────
DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "int8": 1,
}

# ── Hardware specs (B200) ────────────────────────────────────────────────────
_PEAK = json.loads((Path(__file__).parent / "B200.json").read_text())
PEAK_BW_GBs   = _PEAK["peak_bw_GBs"]   # measured HBM peak bandwidth  (GB/s)
PEAK_BW_TBps  = PEAK_BW_GBs / 1000
PEAK_TFLOPS   = {dt: _PEAK["peak_tflops"][dt] for dt in DTYPE_BYTES}

# ── mul2 memory access pattern ───────────────────────────────────────────────
#   1 read + 1 write  →  bytes = n * dtype_size * 2
#   1 multiply        →  flops = n

print("=" * 65)
print(f"{'B200 specs':}")
print(f"  Peak HBM bandwidth : {PEAK_BW_GBs:.1f} GB/s  ({PEAK_BW_TBps:.2f} TB/s, measured)")
print()

print("=" * 65)
print(f"{'mul2 Roofline analysis per dtype':}")
print(f"  Operation : x * 2  (1 load + 1 store per element)")
print()
print(f"{'dtype':<8} {'dtype_size':>10} {'AI (FLOP/B)':>12} "
      f"{'Mem roof (TFLOPS)':>18} {'Peak TFLOPS':>12} {'Ridge pt (FLOP/B)':>18} {'Regime':<12}")
print("-" * 95)

for dtype, ds in DTYPE_BYTES.items():
    # Arithmetic Intensity: flops / bytes = n / (n * ds * 2) = 1 / (ds * 2)
    AI = 1.0 / (ds * 2)

    # Memory bandwidth ceiling at this AI (TFLOPS)
    mem_roof_tflops = PEAK_BW_TBps * AI   # TB/s × FLOP/Byte = TFLOPS

    peak_tf = PEAK_TFLOPS.get(dtype, 0)

    # Ridge point: where memory roof meets compute roof
    # peak_bw_TBps * x = peak_tflops  →  x = peak_tflops / peak_bw_TBps
    ridge = peak_tf / PEAK_BW_TBps if peak_tf else float("nan")

    regime = "memory-bound" if AI < ridge else "compute-bound"
    attained = min(mem_roof_tflops, peak_tf)

    print(f"{dtype:<8} {ds:>10} bytes {AI:>12.4f} "
          f"{mem_roof_tflops:>18.2f} {peak_tf:>12.1f} {ridge:>18.1f} {regime:<12}")

print()
print("=" * 65)
print(f"Bandwidth efficiency reference (% of {PEAK_BW_GBs:.1f} GB/s):")
print()

# For a given n (problem size) and latency, bandwidth = bytes / latency
# At 100% efficiency: latency = bytes / peak_bw
example_n = 16 * 1024 * 1024  # 16M elements
print(f"  Example: n = {example_n // (1024*1024)}M elements")
print()
print(f"  {'dtype':<8} {'bytes (MB)':>12} {'ideal latency (ms)':>20} {'at 50% efficiency (ms)':>24}")
print("  " + "-" * 68)
for dtype, ds in DTYPE_BYTES.items():
    total_bytes = example_n * ds * 2
    ideal_ms    = total_bytes / (PEAK_BW_GBs * 1e9) * 1e3   # ms at 100%
    half_ms     = ideal_ms * 2                                # ms at 50%
    print(f"  {dtype:<8} {total_bytes / 1e6:>12.1f} MB {ideal_ms:>20.4f} ms {half_ms:>24.4f} ms")
