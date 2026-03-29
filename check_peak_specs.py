"""Quick sanity-check: print B200 roofline numbers for mul2.

Run from Tilebench/:
    python check_peak_specs.py
"""

# ── Hardware specs (B200) ────────────────────────────────────────────────────
PEAK_BW_GBs   = 8000.0   # HBM3e peak bandwidth  (GB/s)
PEAK_BW_TBps  = PEAK_BW_GBs / 1000  # = 8.0 TB/s

PEAK_TFLOPS = {
    "fp16": 400.0,
    "bf16": 400.0,
    "fp32":  80.0,
    "int8": 800.0,
}

# ── dtype element sizes ──────────────────────────────────────────────────────
DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "int8": 1,
}

# ── mul2 memory access pattern ───────────────────────────────────────────────
#   1 read + 1 write  →  bytes = n * dtype_size * 2
#   1 multiply        →  flops = n

print("=" * 65)
print(f"{'B200 specs':}")
print(f"  Peak HBM bandwidth : {PEAK_BW_GBs:.0f} GB/s  ({PEAK_BW_TBps:.1f} TB/s)")
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
print("Bandwidth efficiency reference (% of 8000 GB/s):")
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
