# Device Context capture checklist

Commands to run on each remote device to fill the `unknown` fields of
`skills/device/<DEV>/<snapshot>/SKILL.md` and to refresh a snapshot. Record raw output
verbatim (no hand-edited values), note the hostname and date, and cite the capture as a
new `S<n>` source in `manifest.json` (`verified_on_device: true` only when the SKILL.md
numbers were taken from that device's own output). Never copy a value from another device.

Common to all devices:

```bash
hostname; date -u +%Y-%m-%dT%H:%M:%SZ; uname -r; cat /etc/os-release | head -2   # -> Software snapshot: OS / kernel
python -c "import sys; print(sys.version)"                                       # -> Software snapshot: Python
pip freeze                                                                       # -> Software snapshot: every package version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)"
python -c "import triton; print(triton.__version__)"
python -c "import tilelang, importlib.metadata as m; print(tilelang.__version__, m.version('apache-tvm-ffi'))"
python -c "import triton.profiler as p; print(p)"                                 # -> Proton backend (also: scripts/run_bench.py provenance 'proton_backend')
```

## NVIDIA (B200, GH200)

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.memory,pcie.link.gen.max,power.limit --format=csv
#   -> Identity (name); Software (driver); Memory hierarchy (HBM capacity, MiB); Execution model / Limits (max SM clock, memory clock)
nvidia-smi -q -d CLOCK,MEMORY,SUPPORTED_CLOCKS
#   -> Limits: base/boost clock schedule (currently unknown on both devices)
python - <<'EOF'
import torch
p = torch.cuda.get_device_properties(0)
for k in ("name","major","minor","multi_processor_count","warp_size","max_threads_per_block",
          "max_threads_per_multi_processor","regs_per_multiprocessor","shared_memory_per_block",
          "shared_memory_per_block_optin","shared_memory_per_multiprocessor","L2_cache_size",
          "total_memory","clock_rate","memory_clock_rate","memory_bus_width"):
    print(k, getattr(p, k, "n/a"))
EOF
#   -> Identity (compute capability); Execution model (SM count, warp size, threads per SM/block);
#      Memory hierarchy (registers per SM, default/opt-in shared memory per block, shared memory per SM,
#      L2 bytes, HBM bytes); Limits (clock_rate, memory_clock_rate, memory_bus_width).
#      GH200 still needs: shared_memory_per_block (default), clock_rate, memory_clock_rate, memory_bus_width.
nvcc --version; python -c "import cuda.tile, importlib.metadata as m; print(cuda.tile.__version__, m.version('nvidia-cuda-tileiras'))"
#   -> Software snapshot: nvcc / tileiras / cuda-tile (B200 still needs tileiras, nvcc, nvvm, apache-tvm-ffi)
python -c "import tilebench.hardware as h; print(h.detect_arch(), h.supports_tma(), h.supports_tmem(), h.last_level_cache_bytes())"
#   -> framework facts cross-check (arch label, TMA/TMEM flags, LLC bytes)
```

## AMD (MI300X)

```bash
rocminfo                       # -> Identity (marketing name, gfx target); Execution model (compute units, SIMDs per CU,
                               #    wavefront size, max waves per CU, workgroup max size, max clock MHz);
                               #    Memory hierarchy (L1/L2/L3 KB lines, LDS "Segment: GROUP" size)
amd-smi static                 # -> Identity (product name, VBIOS); Memory (VRAM size MB); Limits (max engine/memory clock);
                               #    Software (driver version)
amd-smi metric --clock         # -> Limits: current/max sclk and mclk
rocm-smi --showmeminfo vram --showclocks --showproductname --showdriverversion
cat /opt/rocm/.info/version; hipconfig --version; ls /sys/class/kfd/kfd/topology/nodes/*/properties | head -1
grep -E "num_xcc|simd_count|wave_front_size|max_waves_per_simd|lds_size_in_kb|cu_per_simd_array|array_count|max_engine_clk_fcompute" /sys/class/kfd/kfd/topology/nodes/*/properties
#   -> Execution model (XCD count, SIMDs, max waves per SIMD = currently unknown); Memory (LDS KB);
#      Limits (max engine clock)
python - <<'EOF'
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, p.gcnArchName, p.multi_processor_count, p.warp_size, p.max_threads_per_block,
      p.max_threads_per_multi_processor, p.regs_per_multiprocessor, p.shared_memory_per_block,
      p.shared_memory_per_multiprocessor, p.L2_cache_size, p.total_memory)
EOF
#   -> max LDS per work-group (shared_memory_per_block, currently unknown), HBM bytes, L2 bytes
python -c "import tilebench.hardware as h; print(h.detect_arch(), h.last_level_cache_bytes())"
#   -> framework facts cross-check (cdna3, 268435456)
```

## AWS Trainium2 (Trn2)

```bash
neuron-ls                      # -> Identity / Execution model: chips per instance, NeuronCores per chip, device memory per chip,
                               #    NeuronLink topology
neuron-ls --json-output        # -> same, machine-readable (record verbatim)
neuron-top -j                  # -> per-NeuronCore utilisation + device memory (snapshot only; do not derive performance claims)
neuron-monitor --help          # -> optional: confirm tool version for the Software snapshot
curl -s http://169.254.169.254/latest/meta-data/instance-type        # -> Identity: instance type (trn2.48xlarge / trn2u.48xlarge / trn2.3xlarge)
echo "NEURON_LOGICAL_NC_CONFIG=$NEURON_LOGICAL_NC_CONFIG"; cat /opt/aws/neuron/etc/neuron-logical-nc-config 2>/dev/null
#   -> Execution model: Logical NeuronCore (LNC) setting (currently unknown)
apt list --installed 2>/dev/null | grep -i neuron ; rpm -qa 2>/dev/null | grep -i neuron
#   -> Software snapshot: aws-neuronx-dkms (driver), aws-neuronx-runtime-lib, aws-neuronx-tools, aws-neuronx-collectives
modinfo neuron | grep -E "^version"                                   # -> Software snapshot: kernel driver version
pip freeze | grep -i -E "neuron|nki|torch"                            # -> Software snapshot: neuronx-cc, torch-neuronx, torch, neuronx-distributed, nki
neuronx-cc --version; neuron-profile --version                        # -> Software snapshot: compiler and profiler versions
cat /opt/aws/neuron/.version 2>/dev/null                              # -> Software snapshot: Neuron SDK release
python - <<'EOF'
import nki, nki.language as nl   # if the nki package is installed
print(getattr(nki, "__version__", "n/a"))
print("tile_size.pmax", nl.tile_size.pmax, "psum_fmax", nl.tile_size.psum_fmax, "gemm_moving_fmax", nl.tile_size.gemm_moving_fmax)
EOF
#   -> Limits: SBUF partition count, PSUM free-dim max, matmul moving free-dim max (fills 'PSUM banks / bytes per bank'
#      and 'per-instruction tensor-size limits' once confirmed against the NKI Reference Skill)
```

Fields that no command fills (vendor documentation only; re-check the cited URLs on each
snapshot): per-dtype peak TFLOPS tables, Tensor Core / Matrix Core / Tensor Engine dtype lists,
TMA/TMEM/DMA capability statements, NeuronCore engine widths, Infinity Cache topology.
