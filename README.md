# TileBench

A modular performance benchmarking framework for comparing NVIDIA cuTile, Triton, and PyTorch implementations.

## Features
- **Multi-DSL Support**: Compare cuTile, Triton, and PyTorch.
- **Extensible**: Add new operators by creating a directory in `benchmarks/operators/` and registering a generator in `data/tensors.py`.
- **Robust Verification**: Mandatory accuracy checks against PyTorch reference.
- **Precise Timing**: CUDA event-based timing with warmup rounds.
- **Visualization**: Automated chart generation.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Benchmarks
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 scripts/run_bench.py --operator vector_add
```

### 3. Visualize Results
```bash
python3 scripts/visualize.py
```

## Project Structure
- `benchmarks/operators/`: Operator implementations and configs.
- `core/`: Engine, timer, and verifier logic.
- `data/`: Input tensor generation and registry.
- `scripts/`: Entry points for benchmarking and visualization.
- `results/`: JSON logs and comparison plots.
