"""Launch one tuned matrix-transpose kernel for NCU profiling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

M = 4096
N = 20480

CONFIGS = {
    "fp16": {
        "tilelang": {"BLOCK_TILE": 128, "threads": 128},
        "triton": {"BLOCK_TILE": 64, "num_warps": 8},
        "cutile": {"tile": 128, "occupancy": 4},
    },
    "int8": {
        "tilelang": {"BLOCK_TILE": 64, "threads": 128},
        "triton": {"BLOCK_TILE": 128, "num_warps": 8},
        "cutile": {"tile": 128, "occupancy": 4},
    },
}

TORCH_DTYPES = {"fp16": torch.float16, "int8": torch.int8}
TILELANG_DTYPES = {"fp16": "float16", "int8": "int8"}


def make_input(dtype: str) -> torch.Tensor:
    if dtype == "int8":
        return torch.randint(-128, 128, (M, N), device="cuda", dtype=torch.int8)
    return torch.randn((M, N), device="cuda", dtype=TORCH_DTYPES[dtype])


def tilelang_launcher(x: torch.Tensor, output: torch.Tensor, dtype: str):
    from benchmarks.operators.matrix_transpose import impl_tilelang

    cfg = CONFIGS[dtype]["tilelang"]

    def launch():
        impl_tilelang.matrix_transpose_kernel(
            x,
            output,
            dtype=TILELANG_DTYPES[dtype],
            BLOCK_TILE=cfg["BLOCK_TILE"],
            threads=cfg["threads"],
        )

    return launch


def triton_launcher(x: torch.Tensor, output: torch.Tensor, dtype: str):
    import triton

    from benchmarks.operators.matrix_transpose import impl_triton

    cfg = CONFIGS[dtype]["triton"]
    tile = cfg["BLOCK_TILE"]
    grid = (triton.cdiv(M, tile), triton.cdiv(N, tile))

    def launch():
        impl_triton._transpose_kernel[grid](
            x,
            output,
            M,
            N,
            x.stride(0),
            x.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_TILE=tile,
            num_warps=cfg["num_warps"],
        )

    return launch


def cutile_launcher(x: torch.Tensor, output: torch.Tensor, dtype: str):
    import cuda.tile as ct

    from benchmarks.operators.matrix_transpose import impl_cutile

    cfg = CONFIGS[dtype]["cutile"]
    tile = cfg["tile"]
    grid = ((M + tile - 1) // tile, (N + tile - 1) // tile, 1)
    kernel = impl_cutile._tuner.kernel_with_hints(occupancy=cfg["occupancy"])
    stream = torch.cuda.current_stream()

    def launch():
        ct.launch(stream, grid, kernel, (x, output, tile))

    return launch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("tilelang", "triton", "cutile"), required=True)
    parser.add_argument("--dtype", choices=tuple(CONFIGS), required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    x = make_input(args.dtype)
    output = torch.empty((N, M), device="cuda", dtype=x.dtype)
    launch = {
        "tilelang": tilelang_launcher,
        "triton": triton_launcher,
        "cutile": cutile_launcher,
    }[args.backend](x, output, args.dtype)

    for _ in range(3):
        launch()
    torch.cuda.synchronize()

    if not torch.equal(output, x.T):
        raise RuntimeError("transpose output did not match torch reference")

    torch.cuda.profiler.start()
    launch()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    print(
        f"backend={args.backend} dtype={args.dtype} shape={M}x{N} "
        f"config={CONFIGS[args.dtype][args.backend]} "
        f"checksum={output.to(torch.float64).sum().item():.8g}"
    )


if __name__ == "__main__":
    main()
