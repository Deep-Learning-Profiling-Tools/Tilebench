"""Launch one tuned top_k_selection bitonic step for NCU profiling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir()
)
sys.path.insert(0, str(ROOT))


# Autotune winners recorded in
# results/b200_autotuned_postmerge/operators/top_k_selection.json
# (identical for every k at a given N).
CONFIGS = {
    4096: {
        "tilelang": {"BLOCK_SIZE": 512, "threads": 256},
        "triton": {"BLOCK_SIZE": 512, "num_warps": 8, "num_stages": 1},
        "cutile": {"tile": 1024, "occupancy": 8},
    },
    1048576: {
        "tilelang": {"BLOCK_SIZE": 512, "threads": 256},
        "triton": {"BLOCK_SIZE": 512, "num_warps": 8, "num_stages": 1},
        "cutile": {"tile": 512, "occupancy": 16},
    },
}


def launch_tilelang(values: torch.Tensor, n: int, stage: int, stride: int):
    from benchmarks.operators.top_k_selection import impl_tilelang

    cfg = CONFIGS[n]["tilelang"]
    kernel = impl_tilelang.bitonic_step_kernel(n, "float32", **cfg)

    def launch():
        kernel(values, n, stage, stride)

    return launch


def launch_triton(values: torch.Tensor, n: int, stage: int, stride: int):
    import triton

    from benchmarks.operators.top_k_selection import impl_triton

    cfg = CONFIGS[n]["triton"]
    grid = (triton.cdiv(n, cfg["BLOCK_SIZE"] * 2),)

    def launch():
        impl_triton._bitonic_step_kernel[grid](
            values,
            n,
            stage,
            stride,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return launch


def launch_cutile(values: torch.Tensor, n: int, stage: int, stride: int):
    import cuda.tile as ct

    from benchmarks.operators.top_k_selection import impl_cutile

    cfg = CONFIGS[n]["cutile"]
    kernel = impl_cutile._tuner.kernel_with_hints(occupancy=cfg["occupancy"])
    grid = ((n // 2 + cfg["tile"] - 1) // cfg["tile"], 1, 1)
    stream = torch.cuda.current_stream()

    def launch():
        ct.launch(stream, grid, kernel, (values, n, stage, stride, cfg["tile"]))

    return launch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("tilelang", "triton", "cutile"), required=True)
    parser.add_argument("--n", type=int, choices=tuple(CONFIGS), required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    values = torch.randn(args.n, device="cuda", dtype=torch.float32)

    stage = args.n // 64
    stride = 16
    launch = {
        "tilelang": launch_tilelang,
        "triton": launch_triton,
        "cutile": launch_cutile,
    }[args.backend](values, args.n, stage, stride)

    for _ in range(3):
        launch()
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    launch()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    cfg = CONFIGS[args.n][args.backend]
    print(
        f"backend={args.backend} N={args.n} stage={stage} stride={stride} "
        f"config={cfg} checksum={values.sum().item():.8g}"
    )


if __name__ == "__main__":
    main()
