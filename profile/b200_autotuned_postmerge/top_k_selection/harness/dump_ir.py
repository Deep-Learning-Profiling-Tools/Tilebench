"""Dump PTX / SASS / source IR for the tuned top_k_selection bitonic step."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir()
)
sys.path.insert(0, str(ROOT))

from profile_topk import CONFIGS  # noqa: E402

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
NVDISASM = "/usr/local/cuda/bin/nvdisasm"


def _write(outdir: Path, name: str, data) -> None:
    path = outdir / name
    mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
    with open(path, mode) as f:
        f.write(data)
    print(f"wrote {path} ({path.stat().st_size} bytes)")


def _disasm(outdir: Path, tag: str, cubin: bytes) -> None:
    _write(outdir, f"{tag}.cubin", cubin)
    cubin_path = outdir / f"{tag}.cubin"
    for flag, ext in (("-sass", "sass"), ("-ptx", "cubin_ptx")):
        res = subprocess.run(
            [CUOBJDUMP, flag, str(cubin_path)], capture_output=True, text=True
        )
        if res.returncode == 0 and res.stdout.strip():
            _write(outdir, f"{tag}.{ext}", res.stdout)


def dump_tilelang(n: int, outdir: Path) -> None:
    from benchmarks.operators.top_k_selection import impl_tilelang

    cfg = CONFIGS[n]["tilelang"]
    kernel = impl_tilelang.bitonic_step_kernel(n, "float32", **cfg)
    tag = f"tilelang_N{n}"
    _write(outdir, f"{tag}.cu", kernel.get_kernel_source())
    kernel.export_ptx(str(outdir / f"{tag}.ptx"))
    kernel.export_sass(str(outdir / f"{tag}.sass"))
    print(
        f"tilelang N={n} regs={kernel.n_regs} spills={kernel.n_spills} "
        f"max_threads={kernel.n_max_threads}"
    )


def dump_triton(n: int, outdir: Path) -> None:
    import triton

    from benchmarks.operators.top_k_selection import impl_triton

    cfg = CONFIGS[n]["triton"]
    values = torch.randn(n, device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(n, cfg["BLOCK_SIZE"] * 2),)
    impl_triton._bitonic_step_kernel[grid](
        values,
        n,
        n // 64,
        16,
        BLOCK_SIZE=cfg["BLOCK_SIZE"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    torch.cuda.synchronize()

    compiled = None
    for entry in impl_triton._bitonic_step_kernel.device_caches.values():
        for ck in entry[0].values():
            compiled = ck
    assert compiled is not None, "no compiled triton kernel found"

    tag = f"triton_N{n}"
    for key in ("ttir", "ttgir", "llir", "ptx"):
        if key in compiled.asm:
            _write(outdir, f"{tag}.{key}", compiled.asm[key])
    _disasm(outdir, tag, compiled.asm["cubin"])
    print(f"triton N={n} regs={compiled.n_regs} spills={compiled.n_spills}")


def dump_cutile(n: int, outdir: Path) -> None:
    """Capture the JIT-produced cubin via CUDA_TILE_TEMP_DIR (set by the caller)."""
    import cuda.tile as ct

    from benchmarks.operators.top_k_selection import impl_cutile

    tmpdir = Path(os.environ["CUDA_TILE_TEMP_DIR"])
    for stale in tmpdir.glob("_bitonic_step_kernel*"):
        stale.unlink()

    cfg = CONFIGS[n]["cutile"]
    kernel = impl_cutile._tuner.kernel_with_hints(occupancy=cfg["occupancy"])
    values = torch.randn(n, device="cuda", dtype=torch.float32)
    grid = ((n // 2 + cfg["tile"] - 1) // cfg["tile"], 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (values, n, n // 64, 16, cfg["tile"]),
    )
    torch.cuda.synchronize()

    tag = f"cutile_N{n}"
    cubins = sorted(tmpdir.glob("_bitonic_step_kernel*.cubin"))
    assert cubins, f"no cuTile cubin appeared in {tmpdir}"
    _disasm(outdir, tag, cubins[-1].read_bytes())
    for bc in sorted(tmpdir.glob("_bitonic_step_kernel*.bytecode")):
        _write(outdir, f"{tag}.tileir_bytecode", bc.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--backends", nargs="+", default=["tilelang", "triton", "cutile"]
    )
    parser.add_argument("--ns", nargs="+", type=int, default=[4096, 1048576])
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    dispatch = {
        "tilelang": dump_tilelang,
        "triton": dump_triton,
        "cutile": dump_cutile,
    }
    for backend in args.backends:
        for n in args.ns:
            print(f"=== {backend} N={n} ===")
            dispatch[backend](n, outdir)


if __name__ == "__main__":
    main()
