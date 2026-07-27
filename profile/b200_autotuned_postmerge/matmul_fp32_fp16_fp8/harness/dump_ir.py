"""Dump PTX / SASS / generated CUDA for each matmul backend."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from profile_matmul import DTYPES, make_inputs  # noqa: E402

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"


def _write(out: Path, name, data):
    p = out / name
    mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
    with open(p, mode) as f:
        f.write(data)
    print(f"wrote {p} ({p.stat().st_size} bytes)")


def _sass_from_cubin(out: Path, tag: str, cubin: bytes):
    _write(out, f"{tag}.cubin", cubin)
    r = subprocess.run([CUOBJDUMP, "-sass", str(out / f"{tag}.cubin")],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        _write(out, f"{tag}.sass", r.stdout)


def dump_tilelang(a, b, dt, M, N, K, out, warp_spec):
    from tilelang_variants import DEFAULT_CFG, build_matmul

    impl = build_matmul(warp_spec=warp_spec, **DEFAULT_CFG[a.dtype])
    c = torch.empty((M, N), device="cuda", dtype=a.dtype)
    name = str(a.dtype).removeprefix("torch.")
    kern = impl.compile(a, b, c, dtype=name)   # JITKernel exposes the sources
    tag = f"tilelang{'_ws' if warp_spec else ''}_{dt}"
    _write(out, f"{tag}.cu", kern.get_kernel_source())
    kern.export_ptx(str(out / f"{tag}.ptx"))
    kern.export_sass(str(out / f"{tag}.sass"))


def dump_triton(a, b, dt, M, N, K, out):
    from benchmarks.operators.matmul_fp32_fp16_fp8 import impl_triton

    impl_triton.run(a, b)
    torch.cuda.synchronize()
    compiled = None
    for entry in impl_triton.matmul_kernel.device_caches.values():
        for ck in entry[0].values():
            compiled = ck
    assert compiled is not None
    tag = f"triton_{dt}"
    for key in ("ttgir", "ptx"):
        if key in compiled.asm:
            _write(out, f"{tag}.{key}", compiled.asm[key])
    _sass_from_cubin(out, tag, compiled.asm["cubin"])


def dump_cutile(a, b, dt, M, N, K, out):
    from benchmarks.operators.matmul_fp32_fp16_fp8 import impl_cutile

    tmp = Path(os.environ["CUDA_TILE_TEMP_DIR"])
    for stale in tmp.glob("matmul_kernel*"):
        stale.unlink()
    impl_cutile.run(a, b)
    torch.cuda.synchronize()
    cubins = sorted(tmp.glob("matmul_kernel*.cubin"))
    assert cubins, f"no cuTile cubin in {tmp}"
    _sass_from_cubin(out, f"cutile_{dt}", cubins[-1].read_bytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--dtype", default="fp16", choices=tuple(DTYPES))
    p.add_argument("--M", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=20480)
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    a, b = make_inputs(DTYPES[args.dtype], args.M, args.N, args.K)

    print("=== tilelang (warp-spec off, as shipped) ===")
    dump_tilelang(a, b, args.dtype, args.M, args.N, args.K, out, False)
    print("=== tilelang_ws (warp-spec on) ===")
    dump_tilelang(a, b, args.dtype, args.M, args.N, args.K, out, True)
    print("=== triton ===")
    dump_triton(a, b, args.dtype, args.M, args.N, args.K, out)
    print("=== cutile ===")
    dump_cutile(a, b, args.dtype, args.M, args.N, args.K, out)


if __name__ == "__main__":
    main()
