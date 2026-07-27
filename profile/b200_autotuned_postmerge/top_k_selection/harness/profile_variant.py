"""Launch one TileLang variant bitonic step (NCU-bracketed) and verify it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir()
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from variants_tilelang import LOG_ARGS, VARIANTS  # noqa: E402


def reference_step(values: torch.Tensor, n: int, stage: int, stride: int) -> torch.Tensor:
    out = values.clone()
    idx = torch.arange(n // 2, device=values.device)
    i1 = (idx // stride) * (2 * stride) + (idx % stride)
    i2 = i1 + stride
    a, b = values[i1], values[i2]
    descend = ((i1 // stage) % 2) == 1
    swap = descend == (a > b)
    out[i1] = torch.where(swap, b, a)
    out[i2] = torch.where(swap, a, b)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--n", type=int, default=1048576)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dump-ir", default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    n, stride, stage = args.n, args.stride, args.n // 64

    kernel = VARIANTS[args.variant](
        n, "float32", BLOCK_SIZE=args.block_size, threads=args.threads
    )
    if args.dump_ir:
        out = Path(args.dump_ir)
        tag = f"tilelang_{args.variant}_N{n}"
        (out / f"{tag}.cu").write_text(kernel.get_kernel_source())
        kernel.export_ptx(str(out / f"{tag}.ptx"))
        kernel.export_sass(str(out / f"{tag}.sass"))
        print(f"dumped {tag}")

    if args.variant in LOG_ARGS:
        a2, a3 = stage.bit_length() - 1, stride.bit_length() - 1
    else:
        a2, a3 = stage, stride

    values = torch.randn(n, device="cuda", dtype=torch.float32)

    if args.verify:
        expect = reference_step(values, n, stage, stride)
        got = values.clone()
        kernel(got, n, a2, a3)
        torch.cuda.synchronize()
        ok = torch.equal(expect, got)
        print(f"verify {args.variant}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            bad = (expect != got).nonzero().flatten()
            print(f"  {bad.numel()} mismatches, first at {bad[:8].tolist()}")
            sys.exit(1)

    for _ in range(3):
        kernel(values, n, a2, a3)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    kernel(values, n, a2, a3)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"variant={args.variant} N={n} stage={stage} stride={stride}")


if __name__ == "__main__":
    main()
