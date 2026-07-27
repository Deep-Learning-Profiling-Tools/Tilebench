"""Did the broken fp8 kernel pass TileBench's own verifier at the config.yaml
tolerances (atol=5.0, rtol=0.1)? Runs the real verify() against the real
impl_torch reference.
"""

import sys
from pathlib import Path

import torch
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmarks.operators.matmul_fp32_fp16_fp8 import impl_torch  # noqa: E402
from core.verifier import verify  # noqa: E402
from data.tensors import get_generator  # noqa: E402
from tilelang_variants import build_matmul  # noqa: E402

CFG = yaml.safe_load(
    (ROOT / "benchmarks/operators/matmul_fp32_fp16_fp8/config.yaml").read_text())
ATOL = float(CFG["verify"]["atol"])
RTOL = float(CFG["verify"]["rtol"])


def main():
    print(f"config.yaml tolerances: atol={ATOL}  rtol={RTOL}\n")
    M = N = 4096
    for K in (1024, 8192, 20480):
        gen = get_generator("matmul_fp32_fp16_fp8")
        a, b = gen(M=M, N=N, K=K, dtype=torch.float8_e4m3fn)
        ref = impl_torch.run(a, b)
        torch.cuda.synchronize()
        c = torch.empty(M, N, device=a.device, dtype=a.dtype)

        for tag, cfg in (
            ("BROKEN bn=256 (256x256x128)", dict(bm=256, bn=256, bk=128, threads=256)),
            ("FIXED  bn=128 (256x128x128)", dict(bm=256, bn=128, bk=128, threads=128)),
        ):
            kern = build_matmul(gs=64, stages=3, warp_spec=True, **cfg)
            kern(a, b, c, dtype="float8_e4m3fn")
            torch.cuda.synchronize()
            ok, err = verify(c, ref, atol=ATOL, rtol=RTOL)
            d = (c.float() - ref.float()).abs()
            first = err.strip().splitlines()
            first = next((l for l in first if "Mismatched" in l or "Greatest" in l), "")
            print(f"K={K:<6} {tag:<30} verify -> {'PASS' if ok else 'FAIL'}"
                  f"   maxabs {d.max().item():8.2f}  {first.strip()[:70]}")
        print()


if __name__ == "__main__":
    main()
