"""Which TCGEN5MMA atom does TileLang select for fp8 bn=256 (broken) vs
bn=128 (correct) vs fp16 bn=256 (correct)? Dumps the generated CUDA and greps
the tcgen05 instantiation.
"""

import re
import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tilelang_variants import build_matmul  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "ir"
OUT.mkdir(exist_ok=True)

CASES = [
    ("fp8_bn256_BROKEN", torch.float8_e4m3fn, "float8_e4m3fn", dict(bm=256, bn=256, bk=128)),
    ("fp8_bn128_ok", torch.float8_e4m3fn, "float8_e4m3fn", dict(bm=256, bn=128, bk=128)),
    ("fp16_bn256_ok", torch.float16, "float16", dict(bm=256, bn=256, bk=64)),
]

PAT = re.compile(r"(tl::gemm_ts|tcgen05|TCGEN5|utccp|UMMA|tmem|num_cta|CtaGroup|"
                 r"tl::gemm_sp|gemm_ss|kind::)", re.I)


def main():
    M = N = 512
    K = 1024
    torch.manual_seed(0)
    for tag, dt, name, cfg in CASES:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32).to(dt)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32).to(dt)
        kern = build_matmul(gs=64, threads=256, stages=3, warp_spec=True, **cfg)
        c = torch.empty(M, N, device="cuda", dtype=dt)
        kern(a, b, c, dtype=name)
        src = kern.compile(a, b, c, dtype=name).get_kernel_source()
        p = OUT / f"atom_{tag}.cu"
        p.write_text(src)
        print(f"\n===== {tag}  ({cfg['bm']}x{cfg['bn']}x{cfg['bk']}) -> {p.name} =====")
        seen = set()
        for line in src.splitlines():
            if PAT.search(line):
                s = line.strip()
                if s not in seen and len(s) < 300:
                    seen.add(s)
                    print("  " + s)


if __name__ == "__main__":
    main()
