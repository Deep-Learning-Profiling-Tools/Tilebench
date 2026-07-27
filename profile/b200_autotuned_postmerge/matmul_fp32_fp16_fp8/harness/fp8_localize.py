"""Which tile configs produce a correct fp8 result, and is the wrong result a
permutation of the right one?
"""

import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "benchmarks").is_dir())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tilelang_variants import build_matmul  # noqa: E402

DT = torch.float8_e4m3fn


def main():
    torch.manual_seed(0)
    M = N = 512
    K = 1024
    af = torch.randn(M, K, device="cuda", dtype=torch.float32)
    bf = torch.randn(K, N, device="cuda", dtype=torch.float32)
    a, b = af.to(DT), bf.to(DT)
    ref = a.float().double() @ b.float().double()
    relmax = ref.abs().max()

    print("=== tile sweep (ws=on), fp8 e4m3, M=N=512 K=1024 ===")
    print(f"{'bm x bn x bk  thr':<26}{'rel err':>12}   ok")
    good = []
    for bm, bn, bk, thr in [
        (128, 128, 64, 128), (128, 128, 64, 256),
        (128, 128, 128, 128), (128, 128, 128, 256),
        (128, 256, 64, 256), (256, 128, 64, 256),
        (256, 256, 64, 256), (256, 256, 128, 256),
        (256, 256, 128, 128), (64, 64, 64, 128),
    ]:
        c = torch.empty(M, N, device="cuda", dtype=DT)
        try:
            kern = build_matmul(bm=bm, bn=bn, bk=bk, gs=64, threads=thr,
                                stages=3, warp_spec=True)
            kern(a, b, c, dtype="float8_e4m3fn")
            e = ((c.float().double() - ref).abs().max() / relmax).item()
            tag = "OK" if e < 0.2 else "WRONG"
            if e < 0.2:
                good.append((bm, bn, bk, thr))
            print(f"{f'{bm}x{bn}x{bk}  thr{thr}':<26}{e:12.3e}   {tag}")
        except Exception as ex:
            msg = [l for l in str(ex).strip().splitlines() if l.strip()]
            print(f"{f'{bm}x{bn}x{bk}  thr{thr}':<26}{'':>12}   FAILED "
                  f"{(msg[-1] if msg else '')[:50]}")

    # Is the bad output a permutation of the good one?
    print("\n=== structure of the wrong result (256x256x128) ===")
    c = torch.empty(M, N, device="cuda", dtype=DT)
    kern = build_matmul(bm=256, bn=256, bk=128, gs=64, threads=256,
                        stages=3, warp_spec=True)
    kern(a, b, c, dtype="float8_e4m3fn")
    o = c.float().double()
    rq = ref.float().to(DT).float().double()

    same_mult = torch.sort(o.flatten())[0]
    ref_mult = torch.sort(rq.flatten())[0]
    print(f"sorted-value match (multiset): max diff "
          f"{(same_mult - ref_mult).abs().max().item():.4f}")

    match = (o - rq).abs() < 1e-9
    print(f"elements exactly right: {match.sum().item()} / {match.numel()} "
          f"({100 * match.float().mean().item():.1f}%)")
    rows_ok = match.all(dim=1)
    cols_ok = match.all(dim=0)
    print(f"fully-correct rows: {rows_ok.sum().item()}/{M}   "
          f"fully-correct cols: {cols_ok.sum().item()}/{N}")

    # per 128x128 subtile correctness map
    print("\nper-64x64 subtile % correct (first 8x8 subtiles):")
    for i in range(0, min(512, M), 64):
        row = []
        for j in range(0, min(512, N), 64):
            row.append(f"{100 * match[i:i + 64, j:j + 64].float().mean().item():5.0f}")
        print("  " + " ".join(row))

    # does row i of the output equal row perm(i) of the reference?
    print("\nsearching for a row permutation...")
    hits = 0
    for i in range(0, M, 64):
        d = (rq - o[i]).abs().max(dim=1).values
        j = int(d.argmin())
        if d[j] < 1e-9:
            hits += 1
            if hits <= 6:
                print(f"  out row {i:4d}  ==  ref row {j:4d}")
    print(f"  rows found elsewhere in ref: {hits}/{M // 64} sampled")


if __name__ == "__main__":
    main()
