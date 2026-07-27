#!/usr/bin/env python3
"""Dynamic SASS opcode histogram from a --set source .ncu-rep, via ncu --csv.

Weights each SASS line by its "Instructions Executed" column, so this is a
dynamic count, not a static disassembly listing.

Reports uniform-datapath usage separately: U-prefixed opcodes (UIADD3, USHF,
ULOP3, ULDC, UMOV, ...) and instructions taking UR* operands.

    python3 sass_opcodes.py reports/src_a.ncu-rep:tagA reports/src_b.ncu-rep:tagB
"""
import csv
import io
import os
import re
import subprocess
import sys
from collections import Counter

NCU = os.environ.get("NCU", "/opt/nvidia/nsight-compute/2026.1.1/ncu")

UNIFORM_OPCODES = re.compile(r"^U[A-Z]")          # UIADD3, ULOP3, USHF, ULDC...
UR_OPERAND = re.compile(r"\bUR\d+\b|\bURZ\b")


def dump(path):
    out = subprocess.run(
        [NCU, "--import", path, "--page", "source", "--print-source", "sass",
         "--csv"],
        capture_output=True, text=True, timeout=900).stdout
    # skip the leading "Kernel Name" banner line
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith('"Address"'):
            return "\n".join(lines[i:])
    sys.exit(f"no SASS table in {path}")


def parse(path):
    rows = list(csv.DictReader(io.StringIO(dump(path))))
    hist = Counter()
    uniform = 0
    ur_operand = 0
    total = 0
    for r in rows:
        src = (r.get("Source") or "").strip()
        if not src:
            continue
        try:
            n = int((r.get("Instructions Executed") or "0").replace(",", ""))
        except ValueError:
            n = 0
        if n == 0:
            continue
        # strip predicate prefix like "@!P0 "
        src = re.sub(r"^@!?U?P\d+\s+", "", src)
        op = src.split()[0]
        base = op.split(".")[0]
        hist[base] += n
        total += n
        if UNIFORM_OPCODES.match(base):
            uniform += n
        if UR_OPERAND.search(src):
            ur_operand += n
    return hist, total, uniform, ur_operand


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    data = {}
    for a in args:
        path, _, tag = a.rpartition(":")
        if not path:
            path, tag = a, os.path.basename(a)
        print(f"parsing {path} ...", flush=True)
        data[tag] = parse(path)

    tags = list(data)
    w = 16
    print("\n" + "=" * (22 + w * len(tags)))
    print(f"{'':<22}" + "".join(f"{t:>{w}}" for t in tags))
    print(f"{'total inst executed':<22}"
          + "".join(f"{data[t][1]:>{w},}" for t in tags))
    print(f"{'  uniform opcodes':<22}"
          + "".join(f"{data[t][2]:>{w},}" for t in tags))
    print(f"{'  as % of total':<22}"
          + "".join(f"{100*data[t][2]/max(data[t][1],1):>{w-1}.2f}%" for t in tags))
    print(f"{'  any UR* operand':<22}"
          + "".join(f"{data[t][3]:>{w},}" for t in tags))
    print(f"{'  as % of total':<22}"
          + "".join(f"{100*data[t][3]/max(data[t][1],1):>{w-1}.2f}%" for t in tags))

    print("\ntop opcodes by dynamic count")
    keys = set()
    for t in tags:
        keys |= {k for k, _ in data[t][0].most_common(22)}
    print(f"{'opcode':<22}" + "".join(f"{t:>{w}}" for t in tags))
    for k in sorted(keys, key=lambda k: -max(data[t][0][k] for t in tags)):
        print(f"{k:<22}" + "".join(f"{data[t][0][k]:>{w},}" for t in tags))


if __name__ == "__main__":
    main()
