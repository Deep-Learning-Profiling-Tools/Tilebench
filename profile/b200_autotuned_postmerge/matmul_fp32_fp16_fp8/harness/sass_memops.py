#!/usr/bin/env python3
"""Full memory / TMA / fence opcode inventory from --set source .ncu-rep files.

Complements sass_opcodes.py, which prints only the top-N union and so hides
low-count-but-decisive opcodes (UTMASTG, STG, MEMBAR). This prints every opcode
that touches memory, regardless of count.

    python3 sass_memops.py reports/src_a.ncu-rep:tagA reports/src_b.ncu-rep:tagB
"""
import csv
import io
import os
import re
import subprocess
import sys
from collections import Counter

NCU = os.environ.get("NCU", "/opt/nvidia/nsight-compute/2026.1.1/ncu")
MEM = re.compile(r"LD|ST|TMA|CP|RED|ATOM|MEMBAR|FENCE|YIELD")

MEANING = {
    "UTMALDG": "TMA bulk tensor load (global->smem)",
    "UTMASTG": "TMA bulk tensor store (smem->global)",
    "STG": "plain global store",
    "LDG": "plain global load",
    "LDGSTS": "cp.async global->smem",
    "LDS": "shared load",
    "STS": "shared store",
    "LDSM": "shared matrix load",
    "LDTM": "tensor-memory load",
    "STTM": "tensor-memory store",
    "LDL": "local load (spill)",
    "STL": "local store (spill)",
    "LDC": "constant load",
    "MEMBAR": "memory barrier",
    "FENCE": "fence",
    "YIELD": "warp yield",
    "UTCBAR": "TMA barrier",
    "USETMAXREG": "register reallocation (warp spec)",
}


def hist(path):
    out = subprocess.run(
        [NCU, "--import", path, "--page", "source", "--print-source", "sass",
         "--csv"], capture_output=True, text=True, timeout=900).stdout
    lines = out.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('"Address"'))
    h = Counter()
    for r in csv.DictReader(io.StringIO("\n".join(lines[i:]))):
        s = (r.get("Source") or "").strip()
        if not s:
            continue
        try:
            n = int((r.get("Instructions Executed") or "0").replace(",", ""))
        except ValueError:
            n = 0
        if n:
            h[re.sub(r"^@!?U?P\d+\s+", "", s).split()[0].split(".")[0]] += n
    return h


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    H = {}
    for a in args:
        path, _, tag = a.rpartition(":")
        if not path:
            path, tag = a, os.path.basename(a)
        print(f"parsing {path} ...", flush=True)
        H[tag] = hist(path)

    tags = list(H)
    keys = sorted(set().union(*[set(h) for h in H.values()]))
    print(f"\n{'opcode':<14}{'meaning':<36}"
          + "".join(f"{t:>13}" for t in tags))
    for k in keys:
        if MEM.search(k):
            print(f"{k:<14}{MEANING.get(k, ''):<36}"
                  + "".join(f"{H[t][k]:>13,}" for t in tags))
    print(f"\ndistinct opcodes: " + str({t: len(H[t]) for t in tags}))


if __name__ == "__main__":
    main()
