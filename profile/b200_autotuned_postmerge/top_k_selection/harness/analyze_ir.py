"""Static PTX / SASS instruction-mix analysis for the three top-k backends."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

IR = Path(__file__).resolve().parent.parent / "ir"
OUT = Path(__file__).resolve().parent.parent / "analysis"

PTX_OP = re.compile(r"^\s*(?:@%p\d+\s+)?([a-z][a-z0-9._]*)\s")
SASS_OP = re.compile(r"^\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9._@]*)")

DIV_PTX = ("div.", "rem.")
DIV_SASS = ("MUFU", "I2F", "F2I", "IMAD.HI", "FCHK", "FFMA")


def ptx_hist(path: Path) -> Counter:
    hist: Counter = Counter()
    body = False
    for line in path.read_text().splitlines():
        if ".visible .entry" in line or ".entry " in line:
            body = True
        if line.startswith("\t.section") or ".debug_" in line:
            body = False
        if not body:
            continue
        m = PTX_OP.match(line)
        if not m:
            continue
        op = m.group(1)
        if op.startswith((".", "//")) or op in ("ret",):
            if op != "ret":
                continue
        hist[op] += 1
    return hist


def sass_hist(path: Path) -> Counter:
    hist: Counter = Counter()
    for line in path.read_text(errors="replace").splitlines():
        if ".byte" in line or ".dword" in line or ".word" in line:
            continue
        m = SASS_OP.match(line)
        if m:
            hist[m.group(1)] += 1
    return hist


def base(op: str) -> str:
    return op.split(".")[0]


def render(title: str, hists: dict[str, Counter], top: int = 24) -> str:
    names = list(hists)
    keys = Counter()
    for h in hists.values():
        keys.update(h)
    lines = [f"### {title}", ""]
    width = max(len(k) for k in keys) + 2 if keys else 10
    lines.append("opcode".ljust(width) + "".join(n.rjust(14) for n in names))
    for op, _ in keys.most_common(top):
        lines.append(
            op.ljust(width) + "".join(str(hists[n].get(op, 0)).rjust(14) for n in names)
        )
    lines.append("TOTAL".ljust(width) + "".join(str(sum(hists[n].values())).rjust(14) for n in names))
    lines.append("")
    return "\n".join(lines)


def grouped(hists: dict[str, Counter]) -> dict[str, Counter]:
    out = {}
    for name, h in hists.items():
        g: Counter = Counter()
        for op, c in h.items():
            g[base(op)] += c
        out[name] = g
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    report = []

    for n in (1048576, 4096):
        tag = "high N = 1,048,576" if n == 1048576 else "low N = 4,096"

        ptx = {
            b: ptx_hist(IR / f"{b}_N{n}.ptx")
            for b in ("tilelang", "triton")
            if (IR / f"{b}_N{n}.ptx").exists()
        }
        report.append(render(f"PTX opcode counts, exact ({tag})", ptx, top=40))
        report.append(render(f"PTX opcode counts, grouped ({tag})", grouped(ptx), top=30))

        sass = {
            b: sass_hist(IR / f"{b}_N{n}.sass")
            for b in ("tilelang", "triton", "cutile")
            if (IR / f"{b}_N{n}.sass").exists()
        }
        report.append(render(f"SASS opcode counts, exact ({tag})", sass, top=40))
        report.append(render(f"SASS opcode counts, grouped ({tag})", grouped(sass), top=30))

        # division-related summaries
        lines = [f"### Division-related instruction sites ({tag})", ""]
        for b, h in ptx.items():
            div = {op: c for op, c in h.items() if op.startswith(DIV_PTX)}
            lines.append(f"PTX  {b:9s} {sum(div.values()):3d}  {dict(sorted(div.items()))}")
        for b, h in sass.items():
            div = {op: c for op, c in h.items() if op.startswith(("MUFU", "I2F", "F2I", "FCHK"))}
            lines.append(f"SASS {b:9s} {sum(div.values()):3d}  {dict(sorted(div.items()))}")
        lines.append("")
        report.append("\n".join(lines))

    text = "\n".join(report)
    (OUT / "static_ir_mix.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
