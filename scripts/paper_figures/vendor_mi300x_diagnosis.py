"""Freeze the 2026-10-05 MI300X per-operator diagnosis into a tracked JSON file.

The diagnosis was written by hand in the git-ignored analysis directory
(outputs/profiling/MI300X/analysis_2026-10-05/scripts/{diag.py,diag_evidence.py}). This script
copies those records verbatim (no re-wording, no re-classification) together with the SHA256 of
both source files, so that extract_mi300x.py and the validation do not depend on the
git-ignored directory being present.

    python scripts/paper_figures/vendor_mi300x_diagnosis.py \
        --analysis-dir /root/Tilebench/outputs/profiling/MI300X/analysis_2026-10-05
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts" / "paper_figures" / "data" / "mi300x_diagnosis_2026-10-05.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", type=Path,
                    default=Path("/root/Tilebench/outputs/profiling/MI300X/analysis_2026-10-05"))
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    sd = a.analysis_dir / "scripts"
    diag = _load(sd / "diag.py", "_mi300x_diag")
    ev = _load(sd / "diag_evidence.py", "_mi300x_diag_evidence")
    ops = {}
    for op in sorted(diag.D):
        d = dict(diag.D[op])
        b200 = d.pop("b200")
        evidence, alternative = ev.EV[op]
        ops[op] = {"b200_table5": {"primary_diagnosis": b200[0], "pytorch_path": b200[1], "Q_o": b200[2]},
                   **d, "evidence": evidence, "alternative_explanation": alternative}
    out = {
        "schema": "tilearena-mi300x-diagnosis/1",
        "description": "Hand-written per-operator MI300X diagnosis from the 2026-10-05 analysis, copied verbatim. "
                       "Field meanings are given in the docstring of the source diag.py (copied below).",
        "source_files": {
            f"outputs/profiling/MI300X/analysis_2026-10-05/scripts/{n}": hashlib.sha256((sd / n).read_bytes()).hexdigest()
            for n in ("diag.py", "diag_evidence.py")
        },
        "source_docstring": diag.__doc__,
        "clusters": diag.CLUSTERS,
        "relationship_classes": diag.REL,
        "operators": ops,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {a.out} ({len(ops)} operators)")


if __name__ == "__main__":
    main()
