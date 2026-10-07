"""Stored PyTorch baselines per case (reporting speedup; never shown to a generator).

    speedup_ij = T_torch_j / T_candidate_ij,   round speedup = geomean_j speedup_ij

T_torch_j comes from the device's committed default CSV
(results/<device>/csv/<op>_default.csv, column torch_ms), matched OFFLINE to
the frozen case suite by the CSV's params label (scripts/run_bench.py label
semantics: the varying keys, or n=<problem_size> when no key varies). Torch is
never re-measured here. A case without exactly one matching row has no
baseline (`missing`), and the round speedup is then undefined, never imputed.

The matched table is frozen as artifacts/llm_v2/baselines/<device>/
torch_baseline_<device>_<tag>.json with its source CSV hashes; `load()`
verifies the file's seal."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from tilebench.paths import REPO_ROOT

from tilebench.llm.v2.manifests.schema import canonical_json, sha256_text

SCHEMA = "tilebench-llm-v2-torch-baseline/1"
BASELINE_DIR = REPO_ROOT / "artifacts" / "llm_v2" / "baselines"
DTYPE_ALIASES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def parse_label(label: str) -> dict | None:
    """'k=v, k=v' split at ', <key>=' so list values containing commas stay intact."""
    pos = list(re.finditer(r"(?:^|, )([A-Za-z_][A-Za-z0-9_]*)=", label))
    if not pos or pos[0].start() != 0:
        return None
    out = {}
    for i, m in enumerate(pos):
        end = pos[i + 1].start() if i + 1 < len(pos) else len(label)
        out[m.group(1)] = label[m.end():end].strip()
    return out


def _value_matches(actual, expected: str) -> bool:
    if str(actual) == expected or str(actual).replace(" ", "") == expected.replace(" ", ""):
        return True
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return False


def label_matches(label: str, params: dict, problem_size) -> bool:
    parsed = parse_label(label)
    if not parsed:
        return False
    for k, v in parsed.items():
        if k == "n" and k not in params:
            actual = problem_size
        elif k in params:
            actual = params[k]
        else:
            return False
        if not _value_matches(actual, v):
            return False
    return True


def match_csv(csv_path: Path, dtype: str, cases: list[dict]) -> dict:
    """{case_id: {torch_ms, label, dtype_label} | {missing: reason}} for one operator."""
    rows = list(csv.DictReader(csv_path.open(newline=""))) if csv_path.exists() else []
    label = dtype
    sel = [r for r in rows if r.get("dtype") == dtype]
    if not sel and DTYPE_ALIASES.get(dtype) and any(r.get("dtype") == DTYPE_ALIASES[dtype] for r in rows):
        label = DTYPE_ALIASES[dtype]
        sel = [r for r in rows if r.get("dtype") == label]
    out = {}
    for c in cases:
        hits = [r for r in sel if label_matches(r["params"], c["params"], c.get("problem_size"))]
        if len(hits) != 1:
            out[c["case_id"]] = {"missing": f"{len(hits)} matching rows in {csv_path.name}"}
            continue
        try:
            t = float(hits[0]["torch_ms"])
        except (TypeError, ValueError):
            out[c["case_id"]] = {"missing": f"torch_ms {hits[0].get('torch_ms')!r} is not a number"}
            continue
        if not (t > 0 and t == t):
            out[c["case_id"]] = {"missing": f"torch_ms {t} is not finite positive"}
            continue
        out[c["case_id"]] = {"torch_ms": t, "label": hits[0]["params"], "dtype_label": label}
    return out


def build(device: str, case_sets: dict, *, csv_root: Path | None = None, commit: str | None = None) -> dict:
    csv_root = csv_root or (REPO_ROOT / "results" / device / "csv")
    ops, sources = {}, {}
    for op, entry in sorted(case_sets["operators"].items()):
        path = csv_root / f"{op}_default.csv"
        ops[op] = {"dtype": entry["dtype"], "case_set_id": entry["case_set_id"],
                   "cases": match_csv(path, entry["dtype"], entry["cases"])}
        sources[op] = {"csv": str(path.relative_to(REPO_ROOT)) if path.exists() else None,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
    body = {"schema": SCHEMA, "device": device, "source": "results/<device>/csv/<op>_default.csv torch_ms (not re-measured)",
            "source_commit": commit, "source_csvs": sources, "operators": ops}
    body["sha256"] = sha256_text(canonical_json({k: v for k, v in body.items() if k != "sha256"}))
    return body


def path_for(device: str, tag: str) -> Path:
    return BASELINE_DIR / device / f"torch_baseline_{device}_{tag}.json"


def load(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema must be {SCHEMA}")
    seal = sha256_text(canonical_json({k: v for k, v in data.items() if k != "sha256"}))
    if seal != data.get("sha256"):
        raise ValueError(f"{path}: content does not match its sha256 seal")
    return data


def torch_ms(baseline: dict, operator: str) -> dict[str, float] | None:
    """{case_id: torch_ms} of an operator when EVERY case has a baseline; None otherwise."""
    cases = ((baseline.get("operators") or {}).get(operator) or {}).get("cases") or {}
    if not cases or any("torch_ms" not in v for v in cases.values()):
        return None
    return {cid: v["torch_ms"] for cid, v in cases.items()}
