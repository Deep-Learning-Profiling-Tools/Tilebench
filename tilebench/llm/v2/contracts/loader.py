"""Contract access with leak checks and approval gating."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tilebench.llm.v2.contracts.schema import validate_audit, validate_rules
from tilebench.llm.v2.manifests.schema import sha256_text

DATA_DIR = Path(__file__).resolve().parent / "data"


class ContractError(RuntimeError):
    pass


class ContractLeakError(ContractError):
    pass


# Patterns that must never appear in model-visible contract text.
_LEAK_PATTERNS = [
    (re.compile(r"\b(num_warps|num_stages|BLOCK_[A-Z_]*|TILE_[A-Z_]*|occupancy)\s*[=:]\s*\d", re.I), "tunable value"),
    (re.compile(r"\b\d+(\.\d+)?\s*(ms|us|µs|ns)\b"), "latency figure"),
    (re.compile(r"\b(speedup|winner|autotune winner|best config|T_SOL|SOL efficiency)\b", re.I), "scoring/winner term"),
    (re.compile(r"@triton\.jit|@ct\.kernel|@tilelang\.jit|def run\(", re.I), "kernel source"),
]


@dataclass
class ContractBundle:
    operator: str
    model_text: str
    rules: dict
    audit: dict
    status: str
    sha256: str

    @property
    def revision(self) -> int:
        return int(self.audit.get("contract_revision", 0))


def contract_dir(operator: str) -> Path:
    return DATA_DIR / operator


def leak_check(text: str) -> list[str]:
    hits = []
    for pat, label in _LEAK_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append(f"{label}: {m.group(0)!r}")
    return hits


def load_contract(operator: str, *, require_approved: bool = True) -> ContractBundle:
    d = contract_dir(operator)
    paths = {name: d / name for name in ("contract.md", "evaluator_rules.json", "audit.json")}
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        raise ContractError(f"{operator}: missing {missing} under {d}")
    text = paths["contract.md"].read_text()
    rules = json.loads(paths["evaluator_rules.json"].read_text())
    audit = json.loads(paths["audit.json"].read_text())
    errs = validate_audit(audit) + validate_rules(rules)
    if errs:
        raise ContractError(f"{operator}: " + "; ".join(errs))
    if audit["operator"] != operator or rules["operator"] != operator:
        raise ContractError(f"{operator}: operator field mismatch in audit/rules")
    leaks = leak_check(text)
    if leaks:
        raise ContractLeakError(f"{operator}: contract.md leaks " + "; ".join(leaks))
    status = audit["status"]
    if require_approved and status != "approved":
        raise ContractError(f"{operator}: contract status is {status!r}; live runs require approved")
    return ContractBundle(operator=operator, model_text=text, rules=rules, audit=audit, status=status,
                          sha256=sha256_text(text))


def list_contracts() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if (p / "audit.json").exists())


def audit_summary() -> dict:
    out = {"approved": [], "draft": [], "needs-review": [], "invalid": {}}
    for op in list_contracts():
        try:
            c = load_contract(op, require_approved=False)
            out[c.status].append(op)
        except ContractError as e:
            out["invalid"][op] = str(e)
    return out
