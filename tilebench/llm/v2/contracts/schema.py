"""Shapes of the three contract artifacts.

data/<op>/contract.md          model-visible, DSL-neutral (sent to the generator)
data/<op>/evaluator_rules.json evaluator-only (never sent)
data/<op>/audit.json           source audit / provenance (never sent)

audit.json
{
  "schema": "tilebench-contract-audit/1",
  "operator": str, "source_sha": str, "contract_revision": int,
  "status": "draft" | "needs-review" | "approved",
  "approved_by": str|null, "approved_on": str|null,
  "sources": {"triton": {"file": str, "functions": [{"name": str, "lines": [a, b]}]},
              "cutile": {...}, "torch": {...}},
  "aspects": {<aspect>: {"status": "aligned" | "mapping-only" | "needs-review",
                          "triton": str, "cutile": str, "note": str}},
  "fq_boundary": {"bytes_expr": str, "flops_expr": str, "consistent": "true"|"false"|"unclear",
                  "note": str},
  "human_reference_comparable": bool,
  "review_items": [str],
  "contract_sha256": str|null
}
Aspects: functional_semantics, stages, dependencies, reduction_scan_sort,
precision, preprocessing, intermediate_storage, mutation, hardware_dispatch.

evaluator_rules.json
{
  "schema": "tilebench-evaluator-rules/1",
  "operator": str, "contract_revision": int,
  "required_stages": [{"id": str, "description": str}],
  "forbidden_substitutions": [{"pattern": regex, "message": str, "level": "confirmed"|"suspicious"}],
  "required_evidence": [{"any_of": [regex], "message": str, "level": "suspicious"}],
  "allowed_torch_calls": [dotted names],
  "mutation": {"inputs_mutated": [names], "restore_required": bool},
  "outputs": {"structure": "tensor"|"tuple"|"list", "count": int, "aliasing": str},
  "timing_boundary": {"includes_preprocessing": bool, "notes": str},
  "tolerance_source": "config.verify"
}
"""
from __future__ import annotations

AUDIT_SCHEMA = "tilebench-contract-audit/1"
RULES_SCHEMA = "tilebench-evaluator-rules/1"
ASPECTS = ("functional_semantics", "stages", "dependencies", "reduction_scan_sort", "precision",
           "preprocessing", "intermediate_storage", "mutation", "hardware_dispatch")
ASPECT_STATUSES = ("aligned", "mapping-only", "needs-review")
AUDIT_STATUSES = ("draft", "needs-review", "approved")
RULE_LEVELS = ("confirmed", "suspicious")


class ContractSchemaError(ValueError):
    pass


def validate_audit(d: dict) -> list[str]:
    errs = []
    if d.get("schema") != AUDIT_SCHEMA:
        errs.append(f"schema must be {AUDIT_SCHEMA}")
    for key in ("operator", "source_sha", "status", "sources", "aspects", "fq_boundary", "review_items"):
        if key not in d:
            errs.append(f"missing key {key}")
    if d.get("status") not in AUDIT_STATUSES:
        errs.append(f"status {d.get('status')!r} not in {AUDIT_STATUSES}")
    if d.get("status") == "approved" and not d.get("approved_by"):
        errs.append("approved audit needs approved_by")
    aspects = d.get("aspects", {})
    for a in ASPECTS:
        if a not in aspects:
            errs.append(f"missing aspect {a}")
        elif aspects[a].get("status") not in ASPECT_STATUSES:
            errs.append(f"aspect {a}: status {aspects[a].get('status')!r} invalid")
    if any(a.get("status") == "needs-review" for a in aspects.values()) and d.get("status") == "approved":
        errs.append("an approved audit cannot have needs-review aspects")
    if "human_reference_comparable" not in d:
        errs.append("missing human_reference_comparable")
    return errs


def validate_rules(d: dict) -> list[str]:
    errs = []
    if d.get("schema") != RULES_SCHEMA:
        errs.append(f"schema must be {RULES_SCHEMA}")
    for key in ("operator", "required_stages", "forbidden_substitutions", "required_evidence",
                "allowed_torch_calls", "mutation", "outputs", "timing_boundary", "tolerance_source"):
        if key not in d:
            errs.append(f"missing key {key}")
    for r in d.get("forbidden_substitutions", []):
        if r.get("level") not in RULE_LEVELS:
            errs.append(f"forbidden_substitutions entry {r.get('pattern')!r}: level invalid")
    for r in d.get("required_evidence", []):
        if r.get("level", "suspicious") != "suspicious":
            errs.append("required_evidence entries must be level suspicious")
    return errs
