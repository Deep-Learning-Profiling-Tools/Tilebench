"""Operator-specific contract evidence from evaluator_rules.json.

evaluator_rules.json (evaluator-only) carries, per operator:

    {
      "operator": ..., "contract_revision": ..., "status": draft|approved,
      "required_stages": [{"id": ..., "description": ...}],
      "forbidden_substitutions": [{"pattern": <regex>, "message": ..., "level": confirmed|suspicious}],
      "required_evidence": [{"any_of": [<regex>...], "message": ..., "level": suspicious}],
      "allowed_torch_calls": ["torch.empty", "torch.empty_like", ...],
      "mutation": {"inputs_mutated": [...], "restore_required": bool},
      "timing_boundary": {"includes_preprocessing": bool, "notes": ...},
      "tolerance_source": "config.verify"
    }

Regex rules give evidence only; `required_evidence` misses are suspicious
(review_required), never confirmed violations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tilebench.llm.v2.validation.static_checks import CONFIRMED, SUSPICIOUS, Evidence, StaticReport, analyze, hidden_search_evidence


@dataclass
class ComplianceResult:
    verdict: str                       # clear | review_required | confirmed_violation
    static: StaticReport
    contract_evidence: list[Evidence] = field(default_factory=list)

    def diagnostics(self) -> list[str]:
        items = self.static.confirmed + [e for e in self.contract_evidence if e.level == CONFIRMED]
        return [f"line {e.line}: {e.message}" if e.line else e.message for e in items]

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "static": self.static.to_dict(),
                "contract_evidence": [e.__dict__ for e in self.contract_evidence]}


def check_compliance(source: str, dsl: str, rules: dict) -> ComplianceResult:
    allowed = tuple(rules.get("allowed_torch_calls", ()))
    static = analyze(source, dsl, allowed_torch_calls=allowed)
    static.evidence.extend(hidden_search_evidence(source))
    ev: list[Evidence] = []
    for rule in rules.get("forbidden_substitutions", []):
        for m in re.finditer(rule["pattern"], source, re.M):
            line = source.count("\n", 0, m.start()) + 1
            ev.append(Evidence(rule.get("level", SUSPICIOUS), "rule", rule["message"], line))
    for rule in rules.get("required_evidence", []):
        if not any(re.search(p, source, re.M) for p in rule["any_of"]):
            ev.append(Evidence(rule.get("level", SUSPICIOUS), "rule", f"missing evidence: {rule['message']}", None))
    if static.confirmed or any(e.level == CONFIRMED for e in ev):
        verdict = "confirmed_violation"
    elif static.suspicious or ev:
        verdict = "review_required"
    else:
        verdict = "clear"
    return ComplianceResult(verdict=verdict, static=static, contract_evidence=ev)
