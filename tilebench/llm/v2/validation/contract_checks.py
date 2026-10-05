"""Operator-specific contract evidence from evaluator_rules.json.

evaluator_rules.json (evaluator-only) carries, per operator:

    {
      "operator": ..., "contract_revision": ..., "status": draft|approved,
      "required_stages": [{"id": ..., "description": ...}],
      "forbidden_substitutions": [{"pattern": <regex>, "message": ..., "level": confirmed|suspicious,
                                   "scope": host|kernel|any}],
      "required_evidence": [{"any_of": [<regex>...], "message": ..., "level": suspicious}],
      "allowed_torch_calls": ["torch.empty", "torch.empty_like", ...],
      "mutation": {"inputs_mutated": [...], "restore_required": bool},
      "timing_boundary": {"includes_preprocessing": bool, "notes": ...},
      "tolerance_source": "config.verify"
    }

How a regex rule becomes evidence:
- it is matched against the code-only text (comments and strings blanked);
- `scope` (default `host`) restricts the hit to host code, kernel code or
  either; a host-only pattern never fires inside a @triton.jit / @ct.kernel
  / @T.prim_func body, where `x + y` is tile arithmetic;
- a `confirmed` hit is kept confirmed only when the matched line is a
  computational statement in the AST (call, operator, return, subscript);
  otherwise it is downgraded to `suspicious` with the reason recorded. A
  regex is syntax; the AST corroboration is the structural check that
  keeps a syntax hit from standing in for an algorithmic proof;
- `required_evidence` misses are always suspicious (review_required),
  never confirmed violations; they are searched in comment-stripped text
  that KEEPS string literals (evidence such as float("-inf") is a literal).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tilebench.llm.v2.validation.static_checks import CONFIRMED, SUSPICIOUS, Evidence, StaticReport, analyze, hidden_search_evidence

RULE_SCOPES = ("host", "kernel", "any")
CHECKER_VERSION = "contract_checks/2026-10-05.2"


@dataclass
class ComplianceResult:
    verdict: str                       # clear | review_required | confirmed_violation
    static: StaticReport
    contract_evidence: list[Evidence] = field(default_factory=list)

    def diagnostics(self) -> list[str]:
        items = self.static.confirmed + [e for e in self.contract_evidence if e.level == CONFIRMED]
        return [f"line {e.line}: {e.message}" if e.line else e.message for e in items]

    def review_items(self) -> list[str]:
        items = self.static.suspicious + [e for e in self.contract_evidence if e.level == SUSPICIOUS]
        return [f"line {e.line} [{e.scope or '-'}]: {e.message}" if e.line else e.message for e in items]

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "static": self.static.to_dict(),
                "contract_evidence": [e.__dict__ for e in self.contract_evidence],
                "review_items": self.review_items()}


def _scope_ok(rule_scope: str, in_kernel: bool) -> bool:
    if rule_scope == "any":
        return True
    if rule_scope == "kernel":
        return in_kernel
    return not in_kernel


def check_compliance(source: str, dsl: str, rules: dict) -> ComplianceResult:
    allowed = tuple(rules.get("allowed_torch_calls", ()))
    static = analyze(source, dsl, allowed_torch_calls=allowed)
    static.evidence.extend(hidden_search_evidence(source, static.aliases))
    code = static.code_text
    ev: list[Evidence] = []
    for rule in rules.get("forbidden_substitutions", []):
        scope = rule.get("scope", "host")
        if scope not in RULE_SCOPES:
            raise ValueError(f"rule scope {scope!r} invalid")
        for m in re.finditer(rule["pattern"], code, re.M):
            line = code.count("\n", 0, m.start()) + 1
            in_kernel = line in static.kernel_lines
            if not _scope_ok(scope, in_kernel):
                continue
            level = rule.get("level", SUSPICIOUS)
            message = rule["message"]
            if level == CONFIRMED and line not in static.computational_lines:
                level = SUSPICIOUS
                message += " [regex hit on a non-computational line; downgraded to review]"
            ev.append(Evidence(level, "rule", message, line, "kernel" if in_kernel else "host", m.group(0)[:80]))
    evidence_text = static.code_text_with_strings or code      # literals such as float("-inf") count as evidence
    for rule in rules.get("required_evidence", []):
        if not any(re.search(p, evidence_text, re.M) for p in rule["any_of"]):
            ev.append(Evidence(SUSPICIOUS, "rule", f"missing evidence: {rule['message']}", None, None))
    if static.confirmed or any(e.level == CONFIRMED for e in ev):
        verdict = "confirmed_violation"
    elif static.suspicious or ev:
        verdict = "review_required"
    else:
        verdict = "clear"
    return ComplianceResult(verdict=verdict, static=static, contract_evidence=ev)
