"""Operator-specific contract evidence from evaluator_rules.json (checker v2).

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

Compliance verdicts (study.yaml `compliance`): clear | audit_only |
review_required | confirmed_violation. Only review_required blocks; a
confirmed violation closes the round; audit_only is evaluated normally and
its flags are kept for audit.

How a regex rule becomes evidence (generic semantics, no operator- or
candidate-specific exceptions):
- it is matched against the code-only text (comments and strings blanked);
- `scope` (default `host`) restricts the hit to host code, kernel code or
  either; a host-only pattern never fires inside a kernel body;
- a `confirmed` rule hit is a confirmed violation only when the matched line
  is a computational statement in the AST (call, operator, return,
  subscript); otherwise it is audit-only (a regex is syntax, not proof);
- a match that spans a line break is not evidence unless the rule declares
  `multiline: true` (a regex `\\s` must not reach into the next statement,
  e.g. a decorator line)                                   -> audit_only;
- a match containing `@` counts as the infix matrix-multiply operator only on
  a line where the AST has a MatMult expression; a decorator (`@ct.kernel`,
  `@triton.jit`, `@T.prim_func`, `@tilelang.jit`) never is -> audit_only;
- a `suspicious` rule hit is review_required (positive evidence of a
  possible forbidden substitution / intermediate state) unless the AST shows
  it cannot be one:
    * every identifier the match falls in is a name the candidate defines
      itself (its own function/kernel/variable/parameter name, e.g. a kernel
      called `_swiglu_kernel` or a tensor called `locks`) or a keyword-
      argument name (`block_shape=`), not an imported library object and not
      on an import line                                    -> audit_only
    * `.contiguous()` guarded by an `is_contiguous()` test   -> audit_only
    * the match contains no identifier (an operator or punctuation token
      such as `<=`, which also guards bounds masks)       -> audit_only
    * the line runs only for degenerate inputs (inside an `if size <= 1:
      ... return` early exit)                              -> audit_only
    * the line is not a computational statement            -> audit_only
    * the match is a cache / tensor-identity construct     -> audit_only here;
      whether it keeps tensor data across calls is decided by the static
      persistent-state data-flow analysis (validation.static_checks), which
      raises review_required when it does;
- `required_evidence` misses are audit_only, never blocking: a missing
  positive pattern (a literal -inf, a specific primitive name, a constant
  written another way) is not evidence of a violation. They are searched in
  comment-stripped text that keeps string literals."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tilebench.llm.v2.validation.static_checks import (AUDIT, CONFIRMED, REVIEW, Evidence, StaticReport, analyze,
                                                       hidden_search_evidence)

RULE_SCOPES = ("host", "kernel", "any")
CHECKER_VERSION = "contract_checks/2026-10-07.v3"
_CACHE_MATCH = re.compile(r"data_ptr|cache|WeakTensorKeyDictionary|WeakKeyDictionary|lru_cache", re.I)
_IDENTIFIER = re.compile(r"[A-Za-z_]")


@dataclass
class ComplianceResult:
    verdict: str                       # clear | audit_only | review_required | confirmed_violation
    static: StaticReport
    contract_evidence: list[Evidence] = field(default_factory=list)

    def _items(self, level: str) -> list[Evidence]:
        return [e for e in self.static.evidence if e.level == level] + [e for e in self.contract_evidence if e.level == level]

    @staticmethod
    def _fmt(e: Evidence) -> str:
        return f"line {e.line} [{e.scope or '-'}]: {e.message}" if e.line else e.message

    def diagnostics(self) -> list[str]:
        return [f"line {e.line}: {e.message}" if e.line else e.message for e in self._items(CONFIRMED)]

    def review_items(self) -> list[str]:
        return [self._fmt(e) for e in self._items(REVIEW)]

    def audit_flags(self) -> list[str]:
        return [self._fmt(e) for e in self._items(AUDIT)]

    def to_dict(self) -> dict:
        return {"verdict": self.verdict, "static": self.static.to_dict(),
                "contract_evidence": [e.__dict__ for e in self.contract_evidence],
                "review_items": self.review_items(), "audit_flags": self.audit_flags(),
                "checker_version": CHECKER_VERSION}


def _scope_ok(rule_scope: str, in_kernel: bool) -> bool:
    if rule_scope == "any":
        return True
    if rule_scope == "kernel":
        return in_kernel
    return not in_kernel


def _offset_to_line_col(code: str, offset: int) -> tuple[int, int]:
    line = code.count("\n", 0, offset) + 1
    return line, offset - (code.rfind("\n", 0, offset) + 1)


def _match_identifiers(code: str, start: int, end: int, static: StaticReport) -> list[str]:
    """NAME tokens of the source that overlap the match span [start, end)."""
    l0, c0 = _offset_to_line_col(code, start)
    l1, c1 = _offset_to_line_col(code, end)
    out = []
    for ln, a, b, name in static.name_tokens:
        if ln < l0 or ln > l1:
            continue
        lo = c0 if ln == l0 else 0
        hi = c1 if ln == l1 else 10 ** 9
        if a < hi and b > lo:
            out.append(name)
    return out


def _structural_audit(rule: dict, code: str, m: re.Match, line: int, static: StaticReport) -> str | None:
    """Generic, AST/token-based reasons a regex hit is NOT evidence (no candidate- or operator-specific list)."""
    matched = m.group(0)
    if "\n" in matched and not rule.get("multiline"):
        return "match spans a line break (the rule is a single-statement pattern)"
    if "@" in matched:
        at_lines = {code.count("\n", 0, m.start() + i) + 1 for i, ch in enumerate(matched) if ch == "@"}
        if not (at_lines & static.matmult_lines):
            return "`@` is not an infix matrix-multiply operator here (decorator or other syntax)"
    if "contiguous" in matched and line in static.guarded_contiguous_lines:
        return "`.contiguous()` guarded by an is_contiguous() test"
    idents = _match_identifiers(code, m.start(), m.end(), static)
    if idents and line not in static.import_lines and \
            all((i in static.defined_names or i in static.keyword_names) and i not in static.imported_names for i in idents) and \
            not re.search(r"[^\w\s]", matched.strip()):
        return (f"the term occurs only inside a name the candidate defines or a keyword-argument name "
                f"({', '.join(sorted(set(idents)))}): a name, not a library call")
    return None


def _classify_suspicious(matched: str, line: int, static: StaticReport) -> tuple[str, str | None]:
    if not _IDENTIFIER.search(matched):
        return AUDIT, "syntax-level match (operator/punctuation token only)"
    if line in static.degenerate_exit_lines:
        return AUDIT, "on a degenerate-input early-exit path (size <= 1), never executed for the task"
    if line not in static.computational_lines:
        return AUDIT, "not a computational statement"
    if _CACHE_MATCH.search(matched):
        return AUDIT, "cache/identity construct: persistence of tensor data is decided by the static data-flow analysis"
    return REVIEW, None


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
            message = rule["message"]
            matched = m.group(0)
            why_not = _structural_audit(rule, code, m, line, static)
            if why_not:
                ev.append(Evidence(AUDIT, "rule", f"{message} [audit only: {why_not}]", line,
                                   "kernel" if in_kernel else "host", matched[:80]))
                continue
            if rule.get("level", "suspicious") == "confirmed":
                if line in static.computational_lines and line not in static.degenerate_exit_lines:
                    level = CONFIRMED
                else:
                    level = AUDIT
                    message += " [regex hit on a non-computational or degenerate-input line; audit only]"
            else:
                level, why = _classify_suspicious(matched, line, static)
                if why:
                    message += f" [audit only: {why}]"
            ev.append(Evidence(level, "rule", message, line, "kernel" if in_kernel else "host", matched[:80]))
    evidence_text = static.code_text_with_strings or code      # literals such as float("-inf") count as evidence
    for rule in rules.get("required_evidence", []):
        if not any(re.search(p, evidence_text, re.M) for p in rule["any_of"]):
            ev.append(Evidence(AUDIT, "rule", f"missing evidence: {rule['message']} [audit only: a missing pattern is not "
                                              "evidence of a violation]", None, None))
    if static.confirmed or any(e.level == CONFIRMED for e in ev):
        verdict = "confirmed_violation"
    elif static.review or any(e.level == REVIEW for e in ev):
        verdict = "review_required"
    elif static.audit or any(e.level == AUDIT for e in ev):
        verdict = "audit_only"
    else:
        verdict = "clear"
    return ComplianceResult(verdict=verdict, static=static, contract_evidence=ev)
