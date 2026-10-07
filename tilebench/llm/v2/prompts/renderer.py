"""Deterministic renderer for the on-disk templates.

Placeholders are `{{name}}`; a missing value is an error (nothing is rendered
silently empty). The renderer composes a prompt from already-loaded skill
components and task fields, so the same function produces the production
request and the test snapshots."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tilebench.llm.v2.prompts.feedback import configs_line, diagnostics_block, outcome_text, runtime_history
from tilebench.llm.v2.skills.loader import SkillComponent

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_PLACEHOLDER = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


class RenderError(KeyError):
    pass


def load_template(name: str) -> str:
    path = TEMPLATE_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text()


def render(template: str, values: dict) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in values:
            raise RenderError(f"template placeholder {{{{{key}}}}} has no value")
        v = values[key]
        return v if isinstance(v, str) else json.dumps(v)
    out = _PLACEHOLDER.sub(sub, template)
    leftover = _PLACEHOLDER.search(out)
    if leftover:
        raise RenderError(f"unrendered placeholder {leftover.group(0)}")
    return out


@dataclass
class TaskContext:
    operator: str
    dtype: str
    torch_dtype: str
    dsl: str
    dsl_version: str
    device: str
    output_file: str
    params: dict                       # revision 4: the parameter DOMAIN of the case suite (tasks.case_sets.domain)
    atol: float
    rtol: float
    tolerance_source: str
    run_signature: str
    returns: str
    functional_reference: str
    fp8_format: str | None = None
    components: list[SkillComponent] = field(default_factory=list)
    contract_text: str = ""
    rounds: int = 5
    n_cases: int = 1

    def component(self, kind: str) -> SkillComponent | None:
        for c in self.components:
            if c.kind == kind:
                return c
        return None

    def base_values(self) -> dict:
        ref = self.component("reference")
        dev = self.component("device")
        opt = self.component("optimization")
        if ref is None or dev is None:
            raise RenderError("reference and device components are required")
        opt_block = (f"# Optimization guidance for {self.dsl} (distilled; fold-specific)\n\n{opt.text}"
                     if opt is not None else "")
        domain_block = render_domain(self.params, self.n_cases)
        fp8 = f"; FP8 format `{self.fp8_format}`" if self.fp8_format else ""
        return {
            "operator": self.operator, "dtype": self.dtype, "torch_dtype": self.torch_dtype,
            "dsl": self.dsl, "dsl_version": self.dsl_version, "device": self.device,
            "output_file": self.output_file, "domain_block": domain_block, "n_cases": str(self.n_cases), "fp8_note": fp8,
            "atol": repr(self.atol), "rtol": repr(self.rtol),
            "tolerance_note": f" (source: {self.tolerance_source})",
            "run_signature": self.run_signature, "returns": self.returns or "the output tensor(s) described in the contract",
            "functional_reference": self.functional_reference.rstrip("\n"),
            "reference_skill": ref.text.rstrip("\n"), "device_context_skill": dev.text.rstrip("\n"),
            "optimization_block": opt_block, "algorithm_contract": self.contract_text.rstrip("\n"),
            "rounds": str(self.rounds),
        }


def render_domain(domain: dict, n_cases: int) -> str:
    """The case suite's parameter domain (constants and integer ranges), never the individual cases."""
    if "constant" not in domain and "varying" not in domain:       # a plain parameter dict (single fixed case)
        lines = [f"  - `{k}` = `{v}`" for k, v in domain.items()]
        return "one fixed case:\n" + "\n".join(lines)
    lines = [f"  - `{k}` = `{v}` in every case" for k, v in (domain.get("constant") or {}).items()]
    for k, v in (domain.get("varying") or {}).items():
        if "min" in v:
            lines.append(f"  - `{k}`: an integer from `{v['min']}` to `{v['max']}`, always a multiple of `{v['multiple_of']}`")
        else:
            lines.append(f"  - `{k}`: varies across cases ({v.get('values_are', 'see the reference')})")
    return (f"{n_cases} configured cases of this dtype, the same set in every round; the file must be correct and is "
            f"timed on each of them. Parameter domain:\n" + "\n".join(lines))


def render_system(ctx: TaskContext) -> str:
    return render(load_template("system_interface"), ctx.base_values())


def render_initial(ctx: TaskContext) -> str:
    return render(load_template("initial"), ctx.base_values())


def _prev_block(ctx: TaskContext, prev: dict, limits: dict, fallback: dict | None) -> str:
    """The previous-round section. A round that ended without a compliant
    candidate (a contract violation) or without a parsable file is described
    by its outcome and sanitized diagnostics only: its rejected code is never
    shown as a starting point; the last compliant implementation (if any) is."""
    status = prev.get("status")
    source = prev.get("source")
    if status == "contract_violation" or source is None:
        head = (f"## Previous round (round {prev['round']}): no compliant implementation\n\n"
                f"Outcome: {outcome_text(prev)}\n{diagnostics_block(prev, limits)}")
        if status == "contract_violation":
            head += ("\nThe candidate of that round was rejected for a contract violation before evaluation and is not "
                     "shown; a rejected candidate must not serve as the basis of your next implementation.\n")
        if fallback is not None and fallback.get("source"):
            head += (f"\n## Last compliant implementation (round {fallback['round']})\n\n"
                     f"```python title=\"{ctx.output_file}\"\n{fallback['source'].rstrip()}\n```\n\n"
                     "You may start from it; it is the most recent candidate that satisfied the contract.\n")
        else:
            head += "\nNo compliant implementation exists yet in this task; start again from the contract.\n"
        return head
    return (f"## Previous candidate (round {prev['round']})\n\n"
            f"```python title=\"{ctx.output_file}\"\n{source.rstrip()}\n```\n\n"
            f"{configs_line(prev.get('configs_distinct'))}\n\n"
            f"Outcome: {outcome_text(prev)}\n{diagnostics_block(prev, limits)}")


def render_refinement(ctx: TaskContext, *, round_index: int, prev: dict, best_valid: dict | None,
                      history: list[dict], limits: dict, fallback: dict | None = None) -> str:
    v = ctx.base_values()
    best_block = ""
    if best_valid is not None and best_valid.get("round") != prev.get("round"):
        best_block = (f"\n## Best valid candidate so far (round {best_valid['round']}, geometric mean "
                      f"{best_valid['latency_ms_geomean']:.4f} ms over the {ctx.n_cases} cases)\n\n```python title=\"{ctx.output_file}\"\n"
                      f"{best_valid['source'].rstrip()}\n```\n\n{configs_line(best_valid.get('configs_distinct'))}\n")
    v.update({
        "round": str(round_index),
        "prev_block": _prev_block(ctx, prev, limits, fallback),
        "best_valid_block": best_block, "runtime_history": runtime_history(history),
    })
    return render(load_template("refinement"), v)


def render_distill_extraction(values: dict) -> str:
    return render(load_template("distill_evidence_extraction"), values)


def render_distill_synthesis(values: dict) -> str:
    return render(load_template("distill_synthesis"), values)


def render_reviewer(values: dict) -> str:
    return render(load_template("reviewer"), values)


def render_dev_extraction(values: dict) -> str:
    return render(load_template("dev_contract_extraction"), values)


def render_dev_reconciliation(values: dict) -> str:
    return render(load_template("dev_contract_reconciliation"), values)


def templates_sha256(names: tuple[str, ...] | None = None) -> str:
    """Hash of the actual template contents (not their names)."""
    import hashlib
    names = names or tuple(sorted(p.stem for p in TEMPLATE_DIR.glob("*.md")))
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode())
        h.update(b"\0")
        h.update(load_template(n).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()
