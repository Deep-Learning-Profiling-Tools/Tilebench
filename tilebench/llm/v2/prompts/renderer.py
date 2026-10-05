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

from tilebench.llm.v2.prompts.feedback import diagnostics_block, outcome_text, runtime_history
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
    params: dict
    atol: float
    rtol: float
    tolerance_source: str
    run_signature: str
    returns: str
    functional_reference: str
    fp8_format: str | None = None
    components: list[SkillComponent] = field(default_factory=list)
    contract_text: str = ""
    rounds: int = 10

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
        params_block = "\n".join(f"  - `{k}` = `{v}`" for k, v in self.params.items())
        fp8 = f"; FP8 format `{self.fp8_format}`" if self.fp8_format else ""
        return {
            "operator": self.operator, "dtype": self.dtype, "torch_dtype": self.torch_dtype,
            "dsl": self.dsl, "dsl_version": self.dsl_version, "device": self.device,
            "output_file": self.output_file, "params_block": params_block, "fp8_note": fp8,
            "atol": repr(self.atol), "rtol": repr(self.rtol),
            "tolerance_note": f" (source: {self.tolerance_source})",
            "run_signature": self.run_signature, "returns": self.returns or "the output tensor(s) described in the contract",
            "functional_reference": self.functional_reference.rstrip("\n"),
            "reference_skill": ref.text.rstrip("\n"), "device_context_skill": dev.text.rstrip("\n"),
            "optimization_block": opt_block, "algorithm_contract": self.contract_text.rstrip("\n"),
            "rounds": str(self.rounds),
        }


def render_system(ctx: TaskContext) -> str:
    return render(load_template("system_interface"), ctx.base_values())


def render_initial(ctx: TaskContext) -> str:
    return render(load_template("initial"), ctx.base_values())


def _prev_block(ctx: TaskContext, prev: dict, limits: dict, fallback: dict | None) -> str:
    """The previous-round section. A round that ended without a compliant
    candidate (three contract violations) or without a parsable file is
    described by its outcome only: its rejected code is never shown as a
    starting point; the last compliant implementation (if any) is."""
    status = prev.get("status")
    source = prev.get("source")
    if status == "contract_violation" or source is None:
        head = (f"## Previous round (round {prev['round']}): no compliant implementation\n\n"
                f"Outcome: {outcome_text(prev)}\n{diagnostics_block(prev, limits)}")
        if status == "contract_violation":
            head += ("\nThe candidates of that round were rejected for contract violations and are not shown; "
                     "a rejected candidate must not serve as the basis of your next implementation.\n")
        if fallback is not None and fallback.get("source"):
            head += (f"\n## Last compliant implementation (round {fallback['round']})\n\n"
                     f"```python title=\"{ctx.output_file}\"\n{fallback['source'].rstrip()}\n```\n\n"
                     "You may start from it; it is the most recent candidate that satisfied the contract.\n")
        else:
            head += "\nNo compliant implementation exists yet in this task; start again from the contract.\n"
        return head
    return (f"## Previous candidate (round {prev['round']})\n\n"
            f"```python title=\"{ctx.output_file}\"\n{source.rstrip()}\n```\n\n"
            f"Configuration reported by `get_last_config()`: `{json.dumps(prev.get('config'))}`\n\n"
            f"Outcome: {outcome_text(prev)}\n{diagnostics_block(prev, limits)}")


def render_refinement(ctx: TaskContext, *, round_index: int, prev: dict, best_valid: dict | None,
                      history: list[dict], limits: dict, fallback: dict | None = None) -> str:
    v = ctx.base_values()
    best_block = ""
    if best_valid is not None and best_valid.get("round") != prev.get("round"):
        best_block = (f"\n## Best valid candidate so far (round {best_valid['round']}, "
                      f"{best_valid['latency_ms_mean']:.4f} ms)\n\n```python title=\"{ctx.output_file}\"\n"
                      f"{best_valid['source'].rstrip()}\n```\n\nConfiguration: `{json.dumps(best_valid.get('config'))}`\n")
    v.update({
        "round": str(round_index),
        "prev_block": _prev_block(ctx, prev, limits, fallback),
        "best_valid_block": best_block, "runtime_history": runtime_history(history),
    })
    return render(load_template("refinement"), v)


def render_repair(ctx: TaskContext, *, round_index: int, attempt: int, max_attempts: int,
                  violations: list[str], rejected_source: str, fallback: dict | None) -> str:
    v = ctx.base_values()
    fb = ""
    if fallback is not None:
        fb = (f"## Your earlier compliant implementation (round {fallback['round']})\n\n"
              f"```python title=\"{ctx.output_file}\"\n{fallback['source'].rstrip()}\n```\n\n"
              "You may start from it; it is the last candidate that satisfied the contract.")
    else:
        fb = "No earlier compliant implementation exists in this task; start again from the contract."
    v.update({"round": str(round_index), "attempt": str(attempt), "max_attempts": str(max_attempts),
              "violations": "\n".join(f"- {x}" for x in violations) or "- (see evaluator notice)",
              "rejected_source": rejected_source.rstrip("\n"), "fallback_block": fb})
    return render(load_template("compliance_repair"), v)


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
