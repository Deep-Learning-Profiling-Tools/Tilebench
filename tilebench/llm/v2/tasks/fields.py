"""Prompt-visible task fields, exported by whitelist from config.yaml and
impl_torch.py. Metrics (flops/bytes expressions, peaks), benchmark settings,
manual results and manual kernel sources are never exported."""
from __future__ import annotations

import ast
from dataclasses import dataclass, field

from tilebench.paths import operator_dir
from tilebench.core.verifier import config_tolerance

# Names that mark non-functional material inside impl_torch.py.
_DROP_DEF_NAMES = {"get_last_config"}
_DROP_NAME_FRAGMENTS = ("autotune", "profil", "benchmark", "detect_arch", "nki", "neuron")


@dataclass
class FunctionalReference:
    source: str                     # the excerpt shown to the model
    dropped_definitions: list[str]  # top-level defs removed, with reason
    kept_definitions: list[str]
    byte_identical_to_file: bool


@dataclass
class TaskFields:
    operator: str
    dtype: str
    torch_dtype: str
    params: dict
    tolerance: dict                   # {atol, rtol, source}
    run_signature: str                # of impl_torch.run
    returns: str                      # free text from the docstring/return analysis
    functional_reference: FunctionalReference
    extra: dict = field(default_factory=dict)


def effective_tolerance(config: dict, dtype: str, arch: str | None) -> dict:
    """The tolerance the evaluator will apply: config `verify:` with
    arch_overrides resolved, else the verifier's per-dtype default."""
    from tilebench.core.dtypes import resolve_dtype
    from tilebench.core.verifier import _TOLERANCES, _DEFAULT_ATOL, _DEFAULT_RTOL

    atol, rtol = config_tolerance(config.get("verify", {}) or {}, arch)
    source = "config.verify" + (f"+arch_overrides[{arch}]" if arch and
                                 (config.get("verify", {}) or {}).get("arch_overrides", {}).get(arch) else "")
    if atol is None or rtol is None:
        d_atol, d_rtol = _TOLERANCES.get(resolve_dtype(dtype), (_DEFAULT_ATOL, _DEFAULT_RTOL))
        atol = atol if atol is not None else d_atol
        rtol = rtol if rtol is not None else d_rtol
        source += "+dtype_default"
    return {"atol": atol, "rtol": rtol, "source": source}


def _torch_source(operator: str) -> str:
    return (operator_dir(operator) / "impl_torch.py").read_text()


def functional_reference(operator: str) -> FunctionalReference:
    """impl_torch.py minus non-functional top-level definitions. Only whole
    top-level definitions are removed; the body of run() is never edited."""
    src = _torch_source(operator)
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    drop: list[tuple[int, int, str]] = []
    kept: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            lowered = name.lower()
            if name in _DROP_DEF_NAMES or any(f in lowered for f in _DROP_NAME_FRAGMENTS):
                start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
                drop.append((start, node.end_lineno, name))
            else:
                kept.append(name)
    if not drop:
        return FunctionalReference(src, [], kept, True)
    out = []
    pos = 0
    for start, end, name in sorted(drop):
        out.extend(lines[pos:start])
        pos = end
    out.extend(lines[pos:])
    text = "".join(out)
    return FunctionalReference(text, [f"{n}: non-functional definition" for _, _, n in drop], kept, False)


def run_signature(operator: str) -> tuple[str, str]:
    src = _torch_source(operator)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            seg = ast.get_source_segment(src, node) or ""
            header = seg.split(":\n", 1)[0] + ":"
            doc = ast.get_docstring(node) or ""
            returns = ast.unparse(node.returns) if node.returns else ""
            return header, (returns or doc.strip().splitlines()[0] if doc else returns)
    raise ValueError(f"{operator}: impl_torch.py has no top-level run()")


def task_fields(operator: str, dtype: str, params: dict, config: dict, arch: str | None) -> TaskFields:
    from tilebench.core.dtypes import resolve_dtype
    sig, returns = run_signature(operator)
    return TaskFields(
        operator=operator, dtype=dtype, torch_dtype=str(resolve_dtype(dtype)),
        params=dict(params), tolerance=effective_tolerance(config, dtype, arch),
        run_signature=sig, returns=returns, functional_reference=functional_reference(operator),
    )


FORBIDDEN_CONFIG_KEYS = ("metrics", "benchmark", "plots", "peak", "flops_expr", "bytes_expr")


def assert_no_forbidden_config(text: str) -> None:
    for key in FORBIDDEN_CONFIG_KEYS:
        if f"{key}:" in text or f"{key}_expr" in text:
            raise AssertionError(f"prompt text leaks config key {key!r}")
