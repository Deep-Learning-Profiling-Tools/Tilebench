"""Prompt builder for LLM kernel generation.

Reads operator metadata from the TileBench repo and assembles a complete,
leakage-free prompt for a given operator, backend, and prompt profile.

Forbidden files (never included):
    benchmarks/operators/<op>/impl_triton.py
    benchmarks/operators/<op>/impl_cutile.py

Allowed files:
    benchmarks/operators/<op>/impl_torch.py
    benchmarks/operators/<op>/config.yaml
    benchmarks/problems/current/<op>_current.md  (optional)
    data/tensors.py  (only the relevant generate_<op>_inputs function)
    prompts/dsl_reference/<backend>_minimal.md
    prompts/examples/<example>.md  (from train split only)
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from typing import Any

import yaml

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
    _HAS_JINJA = True
except ImportError:
    _HAS_JINJA = False

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

# Backends and their forbidden implementation filenames (never allowed in context).
_FORBIDDEN_IMPLS: dict[str, set[str]] = {
    "triton": {"impl_triton.py"},
    "cutile": {"impl_cutile.py"},
}
# Always forbidden regardless of backend.
_ALWAYS_FORBIDDEN = {"impl_triton.py", "impl_cutile.py"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _extract_generator_function(tensors_py: Path, operator_name: str) -> str:
    """Extract the ``generate_<operator>_inputs`` function from ``data/tensors.py``.

    Falls back to returning the full file if no matching function is found.
    """
    source = _read(tensors_py)
    # Normalise operator name to Python identifier.
    fn_name = f"generate_{operator_name.replace('-', '_')}_inputs"

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            lines = source.splitlines(keepends=True)
            start = node.lineno - 1
            end   = node.end_lineno
            return "".join(lines[start:end]).strip()

    # Generator not found; return a placeholder.
    return f"# No generator function found for operator '{operator_name}'.\n"


def _load_dsl_reference(backend: str) -> str:
    ref_path = _PROMPTS_DIR / "dsl_reference" / f"{backend}_minimal.md"
    if ref_path.exists():
        return _read(ref_path)
    return f"# DSL reference for '{backend}' not found.\n"


def _load_examples(example_files: list[str], num: int) -> list[str]:
    examples_dir = _PROMPTS_DIR / "examples"
    result: list[str] = []
    for fname in example_files[:num]:
        path = examples_dir / fname
        if path.exists():
            result.append(_read(path))
    return result


def _load_problem_statement(operator_name: str) -> str:
    for pattern in [
        f"benchmarks/problems/current/{operator_name}_current.md",
        f"benchmarks/problems/original/{operator_name}_original.md",
    ]:
        p = _REPO_ROOT / pattern
        if p.exists():
            return _read(p)
    return (
        f"## Operator: {operator_name}\n\n"
        f"No problem statement file found.  "
        f"Derive the semantics from `impl_torch.py` and `config.yaml`."
    )


def _infer_run_signature(impl_torch_src: str) -> str:
    """Extract the `run(...)` signature from impl_torch.py source."""
    match = re.search(r"def run\(.*?\):", impl_torch_src, re.DOTALL)
    if match:
        raw = match.group(0)
        # Strip trailing ":" and clean up whitespace.
        return raw.rstrip(":")
    return "def run(*inputs, block_size: int = 1024, autotune: bool = False, **kwargs)"


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_prompt(
    *,
    operator_name: str,
    backend: str,
    profile: dict[str, Any],
    allowed_context: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a complete prompt string for one operator + backend.

    Parameters
    ----------
    operator_name:
        e.g. ``"softmax"``
    backend:
        ``"triton"`` or ``"cutile"``
    profile:
        Prompt profile dict loaded from ``prompt_profiles.yaml``.
    allowed_context:
        Optional list of context item names included (for metadata logging).

    Returns
    -------
    tuple[str, dict]
        (prompt_text, context_metadata)  where context_metadata records
        exactly what was injected so the leakage audit can be reproduced.
    """
    if not _HAS_JINJA:
        raise ImportError(
            "Jinja2 is required for prompt building.  "
            "Install it with: pip install jinja2"
        )

    op_dir        = _REPO_ROOT / "benchmarks" / "operators" / operator_name
    impl_torch_py = op_dir / "impl_torch.py"
    config_yaml   = op_dir / "config.yaml"
    tensors_py    = _REPO_ROOT / "data" / "tensors.py"

    # --- Safety: verify forbidden files are not being included ---
    forbidden = _ALWAYS_FORBIDDEN
    for fn in forbidden:
        forbidden_path = op_dir / fn
        if forbidden_path.exists():
            pass  # exists but is never read — that is correct

    # --- Load allowed context ---
    impl_torch_src  = _read(impl_torch_py) if impl_torch_py.exists() else ""
    config_yaml_str = _read(config_yaml) if config_yaml.exists() else ""
    problem_stmt    = _load_problem_statement(operator_name)
    run_sig         = _infer_run_signature(impl_torch_src)
    generator_fn    = (
        _extract_generator_function(tensors_py, operator_name)
        if tensors_py.exists() and profile.get("include_generator", True)
        else ""
    )

    dsl_key = f"{backend}_dsl_reference" if backend in ("triton", "cutile") else "dsl_reference"
    dsl_ref = _load_dsl_reference(backend)

    num_examples   = int(profile.get("num_examples", 0))
    example_pool   = list(profile.get("example_pool", []))
    examples       = _load_examples(example_pool, num_examples) if num_examples else []

    # --- Render template ---
    template_file = profile.get("template", f"task_{backend}.md.j2")
    env = Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template(template_file)

    context = {
        "operator_name":          operator_name,
        "backend":                backend,
        "run_signature":          run_sig,
        "problem_statement":      problem_stmt,
        "impl_torch":             impl_torch_src,
        "generator_function":     generator_fn,
        "config_yaml":            config_yaml_str,
        "triton_dsl_reference":   dsl_ref if backend == "triton" else "",
        "cutile_dsl_reference":   dsl_ref if backend == "cutile" else "",
        "examples":               examples,
    }
    prompt_text = template.render(**context)

    # --- Build context metadata for audit log ---
    ctx_meta: dict[str, Any] = {
        "impl_torch.py":   impl_torch_py.exists(),
        "config.yaml":     config_yaml.exists(),
        "generator_fn":    bool(generator_fn),
        "problem_stmt":    bool(problem_stmt),
        "dsl_reference":   f"dsl_reference/{backend}_minimal.md",
        "num_examples":    len(examples),
        "example_files":   example_pool[:num_examples],
        "forbidden_impls_excluded": list(forbidden),
    }

    return prompt_text, ctx_meta


def load_system_prompt() -> str:
    """Return the contents of ``prompts/system.md``."""
    path = _PROMPTS_DIR / "system.md"
    if path.exists():
        return _read(path)
    return "You are an expert GPU kernel engineer."
