"""One-time tool to auto-generate problem descriptions from impl_torch.py.

Writes `benchmarks/problems/current/<op>_current.md` for ops that lack one.
The format mirrors the LeetGPU-style descriptions already present in the repo:
problem statement, input/output spec, mathematical definition, examples,
constraints.

Usage:
    PYTHONPATH=. python tools/llm_codegen/generate_descriptions.py             # all 45 ops
    PYTHONPATH=. python tools/llm_codegen/generate_descriptions.py --operator vector_add
    PYTHONPATH=. python tools/llm_codegen/generate_descriptions.py --force     # overwrite existing
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from tools.llm_codegen.llm_client import LLMClient

PROBLEMS_DIR = _REPO_ROOT / "benchmarks" / "problems" / "current"

_DESC_SYSTEM = """\
You are a technical writer documenting GPU kernel benchmarks. Your job: take the
PyTorch reference implementation of an operator and write a clear problem
description in the LeetGPU style — operator semantics, input/output spec,
math definition, examples, constraints. No implementation details about
Triton or cuTile; just describe WHAT the operator does, not HOW to write it.
"""


_DESC_TEMPLATE = """\
Write a problem description for the operator `{op}` based on the PyTorch reference below.

## Output format (markdown)

```
# {{TitleCase Op Name}}

{{1-2 sentence problem statement describing the math the op performs.}}

The input consists of:
- `{{tensor_name}}`: {{shape, dtype, semantics}}
...

The output should be:
- `{{tensor_name}}`: {{shape, dtype, semantics}}

## Mathematical definition

$$
{{LaTeX formula}}
$$

## Examples

### Example 1
- Input shape: {{small shape}}
- Expected output: {{short tensor or scalar}}

## Constraints
- {{shape ranges from config.yaml's case_grid}}
- {{any algorithmic constraints}}
```

## PyTorch reference: `impl_torch.py`

```python
{impl_torch}
```

## `config.yaml` (for shape ranges)

```yaml
{config}
```

## Reference Triton impl (use only to disambiguate semantics; do NOT mention Triton):

```python
{impl_triton_excerpt}
```

Return ONLY the markdown description (no surrounding code fence, no preamble).
"""


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _make_prompt(op: str) -> str:
    op_dir = _REPO_ROOT / "benchmarks" / "operators" / op
    impl_torch = _read(op_dir / "impl_torch.py")
    config = _read(op_dir / "config.yaml")
    impl_triton = _read(op_dir / "impl_triton.py")
    # Trim impl_triton to first ~80 lines to keep prompt compact.
    triton_lines = impl_triton.splitlines()[:80]
    impl_triton_excerpt = "\n".join(triton_lines)
    return _DESC_TEMPLATE.format(
        op=op, impl_torch=impl_torch, config=config,
        impl_triton_excerpt=impl_triton_excerpt,
    )


def _all_ops() -> list[str]:
    ops_dir = _REPO_ROOT / "benchmarks" / "operators"
    ops = sorted(p.name for p in ops_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
    return ops


def _has_description(op: str) -> bool:
    return (PROBLEMS_DIR / f"{op}_current.md").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", default=None, help="Single op (default: all 45)")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--force", action="store_true", help="Overwrite existing")
    args = ap.parse_args()

    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)
    client = LLMClient(model=args.model, effort="xhigh")

    ops = [args.operator] if args.operator else _all_ops()
    for i, op in enumerate(ops, 1):
        out_path = PROBLEMS_DIR / f"{op}_current.md"
        if out_path.exists() and not args.force:
            print(f"[{i}/{len(ops)}] {op}: skip (exists)")
            continue

        prompt = _make_prompt(op)
        print(f"[{i}/{len(ops)}] {op}: generating...")
        t0 = time.time()
        resp = client.generate(prompt, system=_DESC_SYSTEM)
        elapsed = time.time() - t0

        # Strip surrounding fences / preamble if model added them.
        text = resp.text.strip()
        if text.startswith("```"):
            lines = text.split("\n", 1)
            if len(lines) > 1:
                text = lines[1].rsplit("```", 1)[0].rstrip()
        out_path.write_text(text + "\n")
        print(f"  wrote {len(text)} chars in {elapsed:.1f}s → {out_path}")


if __name__ == "__main__":
    main()
