"""Case selection from an operator's real config (pure functions).

Case ids are indices into ``expand_cases(operator, config)``, the same numbering
``run_bench.py --case-indices`` uses, so every selected case can be replayed with
the production benchmark.
"""
from __future__ import annotations

from tilebench.data.tensors import expand_cases, infer_problem_size

PILOT_OPERATORS = ("matrix_copy", "vector_add", "rmsnorm", "matmul_fp32_fp16_fp8")


def indexed_cases(operator: str, config: dict) -> list[tuple[int, dict]]:
    return list(enumerate(expand_cases(operator, config)))


def _size(operator: str, case: dict) -> float:
    params = {k: v for k, v in case.items() if k not in ("dtype", "block_size")}
    try:
        return float(infer_problem_size(operator, params))
    except Exception:  # noqa: BLE001 - fall back to the product of numeric params
        prod = 1.0
        for v in params.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                prod *= max(1.0, float(v))
        return prod


def by_dtype(operator: str, config: dict) -> dict[str, list[tuple[int, dict]]]:
    out: dict[str, list[tuple[int, dict]]] = {}
    for i, c in indexed_cases(operator, config):
        out.setdefault(c.get("dtype", "fp32"), []).append((i, c))
    for d in out:
        out[d].sort(key=lambda ic: (_size(operator, ic[1]), ic[0]))
    return out


def smoke_cases(operator: str, config: dict) -> list[int]:
    """The smallest case of every configured dtype."""
    return sorted(cases[0][0] for cases in by_dtype(operator, config).values())


def pilot_cases(operator: str, config: dict) -> dict[str, dict[str, int]]:
    """small / medium / large of a half-precision dtype (fp16, else bf16) and of fp32
    when configured. Returns {dtype: {"small": id, "medium": id, "large": id}}."""
    groups = by_dtype(operator, config)
    chosen = {}
    half = next((d for d in ("fp16", "bf16") if d in groups), None)
    for d in ([half] if half else []) + (["fp32"] if "fp32" in groups else []):
        cs = groups[d]
        chosen[d] = {"small": cs[0][0], "medium": cs[len(cs) // 2][0], "large": cs[-1][0]}
    return chosen
