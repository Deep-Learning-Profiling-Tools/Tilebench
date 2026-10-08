"""Representative case per (operator, dtype), chosen from the operator's
actually expanded case list, never by combining per-dimension maxima.

Rule (study.yaml case_selection.rule): among the expanded cases whose dtype
equals the task dtype, take the one with the largest
tilebench.data.tensors.infer_problem_size(); ties keep the last one in grid
order. The evidence (every candidate's size) is stored with the selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from tilebench.data.tensors import expand_cases, infer_problem_size
from tilebench.paths import list_operators, operator_config

from tilebench.llm.v2.manifests.schema import canonical_json, sha256_text

FRAMEWORK_CASE_KEYS = ("dtype", "block_size")


@dataclass
class CaseSelection:
    operator: str
    dtype: str
    case_index: int             # index in the expanded case list
    params: dict                # the selected case's parameters (no dtype/block_size)
    problem_size: int
    case_id: str                # sha256(params+dtype)[:16]
    rule: str
    candidates: list[dict] = field(default_factory=list)   # [{case_index, problem_size}]

    def to_dict(self) -> dict:
        return asdict(self)


def load_operator_config(operator: str) -> dict:
    with open(operator_config(operator)) as fh:
        return yaml.safe_load(fh)


def operator_dtypes(config: dict) -> list[str]:
    dtypes: set[str] = set()
    for case in _expanded(config.get("_operator", ""), config):
        dtypes.add(str(case.get("dtype", "fp32")))
    return sorted(dtypes)


def _expanded(operator: str, config: dict) -> list[dict]:
    return expand_cases(operator, config)


def case_params(case: dict) -> dict:
    return {k: v for k, v in case.items() if k not in FRAMEWORK_CASE_KEYS}


def make_case_id(params: dict, dtype: str) -> str:
    return sha256_text(canonical_json({"params": params, "dtype": dtype}))[:16]


def select_representative_case(operator: str, dtype: str, config: dict | None = None) -> CaseSelection:
    config = config if config is not None else load_operator_config(operator)
    cases = _expanded(operator, config)
    candidates = []
    best_idx = None
    best_size = -1
    for idx, case in enumerate(cases):
        if str(case.get("dtype", "fp32")) != dtype:
            continue
        params = case_params(case)
        size = int(infer_problem_size(operator, params))
        candidates.append({"case_index": idx, "problem_size": size})
        if size >= best_size:          # ">=" keeps the LAST maximal case in grid order
            best_size, best_idx = size, idx
    if best_idx is None:
        raise ValueError(f"{operator}: no expanded case has dtype {dtype!r}")
    params = case_params(cases[best_idx])
    return CaseSelection(
        operator=operator, dtype=dtype, case_index=best_idx, params=params,
        problem_size=best_size, case_id=make_case_id(params, dtype),
        rule="largest infer_problem_size() among expanded cases of this dtype; ties keep the last case",
        candidates=candidates,
    )


def all_selections() -> list[CaseSelection]:
    out = []
    for op in list_operators():
        cfg = load_operator_config(op)
        for dtype in sorted({str(c.get("dtype", "fp32")) for c in _expanded(op, cfg)}):
            out.append(select_representative_case(op, dtype, cfg))
    return out
