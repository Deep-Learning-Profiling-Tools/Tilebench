"""Task eligibility, registered before generation and held fixed.

A task is (operator, dtype, case_id, device, dsl); the trajectory adds model
and condition. Status values:

- eligible      : in the support matrix and known to run (torch reference +
                  input generation succeed on that device)
- unsupported   : predeclared, with a reason; reported separately, never
                  scored, never the outcome of an ordinary generation failure
- needs_review  : no evidence either way on that device yet (e.g. Trn2
                  before the NKI adapter has been validated there)

Known device facts come from the archived manual campaigns and are cited in
`reason`; nothing here is inferred from a generation attempt."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from tilebench.paths import list_operators

from tilebench.llm.v2.manifests.schema import DEVICES, fold_of
from tilebench.llm.v2.tasks.case_selection import (
    load_operator_config, operator_dtypes, select_representative_case,
)

FP8_FORMATS = {"fp8_e4m3fn": "float8_e4m3fn", "fp8_e5m2": "float8_e5m2"}

# (device, dtype) pairs the manual campaigns established as unsupported.
_KNOWN_UNSUPPORTED = {
    ("MI300X", "fp8_e4m3fn"): (
        "torch reference rejects float8_e4m3fn on gfx942 (manual MI300X campaign, "
        "results commit 5e714117: 20 matmul_fp32_fp16_fp8 cases UNSUPPORTED_DTYPE)"),
}


@dataclass
class TaskKey:
    operator: str
    dtype: str
    case_id: str
    device: str
    dsl: str

    def as_str(self) -> str:
        return f"{self.device}/{self.dsl}/{self.operator}/{self.dtype}/{self.case_id}"


@dataclass
class Eligibility:
    key: TaskKey
    fold: str
    status: str                 # eligible | unsupported | needs_review
    reason: str
    fp8_format: str | None
    params: dict
    problem_size: int
    case_index: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = asdict(self.key)
        return d


def eligibility(operator: str, dtype: str, device: str, dsl: str, study: dict, folds: dict,
                config: dict | None = None) -> Eligibility:
    if device not in DEVICES:
        raise ValueError(f"unknown device {device}")
    sel = select_representative_case(operator, dtype, config)
    key = TaskKey(operator, dtype, sel.case_id, device, dsl)
    fold = fold_of(folds, operator)
    fp8 = FP8_FORMATS.get(dtype)
    matrix = study["support_matrix"][device]
    if dsl not in matrix["dsls"]:
        status, reason = "unsupported", f"{dsl} is not in the support matrix of {device}"
    elif (device, dtype) in _KNOWN_UNSUPPORTED:
        status, reason = "unsupported", _KNOWN_UNSUPPORTED[(device, dtype)]
    elif device == "Trn2":
        status, reason = "needs_review", "NKI adapter and Trn2 eligibility not yet validated on the device"
    elif device in ("GH200", "MI300X"):
        status, reason = "eligible", f"manual {device} campaign ran this operator/dtype (results/{device}/csv)"
    else:
        status, reason = "eligible", "manual B200 campaign ran this operator/dtype (results/B200/csv)"
    return Eligibility(key=key, fold=fold, status=status, reason=reason, fp8_format=fp8,
                       params=sel.params, problem_size=sel.problem_size, case_index=sel.case_index)


def task_table(study: dict, folds: dict, operators: list[str] | None = None) -> list[Eligibility]:
    """Every (device, dsl, operator, dtype) of the support matrix with its status."""
    out: list[Eligibility] = []
    for op in operators or list_operators():
        cfg = load_operator_config(op)
        cfg["_operator"] = op
        for dtype in operator_dtypes(cfg):
            for device, entry in study["support_matrix"].items():
                for dsl in entry["dsls"]:
                    out.append(eligibility(op, dtype, device, dsl, study, folds, cfg))
    return out


def summarize(table: list[Eligibility]) -> dict:
    summary: dict = {}
    for e in table:
        bucket = summary.setdefault(e.key.device, {}).setdefault(e.key.dsl, {})
        bucket[e.status] = bucket.get(e.status, 0) + 1
    return summary
