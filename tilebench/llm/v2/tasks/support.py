"""Task eligibility, registered before generation and held fixed.

A task is (operator, dtype, case_id, device, dsl); the trajectory adds model
and condition. Under protocol revision 4 the task's `case_id` is the frozen
CASE-SET id (manifests/case_sets.yaml): every candidate is evaluated on all 20
configured cases of the selected dtype (`cases`); `params` keeps the largest
(representative) case for reference only. Status values:

- eligible      : in the support matrix and known to run (torch reference +
                  input generation succeed on that device)
- unsupported   : predeclared, with a reason; reported separately, never
                  scored, never the outcome of an ordinary generation failure
- needs_review  : no evidence either way on that device yet (e.g. Trn2
                  before the NKI adapter has been validated there)

Trn2 reads the frozen manifests/trn2_eligibility.json (36 eligible / 9 excluded
under revision 4): an operator is eligible only when all 20 cases of its
representative dtype have a valid formal native-PyTorch baseline and a supported
empirical SOL target; an excluded operator is `unsupported` with the manifest's
exclusion class and reason.

Known device facts come from the archived manual campaigns and are cited in
`reason`; nothing here is inferred from a generation attempt."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
    cases: list = field(default_factory=list)      # revision 4: every case the candidate is evaluated on
    case_set_id: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = asdict(self.key)
        return d


def _case_set(operator: str, dtype: str, study: dict, config: dict | None) -> tuple[str | None, list]:
    """(case_set_id, cases) under the all-configured-cases task unit; (None, []) otherwise. The frozen
    manifest is used for the selected dtype; another dtype (inventory tools only) is expanded directly."""
    if (study.get("task_unit") or {}).get("cases") != "all_configured":
        return None, []
    from tilebench.llm.v2.tasks import case_sets
    entry = (case_sets.load_manifest()["operators"] or {}).get(operator)
    if entry is not None and entry["dtype"] == dtype:
        return entry["case_set_id"], [dict(c) for c in entry["cases"]]
    cases = case_sets.operator_cases(operator, dtype, config)
    return case_sets.case_set_id(operator, dtype, [c["case_id"] for c in cases]), cases


TRN2_ELIGIBILITY_PATH = Path(__file__).resolve().parents[1] / "manifests" / "trn2_eligibility.json"
TRN2_ELIGIBILITY_SCHEMA = "tilebench-llm-v2-trn2-eligibility/2"
_TRN2_CACHE: dict = {}


def load_trn2_eligibility(path: Path | None = None) -> dict | None:
    """The frozen Trn2 eligibility manifest (scripts/llm_v2/build_trn2_eligibility.py), seal-checked; None when
    absent. Built from results/TRN2/csv and the registered Trn2 profile: an operator is eligible only with 20/20 valid
    formal baseline cases of its representative dtype and a supported empirical SOL target on every case."""
    p = Path(path or TRN2_ELIGIBILITY_PATH)
    key = str(p)
    if key in _TRN2_CACHE:
        return _TRN2_CACHE[key]
    if not p.exists():
        return None
    from tilebench.llm.v2.manifests.schema import canonical_json, sha256_text
    data = json.loads(p.read_text())
    if data.get("schema") != TRN2_ELIGIBILITY_SCHEMA:
        raise ValueError(f"{p}: schema must be {TRN2_ELIGIBILITY_SCHEMA}")
    if sha256_text(canonical_json({k: v for k, v in data.items() if k != "sha256"})) != data.get("sha256"):
        raise ValueError(f"{p}: content does not match its sha256 seal")
    _TRN2_CACHE[key] = data
    return data


def trn2_status(operator: str, dtype: str, case_set_id: str | None) -> tuple[str, str]:
    """Trn2 status from the frozen eligibility manifest. Only the operator's frozen (dtype, case set) is ever
    eligible; an excluded operator is `unsupported` with the manifest's reason; another dtype is never substituted."""
    data = load_trn2_eligibility()
    if data is None:
        return "needs_review", "NKI adapter and Trn2 eligibility not yet validated on the device"
    entry = (data.get("operators") or {}).get(operator)
    if entry is None:
        return "needs_review", f"{operator} is not in the frozen Trn2 eligibility manifest"
    if entry["dtype"] != dtype:
        return "unsupported", (f"Trn2 eligibility is frozen for the representative dtype {entry['dtype']} only "
                               f"(trn2_eligibility.json); {dtype} is not a Trn2 task")
    if case_set_id is not None and entry["case_set_id"] != case_set_id:
        raise ValueError(f"{operator}: trn2_eligibility.json case set {entry['case_set_id']} != frozen case set {case_set_id}")
    if entry["status"] == "eligible":
        return "eligible", f"Trn2 eligibility manifest: {entry['reason']}"
    return "unsupported", f"excluded by the frozen Trn2 eligibility manifest ({entry['exclusion_class']}): {entry['reason']}"


def eligibility(operator: str, dtype: str, device: str, dsl: str, study: dict, folds: dict,
                config: dict | None = None) -> Eligibility:
    if device not in DEVICES:
        raise ValueError(f"unknown device {device}")
    sel = select_representative_case(operator, dtype, config)
    cs_id, cases = _case_set(operator, dtype, study, config)
    key = TaskKey(operator, dtype, cs_id or sel.case_id, device, dsl)
    fold = fold_of(folds, operator)
    fp8 = FP8_FORMATS.get(dtype)
    matrix = study["support_matrix"][device]
    if dsl not in matrix["dsls"]:
        status, reason = "unsupported", f"{dsl} is not in the support matrix of {device}"
    elif (device, dtype) in _KNOWN_UNSUPPORTED:
        status, reason = "unsupported", _KNOWN_UNSUPPORTED[(device, dtype)]
    elif device == "Trn2":
        status, reason = trn2_status(operator, dtype, cs_id)
    elif device in ("GH200", "MI300X"):
        status, reason = "eligible", f"manual {device} campaign ran this operator/dtype (results/{device}/csv)"
    else:
        status, reason = "eligible", "manual B200 campaign ran this operator/dtype (results/B200/csv)"
    return Eligibility(key=key, fold=fold, status=status, reason=reason, fp8_format=fp8,
                       params=sel.params, problem_size=sel.problem_size, case_index=sel.case_index,
                       cases=cases, case_set_id=cs_id)


def selected_dtypes(study: dict) -> dict[str, str] | None:
    """operator -> the predeclared representative dtype when the study's task
    unit is `representative_dtype` (protocol revision 3); None otherwise."""
    if (study.get("task_unit") or {}).get("kind") == "representative_dtype":
        from tilebench.llm.v2.tasks.representative import selected
        return selected()
    return None


def task_table(study: dict, folds: dict, operators: list[str] | None = None, *,
               all_dtypes: bool = False) -> list[Eligibility]:
    """Every (device, dsl, operator, dtype) of the support matrix with its
    status. Under the representative-dtype task unit only the operator's
    selected dtype is a task; `all_dtypes=True` lists every configured dtype
    (inventory tools only, never a formal task list)."""
    chosen = None if all_dtypes else selected_dtypes(study)
    out: list[Eligibility] = []
    for op in operators or list_operators():
        cfg = load_operator_config(op)
        cfg["_operator"] = op
        dtypes = operator_dtypes(cfg)
        if chosen is not None:
            if op not in chosen:
                raise ValueError(f"{op}: no representative dtype in the frozen manifest")
            dtypes = [chosen[op]]
        for dtype in dtypes:
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
