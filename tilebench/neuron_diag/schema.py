"""Execution modes, status vocabularies and the per-measurement result record.

Every result row names its software stack and execution mode explicitly, so an
XLA number and a native number can never be confused, and a latency can only be
compared with a baseline measured on the same stack with the same timing method
and scope.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

# mode -> (stack, target)
EXECUTION_MODES: dict[str, tuple[str, str]] = {
    "xla_torch": ("xla", "torch"),
    "xla_nki": ("xla", "nki"),
    "native_torch_eager": ("native", "torch"),
    "native_torch_compiled": ("native", "torch"),
    "native_nki": ("native", "nki"),
}

STACKS = ("xla", "native")

# The TileBench++ Trn2 benchmark compares PyTorch eager with NKI, both on the native stack
# (speedup_nki = torch_eager_ms / nki_ms). Every other mode is a diagnostic: the xla_* modes
# are kept so archived legacy XLA runs still parse, and native_torch_compiled
# (torch.compile) is a diagnostic only. None of them enters a benchmark aggregate,
# speedup, coverage count, table, figure or classification.
BENCHMARK_STACK = "native"
BENCHMARK_MODES = ("native_torch_eager", "native_nki")
LEGACY_DIAGNOSTIC_MODES = tuple(m for m, (s, _) in EXECUTION_MODES.items() if s == "xla")
COMPILE_DIAGNOSTIC_MODES = ("native_torch_compiled",)
DIAGNOSTIC_MODES = LEGACY_DIAGNOSTIC_MODES + COMPILE_DIAGNOSTIC_MODES

# Outcome of one (operator, case, mode). Exactly one per record.
STATUSES = (
    "pass",                     # ran, verified, timed as requested
    "no_nki_impl",              # operator has no NKI implementation
    "import_incompatible",      # module/API import failed on this stack
    "unsupported_dtype_shape",  # implementation or stack declares the case unsupported
    "compile_failure",
    "runtime_failure",
    "correctness_failure",
    "cpu_fallback",             # verified output, but confirmed CPU execution inside run()
    "timing_unavailable",       # verified, but no timing of the requested kind exists
    "timeout",
    "blocked_env",              # the stack is not installed / device not accessible
    "not_run",                  # skipped; `reason` says why
)

# CPU fallback evidence. "unable_to_determine" is never reported as
# "no_fallback_observed_with_evidence". A device->host scalar read (.item()) is a host
# round trip, not CPU computation, and has its own state.
FALLBACK_STATES = ("no_fallback_observed_with_evidence", "confirmed_cpu_fallback",
                   "host_scalar_read", "unable_to_determine")
# Names used by runs before the vocabulary was split (r1/r2 rows).
_FALLBACK_ALIASES = {"none_found": "no_fallback_observed_with_evidence",
                     "confirmed": "confirmed_cpu_fallback",
                     "undetermined": "unable_to_determine"}

# How a latency was obtained. Only the *_device methods may fill device_ms.
TIMING_METHODS = (
    "wall_sync",             # host wall clock around run() + device synchronization
    "neuron_rt_inspect",     # Neuron runtime execution trace (existing XLA-path timer)
    "native_device_trace",   # native-stack device trace (only once an API is verified)
)
DEVICE_TIMING_METHODS = ("neuron_rt_inspect", "native_device_trace")

TIMING_SCOPES = ("full_run", "kernel_only", "first_call")


class RecordError(ValueError):
    """A record would misreport what was measured."""


@dataclasses.dataclass
class Record:
    operator: str
    case_id: int
    mode: str
    status: str
    shape: dict = dataclasses.field(default_factory=dict)
    dtype: str = ""
    source_hash: str = ""
    config: dict = dataclasses.field(default_factory=dict)
    hardware: dict = dataclasses.field(default_factory=dict)
    verification_status: str = "not_checked"   # pass / fail / not_checked
    verification_detail: str = ""
    changed_input_status: str = "not_checked"  # second, different input bundle
    fallback_status: str = "unable_to_determine"
    fallback_evidence: str = ""
    wall_timing_method: str | None = None   # how wall_ms was taken ("wall_sync")
    timing_method: str | None = None        # how device_ms was taken (a device method)
    timing_scope: str | None = None
    warmup: int = 0
    repeat: int = 0
    smoke: bool = False
    wall_ms: float | None = None
    wall_samples_ms: list = dataclasses.field(default_factory=list)
    device_ms: float | None = None
    device_samples_ms: list = dataclasses.field(default_factory=list)
    device_unavailable_reason: str = ""
    first_call_ms: float | None = None
    artifact_path: str = ""
    artifact_identity: list = dataclasses.field(default_factory=list)
    compile_info: dict = dataclasses.field(default_factory=dict)
    reason: str = ""
    error: str = ""

    @property
    def stack(self) -> str:
        return EXECUTION_MODES[self.mode][0]

    @property
    def target(self) -> str:
        return EXECUTION_MODES[self.mode][1]

    def validate(self) -> "Record":
        if self.mode not in EXECUTION_MODES:
            raise RecordError(f"unknown execution mode {self.mode!r}")
        if self.status not in STATUSES:
            raise RecordError(f"unknown status {self.status!r}")
        if self.fallback_status not in FALLBACK_STATES:
            raise RecordError(f"unknown fallback status {self.fallback_status!r}")
        if self.timing_method is not None and self.timing_method not in DEVICE_TIMING_METHODS:
            raise RecordError(f"timing_method {self.timing_method!r} is not a device timing method")
        if self.wall_timing_method is not None and self.wall_timing_method != "wall_sync":
            raise RecordError(f"unknown wall timing method {self.wall_timing_method!r}")
        if self.wall_ms is not None and self.wall_timing_method is None:
            raise RecordError("wall_ms needs wall_timing_method")
        if self.timing_scope is not None and self.timing_scope not in TIMING_SCOPES:
            raise RecordError(f"unknown timing scope {self.timing_scope!r}")
        if self.device_ms is not None:
            if not _finite(self.device_ms):
                raise RecordError("device_ms must be finite or None")
            if self.timing_method is None:
                raise RecordError("device_ms needs a device timing_method")
            if not self.artifact_identity:
                raise RecordError(
                    "device_ms without an executed-artifact identity: the time cannot be "
                    "attributed to this operator, report it as None with a reason")
        elif self.device_samples_ms:
            raise RecordError("device samples present but device_ms is None")
        if self.device_ms is None and self.status == "pass" and not self.device_unavailable_reason:
            raise RecordError("device_ms is None: device_unavailable_reason must say why")
        if self.wall_ms is not None and not _finite(self.wall_ms):
            raise RecordError("wall_ms must be finite or None")
        if self.status == "pass":
            if self.verification_status != "pass":
                raise RecordError("status 'pass' requires verification_status 'pass'")
            if self.fallback_status == "confirmed_cpu_fallback":
                raise RecordError("a confirmed CPU fallback cannot be status 'pass'")
        if self.status in ("correctness_failure", "compile_failure", "runtime_failure",
                           "timeout", "blocked_env", "import_incompatible") and (
                self.wall_ms is not None or self.device_ms is not None):
            raise RecordError(f"status {self.status!r} cannot carry a latency")
        return self

    def to_json(self) -> dict:
        self.validate()
        d = dataclasses.asdict(self)
        d["stack"] = self.stack
        d["target"] = self.target
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Record":
        fields = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in fields}
        if kw.get("fallback_status") in _FALLBACK_ALIASES:
            kw["fallback_status"] = _FALLBACK_ALIASES[kw["fallback_status"]]
        return cls(**kw).validate()


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def comparable(a: Record, b: Record, metric: str) -> tuple[bool, str]:
    """Whether latency `metric` ("wall_ms" or "device_ms") of `a` and `b` may form a
    speedup. Same operator, case, stack, timing scope and method; both verified;
    neither with a confirmed CPU fallback. An undetermined fallback state does not
    block the ratio, but the report shows it next to the number."""
    if metric not in ("wall_ms", "device_ms"):
        raise ValueError(metric)
    checks = [
        (a.operator == b.operator and a.case_id == b.case_id, "different operator/case"),
        (a.stack == b.stack, "different software stacks: report cross-stack ratios separately"),
        (a.timing_scope == b.timing_scope, "different timing scopes"),
        (metric == "wall_ms" or a.timing_method == b.timing_method, "different device timing methods"),
        (a.verification_status == "pass" and b.verification_status == "pass", "unverified output"),
        ("confirmed_cpu_fallback" not in (a.fallback_status, b.fallback_status), "confirmed CPU fallback"),
        (getattr(a, metric) is not None and getattr(b, metric) is not None, f"{metric} missing"),
    ]
    for ok, why in checks:
        if not ok:
            return False, why
    return True, ""


def speedup(baseline: Record, candidate: Record, metric: str) -> float | None:
    ok, _ = comparable(baseline, candidate, metric)
    if not ok:
        return None
    return getattr(baseline, metric) / getattr(candidate, metric)
