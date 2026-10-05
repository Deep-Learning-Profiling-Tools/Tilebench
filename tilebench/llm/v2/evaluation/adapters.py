"""Timing adapters per device class.

proton_cuda_graph    NVIDIA: Proton + CUDA-graph replay (evaluation.timing.measure)
proton_rocm_eager    ROCm: Proton/roctracer, eager (the same function; the
                     torch build decides, see core.timer.effective_use_cuda_graph)
neuron_runtime_trace Trainium2 / NKI: NOT READY. The manual campaign measures NKI
                     through tilebench.core.nki_orchestrator.profile_case_on_neuron
                     (runtime-inspect trace on real inputs). Its Beta-5
                     compatibility, 1-warmup/3-timed support, timing boundary and
                     interference must be validated on a Trn2 host before it can
                     serve as the v2 evaluator; see docs/llm_v2/NKI_HANDOFF.md."""
from __future__ import annotations


class AdapterNotReady(RuntimeError):
    pass


ADAPTERS = {
    "proton_cuda_graph": {"ready": True, "module": "tilebench.llm.v2.evaluation.timing"},
    "proton_rocm_eager": {"ready": True, "module": "tilebench.llm.v2.evaluation.timing"},
    "neuron_runtime_trace": {"ready": False, "module": "tilebench.core.nki_orchestrator",
                             "handoff": "docs/llm_v2/NKI_HANDOFF.md"},
}


def require_adapter(name: str) -> dict:
    if name not in ADAPTERS:
        raise AdapterNotReady(f"unknown timing adapter {name!r}")
    a = ADAPTERS[name]
    if not a["ready"]:
        raise AdapterNotReady(f"timing adapter {name!r} is not validated; see {a['handoff']}")
    return a
