"""Timing adapters per device class.

proton_cuda_graph    NVIDIA: Proton + CUDA-graph replay (evaluation.timing.measure)
proton_rocm_eager    ROCm: Proton/roctracer, eager (the same function; the
                     torch build decides, see core.timer.effective_use_cuda_graph)
neuron_runtime_trace Trainium2 / NKI, LEGACY torch-xla path (core.nki_orchestrator,
                     runtime-inspect trace): not used, not ready.
neuron_native_trace  Trainium2 / NKI 0.7 on the native PyTorch Neuron stack
                     (evaluation.neuron + evaluation.neuron_worker): torch.device("neuron"),
                     no torch-xla, no torch.compile; device busy sum of every execution of
                     one run() from the Neuron runtime trace (per-core copies merged,
                     tilebench.core.neuron_native / neuron_intervals); 1 warmup + 3 timed."""
from __future__ import annotations


class AdapterNotReady(RuntimeError):
    pass


ADAPTERS = {
    "proton_cuda_graph": {"ready": True, "module": "tilebench.llm.v2.evaluation.timing",
                          "device_nodes": ["/dev/nvidia*"], "runtime": "CUDA driver + CUPTI (Proton)",
                          "isolation_verified_on": ["B200 (validation campaign 2026-10-05)"]},
    "proton_rocm_eager": {"ready": True, "module": "tilebench.llm.v2.evaluation.timing",
                          "device_nodes": ["/dev/kfd", "/dev/dri"], "runtime": "ROCm + roctracer (Proton)",
                          "isolation_verified_on": [], "pending": "sandbox device binding and roctracer inside bwrap not exercised on MI300X"},
    "neuron_runtime_trace": {"ready": False, "module": "tilebench.core.nki_orchestrator",
                             "device_nodes": ["/dev/neuron*"], "runtime": "Neuron runtime + neuron-profile",
                             "isolation_verified_on": [], "handoff": "docs/llm_v2/NKI_HANDOFF.md"},
    "neuron_native_trace": {"ready": True, "module": "tilebench.llm.v2.evaluation.neuron",
                            "device_nodes": ["/dev/neuron*"],
                            "runtime": "native torch-neuronx + Neuron runtime profiler (NeuronConfig RUNTIME, nc_exec_running)",
                            "isolation_verified_on": ["Trn2 trn2.3xlarge, docker backend (validation 2026-10-10)"]},
}


def require_adapter(name: str) -> dict:
    if name not in ADAPTERS:
        raise AdapterNotReady(f"unknown timing adapter {name!r}")
    a = ADAPTERS[name]
    if not a["ready"]:
        raise AdapterNotReady(f"timing adapter {name!r} is not validated; see {a['handoff']}")
    return a
