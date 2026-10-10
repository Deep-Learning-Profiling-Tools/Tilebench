"""Trn2 native NKI evaluation pieces (CPU-only; the device run is evidence under artifacts/llm_v2/preflight/Trn2)."""
import subprocess
import sys

import pytest
import torch

from tilebench.llm.v2.evaluation import fingerprint as fpm
from tilebench.llm.v2.evaluation import neuron as N
from tilebench.llm.v2.evaluation.job import build_evaluation_job
from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.tasks import support
from tilebench.paths import REPO_ROOT


def test_trn2_eligibility_manifest_is_sealed_and_rebuilds():
    data = support.load_trn2_eligibility()
    assert data["counts"] == {"eligible": 36, "excluded": 9} and data["status"] == "frozen"
    assert data["excluded_by_class"] == {
        "cpu_fallback": ["2d_max_pooling", "argmax", "radix_sort"], "baseline_correctness_failure": ["flash_attention"],
        "no_eager_baseline": ["block_sparse_attention"],
        "baseline_case_failure": ["gaussian_blur", "reverse_array", "top_k_selection"],
        "scoring_peak_unavailable": ["matmul_int8"]}
    for op, e in data["operators"].items():
        assert e["n_cases"] == 20 and e["cases_matched"] == 20, op
        assert (e["status"] == "eligible") == (e["baseline_valid_cases"] == 20 and e["scoring"]["status_counts"] == {"ok": 20}), op
        assert (e["exclusion_class"] is None) == (e["status"] == "eligible"), op
    for op in ("interleave", "matrix_transpose", "matmul_fp32_fp16_fp8"):
        assert data["operators"][op]["status"] == "eligible", op
    # matmul_int8 runs (20/20 baselines) but has no int8 SOL ceiling: excluded for scoring, not as a CPU fallback
    i8 = data["operators"]["matmul_int8"]
    assert (i8["baseline_valid_cases"], i8["exclusion_class"], i8["scoring"]["mode"]) == (20, "scoring_peak_unavailable", "mma_int8_i32acc")
    assert sum(e["n_cases"] for e in data["operators"].values() if e["status"] == "eligible") == 720
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/llm_v2/build_trn2_eligibility.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_trn2_status_never_substitutes_a_dtype():
    data = support.load_trn2_eligibility()
    e = data["operators"]["vector_add"]
    assert support.trn2_status("vector_add", e["dtype"], e["case_set_id"])[0] == "eligible"
    assert support.trn2_status("vector_add", "fp32", None)[0] == "unsupported"
    st, why = support.trn2_status("argmax", "fp16", data["operators"]["argmax"]["case_set_id"])
    assert st == "unsupported" and "CPU fallback" in why and "(cpu_fallback)" in why
    st, why = support.trn2_status("matmul_int8", "int8", data["operators"]["matmul_int8"]["case_set_id"])
    assert st == "unsupported" and "(scoring_peak_unavailable)" in why
    with pytest.raises(ValueError):
        support.trn2_status("vector_add", e["dtype"], "cs_wrong")


def _job(device, dsl, adapter_study=None):
    study = adapter_study or ms.load_study()
    return build_evaluation_job(operator="vector_add", dtype="fp16", params={"n": 1048576}, dsl=dsl, device=device,
                                arch=None, rules={}, study=study)


def test_gpu_fingerprint_body_has_no_neuron_key_and_trn2_body_has_it():
    b200 = fpm.evaluator_fingerprint(_job("B200", "triton"), worker_timeout_s=3600, isolation_backend="bwrap")
    trn2 = fpm.evaluator_fingerprint(_job("Trn2", "nki"), worker_timeout_s=3600, isolation_backend="docker")
    assert "neuron" not in b200
    assert set(trn2["neuron"]) == {"evaluation_sources_sha256", "core_sources_sha256", "packages"}
    assert set(trn2["neuron"]["packages"]) == set(N.NEURON_PACKAGES)
    assert _job("Trn2", "nki").timing_adapter == N.ADAPTER and _job("Trn2", "nki").expected_timing_mode == "eager"


def test_formal_isolation_backends_per_device():
    study = ms.load_study()
    assert N.formal_isolation_backends("B200", study) == ("bwrap",)
    assert N.formal_isolation_backends("Trn2", study) == ("bwrap", "docker")


def test_numerical_checks_neuron_semantics_on_cpu_tensors():
    def ref(x, y):
        return x + y

    def make():
        return (torch.randn(64), torch.randn(64))

    out = N.run_numerical_checks_neuron(lambda x, y: x + y, ref, make, atol=1e-6, rtol=1e-6, sync=lambda: None)
    assert [c.name for c in out] == ["fresh_storage", "same_address_new_values", "repeat_same_inputs"] and all(c.ok for c in out)
    aliasing = N.run_numerical_checks_neuron(lambda x, y: x.add_(y), ref, make, atol=1e-6, rtol=1e-6, sync=lambda: None)
    assert not aliasing[0].ok and aliasing[0].output_aliases_inputs == [0] and aliasing[0].inputs_mutated == [0]
    state = {"n": 0}

    def stateful(x, y):
        state["n"] += 1
        return x + y + (1.0 if state["n"] >= 3 else 0.0)
    rep = N.run_numerical_checks_neuron(stateful, ref, make, atol=1e-6, rtol=1e-6, sync=lambda: None)
    assert rep[-1].name == "repeat_same_inputs" and not rep[-1].ok


def test_host_fallback_tracker_raises_contract_violation():
    from tilebench.llm.v2.evaluation.worker import PrecisionTampered

    class FakeTN:
        def __init__(self, fb, ex):
            self.fb, self.ex = fb, ex

        def clear_op_tracking(self):
            pass

        def get_fallback_ops(self):
            return self.fb

        def get_executed_ops(self):
            return self.ex

    ok = N.track_host_fallback(lambda: 1, FakeTN([], ["neuron::nki_call", "aten::empty"]))
    assert ok() == 1
    for fb, ex in ((["aten::sort.values_stable"], []), ([], ["neuron::copy::neuron_to_cpu"])):
        bad = N.track_host_fallback(lambda: 1, FakeTN(fb, ex))
        with pytest.raises(PrecisionTampered, match="left the Neuron device"):
            bad()


def test_make_evaluator_keeps_the_shared_evaluator_for_gpus(monkeypatch):
    from tilebench.llm.v2.evaluation import launcher as L
    monkeypatch.setattr(L, "detect_isolation", lambda mode="auto": {"backend": "none", "device_nodes": {}})
    ev = N.make_evaluator("B200", timeout_s=10)
    assert type(ev) is L.SubprocessEvaluator
    monkeypatch.setattr(N, "detect_isolation_neuron", lambda mode="auto": {"backend": "none", "device_nodes": {}})
    ev2 = N.make_evaluator("Trn2", timeout_s=10)
    assert isinstance(ev2, N.NeuronSubprocessEvaluator) and ev2.report["timing_adapter"] == N.ADAPTER


def test_neuron_worker_does_not_import_torch_at_import_time():
    code = "import sys, tilebench.llm.v2.evaluation.neuron_worker as w; print('torch' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert r.stdout.strip() == "False", r.stderr[-2000:]


def test_neuron_worker_refuses_without_sandbox_flag(tmp_path, monkeypatch):
    from tilebench.llm.v2.evaluation import neuron_worker
    monkeypatch.delenv("TILEBENCH_V2_SANDBOX", raising=False)
    (tmp_path / "job.json").write_text("{}")
    assert neuron_worker.main(["--job", str(tmp_path / "job.json"), "--out", str(tmp_path / "o.json")]) == 2
