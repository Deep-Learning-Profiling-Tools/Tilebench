"""Empirical Roofline calibration, profile schema, the empirical target and
its binding to campaigns (NEXT_STEP_EMPIRICAL_CALIBRATION.md §9)."""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
from pathlib import Path

import pytest

from tilebench.llm.v2.calibration import protocol, schema as cs, stats
from tilebench.llm.v2.calibration import run as crun
from tilebench.llm.v2.metrics import empirical as E
from tilebench.llm.v2.orchestration import campaign
from tilebench.llm.v2.orchestration.campaign import ResumeRefused, _check_resume
from tilebench.llm.v2.orchestration.state import TrajectoryState
from tilebench.llm.v2.providers.factory import GeneratorSpec
from tilebench.paths import PEAK_PERFORMANCE_ROOT, REPO_ROOT

CONTRACTS = REPO_ROOT / "tilebench" / "llm" / "v2" / "contracts" / "data"
TIMING_PARAGRAPH = ("The measured quantity is the GPU time of all device work that `run()` causes on every call: every kernel, "
                    "fill, copy, cast or repack launched inside `run()` is counted. Host-side work inside `run()` (allocation "
                    "calls, shape, stride and metadata reads, Python control flow) is not GPU time and is not part of the "
                    "measured number.")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_profile(values: dict, *, device="B200", quick=False, cal_id="T-1") -> dict:
    modes = {}
    for name, reg in cs.MODES.items():
        v = values.get(name)
        modes[name] = {"unit": reg["unit"], "source_kind": "measured", "role": reg["role"],
                       "status": "calibrated" if v is not None else "not_measured", "value": v}
    p = {"schema": cs.PROFILE_SCHEMA, "ceiling_basis": cs.CEILING_BASIS, "calibration_id": cal_id, "device": device,
         "scope": {"kind": "single_device_full"}, "protocol": {"id": protocol.PROTOCOL_ID, "sha256": "x", "quick": quick},
         "code": {}, "environment": {}, "modes": modes}
    return cs.seal(p)


def approved_modes_doc() -> dict:
    doc = json.loads(json.dumps(E.load_modes()))
    doc["status"] = "approved"
    for d in doc["decisions"].values():
        d["status"] = "approved"
    return doc


BW, P16 = 6.84e12, 1.267e15
MATMUL = {"flops_expr": "2 * M * N * K", "bytes_expr": "(M * K + K * N + M * N) * dtype_size"}


# ----------------------------------------------------------------------------- the calibration never writes canonical data
def test_calibration_output_dir_refuses_legacy_and_package_locations(tmp_path):
    for bad in (PEAK_PERFORMANCE_ROOT / "B200-new", REPO_ROOT / "results" / "x", REPO_ROOT / "tilebench" / "x"):
        with pytest.raises(crun.CalibrationRefused):
            crun.check_output_dir(bad)
    assert crun.check_output_dir(tmp_path / "fresh") == (tmp_path / "fresh").resolve()
    (tmp_path / "used").mkdir(); (tmp_path / "used" / "f").write_text("x")
    with pytest.raises(crun.CalibrationRefused, match="not empty"):
        crun.check_output_dir(tmp_path / "used")


def _fake_points(mode: str, proto: dict) -> list[dict]:
    def pt(pid, thr, *, valid=True, eligible=True, primary=True, probe="d2d_copy"):
        return {"id": pid, "probe": probe, "valid": valid, "throughput": thr, "point_time_s": 1e-3, "batch_spread": 0.001,
                "hbm_eligible": eligible, "primary": primary, "role": "hbm" if eligible else "cache_scale_diagnostic",
                "check": {"ok": True}, "device_vs_event_time": 1.0, "trace": {"busy_fraction": 0.99}}
    if mode == "hbm_stream_bw":
        return [pt("d2d_copy/16MiB", 9.0e12, eligible=False),          # cache-scale, faster: must not be selected
                pt("d2d_copy/512MiB", 6.40e12), pt("d2d_copy/1024MiB", 6.43e12), pt("d2d_copy/4096MiB", 6.42e12),
                pt("sm_read/4096MiB", 7.4e12, primary=False, probe="sm_read")]   # diagnostic probe, faster: not primary
    return [pt(f"{mode}/a", 1.0e14, probe=mode), pt(f"{mode}/b", 1.1e14, probe=mode)]


def test_calibration_run_writes_only_its_directory_and_keeps_B200_json(monkeypatch, tmp_path):
    from tilebench import hardware
    from tilebench.llm.v2.calibration import environment, gpu_probes
    import torch
    canonical = PEAK_PERFORMANCE_ROOT / "B200.json"
    before = _sha(canonical)
    monkeypatch.setattr(crun, "detect_backend", lambda: "cuda")
    monkeypatch.setattr(hardware, "detect_arch", lambda: "blackwell")
    monkeypatch.setattr(hardware, "last_level_cache_bytes", lambda: 126_000_000)
    monkeypatch.setattr(environment, "blas_stack_check", lambda env=None, timeout=0: {"ok": True})
    monkeypatch.setattr(environment, "capture", lambda backend, index: {
        "device_properties": {"name": "fake", "uuid": "u", "multi_processor_count": 4}, "visible_devices": {},
        "nvidia_smi": {}, "other_compute_processes": []})
    monkeypatch.setattr(environment, "device_sample", lambda backend, index: {})

    class T:
        def __init__(self, *a): pass
        def start(self): pass
        def stop(self): return {"available": False}
    monkeypatch.setattr(environment, "Telemetry", T)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: None)
    import tilebench.llm.v2.orchestration.locks as locks
    monkeypatch.setattr(locks, "device_lock", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(gpu_probes, "probe_stream", lambda probe, size, proto, dev, llc: None)
    monkeypatch.setattr(gpu_probes, "probe_gemm", lambda mode, proto, dev: {"api": "fake", "points": _fake_points(mode, proto)})
    monkeypatch.setattr(gpu_probes, "probe_fma", lambda mode, proto, dev, sm, backend: {"api": "fake", "points": _fake_points(mode, proto)})
    # hbm points come from probe_stream per (probe,size); replace the whole loop result through probe_stream returning points
    pts = iter(_fake_points("hbm_stream_bw", protocol.DEFAULT))
    filler = {"id": "filler", "probe": "fill_write", "valid": False, "throughput": 0.0, "point_time_s": 1.0,
              "batch_spread": 0.0, "hbm_eligible": False, "primary": False, "role": "cache_scale_diagnostic",
              "check": {"ok": False}, "device_vs_event_time": 0.0, "trace": {"busy_fraction": 0.0}}
    monkeypatch.setattr(gpu_probes, "probe_stream", lambda probe, size, proto, dev, llc: next(pts, dict(filler)))
    out = tmp_path / "cal"
    r = crun.run_calibration("B200", out, modes=["hbm_stream_bw", "mma_fp16_f32acc", "fp32_fma_vector"], log=lambda s: None)
    assert _sha(canonical) == before                                   # canonical table untouched
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert {"profile.json", "protocol.json", "environment.json", "summary.json", "SHA256SUMS"} <= set(files)
    sums = dict(l.split("  ", 1)[::-1] for l in (out / "SHA256SUMS").read_text().splitlines())
    assert all(_sha(out / f) == sums[f] for f in files if f != "SHA256SUMS")
    prof = json.loads((out / "profile.json").read_text())
    assert cs.validate_profile(prof) == [] and prof["profile_sha256"] == r["profile_sha256"]
    hbm = prof["modes"]["hbm_stream_bw"]
    assert hbm["status"] == "calibrated" and hbm["selected_point"] == "d2d_copy/1024MiB" and hbm["value"] == 6.43e12
    assert prof["modes"]["mma_fp16_f32acc"]["value"] == 1.1e14 and prof["modes"]["mma_tf32_f32acc"]["status"] == "not_measured"
    assert all(not str(p).startswith(str(PEAK_PERFORMANCE_ROOT)) for p in out.rglob("*"))


def test_hbm_selection_excludes_cache_points_and_requires_a_plateau():
    def pt(pid, thr, eligible=True, primary=True, probe="d2d_copy", valid=True):
        return {"id": pid, "probe": probe, "valid": valid, "throughput": thr, "hbm_eligible": eligible, "primary": primary}
    sel = stats.select_mode_value([pt("c", 9e12, eligible=False), pt("a", 6.80e12), pt("b", 6.84e12), pt("d", 6.82e12),
                                   pt("r", 7.5e12, primary=False, probe="sm_read")], bandwidth=True)
    assert sel["status"] == "calibrated" and sel["value"] == 6.84e12 and sel["selected"] == "b"
    one = stats.select_mode_value([pt("c", 9e12, eligible=False), pt("a", 6.80e12)], bandwidth=True)
    assert one["status"] == "failed" and "plateau" in one["reason"]
    none = stats.select_mode_value([pt("a", 6.80e12, valid=False)], bandwidth=True)
    assert none["status"] == "failed" and none["value"] is None
    assert not stats.hbm_eligible(10 ** 9, None, 4.0)                   # unknown LLC: nothing is eligible
    assert stats.hbm_eligible(4 * 126_000_000, 126_000_000, 4.0) and not stats.hbm_eligible(126_000_000, 126_000_000, 4.0)
    flagged = stats.select_mode_value([{"id": "x", "valid": True, "throughput": 1.0}, {"id": "y", "valid": True, "throughput": 1.5}])
    assert flagged["flags"] and "saturation_not_demonstrated" in flagged["flags"][0]


# ----------------------------------------------------------------------------- modes are not interchangeable
def test_modes_fail_closed_no_aliases():
    prof = make_profile({"hbm_stream_bw": BW, "gemm_fp32_ieee": 66.5e12, "mma_int8_i32acc": 2.7e15,
                         "mma_fp8_e4m3_f32acc": 2.6e15, "mma_fp16_f32acc": P16})
    assert E.peak_for_mode(prof, "mma_fp16_f32acc", unit="FLOP/s") == P16
    with pytest.raises(E.PeakUnavailable, match="not_measured"):          # IEEE fp32 GEMM is not TF32
        E.peak_for_mode(prof, "mma_tf32_f32acc", unit="FLOP/s")
    with pytest.raises(E.PeakUnavailable, match="control"):               # the SGEMM control is never a ceiling
        E.peak_for_mode(prof, "gemm_fp32_ieee", unit="FLOP/s")
    with pytest.raises(E.PeakUnavailable):                                # int8 matrix dot is not an int32 vector rate
        E.peak_for_mode(prof, "int32_vector", unit="OP/s")
    with pytest.raises(E.PeakUnavailable, match="unit"):                  # OP/s is not FLOP/s
        E.peak_for_mode(prof, "mma_int8_i32acc", unit="FLOP/s")
    with pytest.raises(E.PeakUnavailable):                                # e4m3 is not e5m2
        E.peak_for_mode(prof, "mma_fp8_e5m2_f32acc", unit="FLOP/s")
    with pytest.raises(E.PeakUnavailable, match="not registered"):
        E.peak_for_mode(prof, "tc_fp16", unit="FLOP/s")
    bad = make_profile({"hbm_stream_bw": BW}); bad["modes"]["hbm_stream_bw"]["value"] = float("nan")
    assert any("finite" in e for e in cs.validate_profile(bad, require_sealed=False))


def test_declaration_validation_rejects_defaults_unit_mismatch_and_control_modes():
    doc = E.load_modes()
    assert E.validate_modes(doc) == []
    bad = json.loads(json.dumps(doc))
    bad["default_by_dtype"] = {"fp16": "mma_fp16_f32acc"}
    bad["operators"]["matmul_int8"]["modes"]["int8"] = "mma_fp16_f32acc"            # OP count vs FLOP/s mode
    bad["operators"]["softmax"]["modes"]["fp32"] = "gemm_fp32_ieee"                  # control mode
    bad["operators"]["relu"]["modes"]["fp32"] = "tc_fp32"                            # legacy alias
    errs = E.validate_modes(bad)
    assert any("dtype-wide defaults" in e for e in errs) and any("does not match mode" in e for e in errs)
    assert any("control mode" in e for e in errs) and any("not in the calibration registry" in e for e in errs)


# ----------------------------------------------------------------------------- T_emp: units, status, provisional vs final
def test_t_emp_units_and_statuses(tmp_path):
    prof = make_profile({"hbm_stream_bw": BW, "mma_fp16_f32acc": P16})
    params = {"M": 4096, "N": 4096, "K": 20480}
    rec = E.t_emp("B200", "matmul_fp32_fp16_fp8", "fp16", params, 1, MATMUL, approved_modes_doc(), prof)
    F, Q = 2 * 4096 * 4096 * 20480, (4096 * 20480 + 20480 * 4096 + 4096 * 4096) * 2
    assert rec.status == "ok" and rec.mode == "mma_fp16_f32acc"
    assert rec.compute_term_ms == pytest.approx(F / P16 * 1e3) and rec.memory_term_ms == pytest.approx(Q / BW * 1e3)
    assert rec.t_emp_ms == pytest.approx(max(F / P16, Q / BW) * 1e3) and rec.ceiling_basis == "empirical"
    pending = E.t_emp("B200", "matmul_fp32_fp16_fp8", "fp16", params, 1, MATMUL, E.load_modes(), prof)
    assert pending.status == "definition_pending" and pending.t_emp_ms is None and pending.provisional_t_emp_ms == rec.t_emp_ms
    assert "M1_declaration_revision_2" in pending.pending_decisions
    assert E.t_emp("B200", "matmul_fp32_fp16_fp8", "fp16", params, 1, MATMUL, approved_modes_doc(), None).status == "profile_missing"
    tf32 = E.t_emp("B200", "matmul_fp32_fp16_fp8", "fp32", params, 1, MATMUL, approved_modes_doc(), prof)
    assert tf32.status == "peak_unavailable" and "mma_tf32_f32acc" in tf32.reason
    mem = E.t_emp("B200", "relu", "fp32", {"n": 1 << 20}, 1 << 20, {"flops_expr": "n", "bytes_expr": "2*n*dtype_size"},
                  approved_modes_doc(), prof)
    assert mem.status == "ok" and mem.compute_term_ms is None and mem.t_emp_ms == pytest.approx(mem.memory_term_ms)
    # profiles are loaded by registered sha256; quick profiles and sha mismatches are refused
    p = tmp_path / "profile.json"; p.write_text(json.dumps(prof))
    assert E.load_profile({"profile": str(p), "sha256": _sha(p)})["calibration_id"] == "T-1"
    with pytest.raises(E.ProfileUnavailable, match="sha256"):
        E.load_profile({"profile": str(p), "sha256": "0" * 64})
    q = tmp_path / "quick.json"; q.write_text(json.dumps(make_profile({"hbm_stream_bw": BW}, quick=True)))
    with pytest.raises(E.ProfileUnavailable, match="quick"):
        E.load_profile({"profile": str(q), "sha256": _sha(q)})
    legacy = tmp_path / "legacy.json"; legacy.write_text(json.dumps({"peak_bw_GBs": 6539.4, "peak_tflops": {"fp16": 2250}}))
    with pytest.raises(E.ProfileUnavailable, match="invalid"):
        E.load_profile({"profile": str(legacy), "sha256": _sha(legacy)})


# ----------------------------------------------------------------------------- binding: per device, pinned, formal resume refuses change
def _state(binding, run_type="formal"):
    return TrajectoryState(schema="s", trajectory_id="t", task={"device": "B200", "dsl": "triton", "operator": "softmax",
                           "dtype": "fp16"}, model="m", condition="base", config_hash="h", content_hashes={},
                           output_file="f", run_type=run_type, generator={"model_id": "m", "settings": {}},
                           scoring_binding=binding)


def test_binding_is_per_device_and_pinned_for_formal_resume():
    man = {"schema": "tilebench-calibration-manifest/1",
           "devices": {"B200": {"profile": "a.json", "sha256": "a" * 64, "status": "frozen"}}}
    b1 = E.scoring_binding("B200", manifest=man)
    man["devices"]["GH200"] = {"profile": "g.json", "sha256": "g" * 64, "status": "candidate"}
    assert E.scoring_binding("B200", manifest=man) == b1                  # another device's profile changes nothing
    assert not {"dsl", "model", "condition"} & set(b1)                      # one binding per device, shared by every DSL/model/condition
    gen = GeneratorSpec(name="m", provider="mock", model_id="m", api_key_env="X", settings={}, status="approved", set_by=None)
    st = _state(b1)
    _check_resume(st, config_hash="h", content_hashes={}, run_type="formal", gen=gen, scoring_binding=b1)
    man["devices"]["B200"]["sha256"] = "b" * 64
    b2 = E.scoring_binding("B200", manifest=man)
    with pytest.raises(ResumeRefused, match="scoring binding changed"):
        _check_resume(st, config_hash="h", content_hashes={}, run_type="formal", gen=gen, scoring_binding=b2)
    with pytest.raises(ResumeRefused, match="legacy"):
        _check_resume(_state(None), config_hash="h", content_hashes={}, run_type="formal", gen=gen, scoring_binding=b1)
    v = _state(b1, run_type="validation")
    _check_resume(v, config_hash="h", content_hashes={}, run_type="validation", gen=gen, scoring_binding=b2)
    assert v.scoring_binding == b2 and any("scoring binding changed" in n for n in v.notes)
    same_status = dict(b1, profile_status="candidate")                      # candidate -> frozen with the same sha is not a change
    _check_resume(_state(same_status), config_hash="h", content_hashes={}, run_type="formal", gen=gen, scoring_binding=b1)


def test_formal_preflight_blocks_until_profile_frozen_and_declaration_approved(study):
    pf = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="formal", operators=["softmax"])
    text = " ".join(pf.blockers)
    assert "frozen" in text and "decision M1" in text and "scoring targets not ready" in text
    assert pf.facts["scoring"]["binding"]["device"] == "B200" and pf.facts["scoring"]["status_counts"]
    pfv = campaign.preflight("B200", "triton", "base", live=True, study=study, run_type="validation",
                             models_selected=("gpt",), provider="openai", operators=["softmax"])
    assert not any("decision M1" in b for b in pfv.blockers) and any("decision M1" in w for w in pfv.warnings)


# ----------------------------------------------------------------------------- contract wording: timing boundary
def test_contracts_describe_the_measured_quantity_as_device_work_only():
    ops = sorted(p.name for p in CONTRACTS.iterdir() if p.is_dir())
    assert len(ops) == 45
    for op in ops:
        text = (CONTRACTS / op / "contract.md").read_text()
        sec = re.search(r"## Preprocessing and timing boundary\n\n(.*?)(?:\n## |\Z)", text, re.S)
        assert sec and sec.group(1).startswith(TIMING_PARAGRAPH), op
        assert not re.search(r"(does|is) timed: the (output )?allocation|run\(\) performs only(?! host-side)|Everything `?run\(\)`? does is timed", text), op
        rules = json.loads((CONTRACTS / op / "evaluator_rules.json").read_text())
        notes = rules["timing_boundary"]["notes"].lower()
        assert rules["contract_revision"] == 2 and ("host-side" in notes or "host side" in notes or "host " in notes), op
        assert "device work" in notes or "gpu time" in notes, op
        assert not re.search(r"allocation and the (single |one |launch)", notes), op
