"""Device-free tests of the native Neuron diagnostics (tilebench/neuron_diag).

These use mocks and synthetic data only; passing them says nothing about a
real Neuron device.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tilebench.neuron_diag import cases, intervals, sources
from tilebench.neuron_diag.schema import Record, RecordError, comparable, speedup
from tilebench.neuron_diag.store import RunStore

REPO = Path(__file__).resolve().parents[1]


def _rec(**kw):
    base = dict(operator="op", case_id=0, mode="xla_nki", status="pass",
                verification_status="pass", fallback_status="no_fallback_observed_with_evidence",
                wall_timing_method="wall_sync", wall_ms=1.0, timing_scope="full_run",
                device_unavailable_reason="n/a")
    base.update(kw)
    return Record(**base)


# ---------------------------------------------------------------- imports / isolation
def _python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=REPO,
                          capture_output=True, text=True)


def test_package_imports_without_neuron_backends():
    r = _python("""
        import sys
        for m in ("torch_xla", "torch_neuronx", "nki"):
            sys.modules[m] = None      # any import of these raises ImportError
        import tilebench.neuron_diag.schema, tilebench.neuron_diag.store
        import tilebench.neuron_diag.intervals, tilebench.neuron_diag.report
        import tilebench.neuron_diag.runtime, tilebench.neuron_diag.cases
        import tilebench.neuron_diag.worker, tilebench.neuron_diag.xla_device
        print("ok")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


def test_importing_diagnostics_loads_no_backend():
    r = _python("""
        import sys
        import tilebench.neuron_diag.runtime, tilebench.neuron_diag.report
        print(sorted(m for m in ("torch_xla", "torch_neuronx") if m in sys.modules))
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


def test_native_runtime_reports_unavailable_backend():
    r = _python("""
        import sys
        sys.modules["torch_neuronx"] = None
        from tilebench.neuron_diag.runtime import RuntimeUnavailable, get_runtime
        try:
            get_runtime("native")
        except RuntimeUnavailable as e:
            print("unavailable")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "unavailable"


def test_xla_runtime_reports_unavailable_backend():
    r = _python("""
        import sys
        sys.modules["torch_xla"] = None
        from tilebench.neuron_diag.runtime import RuntimeUnavailable, get_runtime
        try:
            get_runtime("xla")
        except RuntimeUnavailable:
            print("unavailable")
    """)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "unavailable"


def test_native_runtime_refuses_without_sync_api(monkeypatch):
    import types

    import torch

    from tilebench.neuron_diag import runtime

    fake = types.ModuleType("torch_neuronx")
    monkeypatch.setitem(sys.modules, "torch_neuronx", fake)
    monkeypatch.setattr(torch, "empty", lambda *a, **k: None)  # pretend the device exists
    monkeypatch.delattr(torch, "neuron", raising=False)
    with pytest.raises(runtime.RuntimeUnavailable, match="synchronize"):
        runtime.NativeRuntime()


# ---------------------------------------------------------------- records
def test_stack_and_mode_are_distinct_in_json():
    d = _rec(mode="native_torch_compiled").to_json()
    assert (d["stack"], d["target"], d["mode"]) == ("native", "torch", "native_torch_compiled")
    d = _rec(mode="xla_torch").to_json()
    assert (d["stack"], d["target"]) == ("xla", "torch")


def test_unknown_mode_rejected():
    with pytest.raises(RecordError):
        _rec(mode="neuron_torch").validate()


def test_device_ms_requires_artifact_identity():
    with pytest.raises(RecordError, match="identity"):
        _rec(device_ms=0.5, timing_method="neuron_rt_inspect").validate()
    _rec(device_ms=0.5, timing_method="neuron_rt_inspect", artifact_identity=["abc"]).validate()


def test_wall_time_cannot_pose_as_device_time():
    with pytest.raises(RecordError, match="device timing method"):
        _rec(device_ms=0.5, timing_method="wall_sync", artifact_identity=["abc"]).validate()


def test_missing_device_timing_needs_a_reason():
    with pytest.raises(RecordError, match="reason"):
        _rec(device_unavailable_reason="").validate()


def test_correctness_failure_cannot_carry_latency():
    with pytest.raises(RecordError):
        _rec(status="correctness_failure", verification_status="fail").validate()
    _rec(status="correctness_failure", verification_status="fail", wall_ms=None,
         wall_timing_method=None).validate()


def test_timeout_cannot_carry_latency():
    with pytest.raises(RecordError):
        _rec(status="timeout").validate()


def test_confirmed_fallback_is_not_pass():
    with pytest.raises(RecordError):
        _rec(fallback_status="confirmed_cpu_fallback").validate()
    _rec(status="cpu_fallback", fallback_status="confirmed_cpu_fallback").validate()


def test_speedup_only_within_one_stack():
    t = _rec(mode="xla_torch", wall_ms=2.0)
    n = _rec(mode="xla_nki", wall_ms=1.0)
    assert speedup(t, n, "wall_ms") == 2.0
    nat = _rec(mode="native_nki", wall_ms=1.0)
    ok, why = comparable(t, nat, "wall_ms")
    assert not ok and "stack" in why
    assert speedup(t, nat, "wall_ms") is None


def test_speedup_excludes_confirmed_fallback_and_missing_device_time():
    t = _rec(mode="xla_torch", status="cpu_fallback", fallback_status="confirmed_cpu_fallback")
    n = _rec(mode="xla_nki")
    assert speedup(t, n, "wall_ms") is None
    assert speedup(_rec(mode="xla_torch"), n, "device_ms") is None


# ---------------------------------------------------------------- intervals
def test_intervals_merge_cores_and_separate_sum_union_span():
    ex = [
        {"start_ns": 0, "end_ns": 10, "execution_id": "a"},
        {"start_ns": 2, "end_ns": 12, "execution_id": "a"},   # second core of "a"
        {"start_ns": 8, "end_ns": 20, "execution_id": "b"},   # overlaps "a"
        {"start_ns": 30, "end_ns": 35, "execution_id": "c"},  # after a gap
    ]
    agg = intervals.aggregate(ex)
    assert agg["n_executions"] == 3
    assert agg["busy_sum_ms"] == pytest.approx((12 + 12 + 5) / 1e6)
    assert agg["union_ms"] == pytest.approx((20 + 5) / 1e6)
    assert agg["span_ms"] == pytest.approx(35 / 1e6)
    assert agg["overlap_between_executions"]


def test_intervals_empty_and_invalid():
    assert intervals.aggregate([])["busy_sum_ms"] is None
    with pytest.raises(ValueError):
        intervals.aggregate([{"start_ns": 5, "end_ns": 1}])


# ---------------------------------------------------------------- store / resume
def test_run_directory_is_never_reused(tmp_path):
    RunStore("r1", root=tmp_path)
    with pytest.raises(FileExistsError):
        RunStore("r1", root=tmp_path)


def test_files_are_not_overwritten(tmp_path):
    s = RunStore("r2", root=tmp_path)
    s.write_new("env/x.json", {"a": 1})
    with pytest.raises(FileExistsError):
        s.write_new("env/x.json", {"a": 2})
    assert json.loads(s.path("env/x.json").read_text()) == {"a": 1}


def test_refuses_to_write_under_results(tmp_path):
    from tilebench.paths import REPO_ROOT
    with pytest.raises(ValueError, match="results"):
        RunStore("r3", root=REPO_ROOT / "results" / "B200")


def test_paths_cannot_escape_run_directory(tmp_path):
    s = RunStore("r4", root=tmp_path)
    with pytest.raises(ValueError):
        s.path("..", "other", "file.json")


def test_resume_skips_only_same_hashes(tmp_path):
    s = RunStore("r5", root=tmp_path)
    s.append(_rec(source_hash="h1"), env_hash="e1")
    st = RunStore("r5", root=tmp_path, resume=True)
    assert st.resume_state("op", 0, "xla_nki", source_hash="h1", env_hash="e1") == "done"
    assert st.resume_state("op", 0, "xla_nki", source_hash="h2", env_hash="e1") == "stale"
    assert st.resume_state("op", 0, "xla_nki", source_hash="h1", env_hash="e2") == "stale"
    assert st.resume_state("op", 1, "xla_nki", source_hash="h1", env_hash="e1") == "todo"
    assert len(st.records()) == 1  # nothing was dropped


def test_invalid_rows_are_excluded_but_kept(tmp_path):
    from tilebench.neuron_diag.report import latest_records
    s = RunStore("r6", root=tmp_path)
    s.append(_rec(source_hash="h1"), env_hash="e1")
    bad = s.records()[0]["row_id"]
    s.mark_invalid(bad, reason="wrong_neuronx_cc_on_PATH", superseded_by=None)
    assert s.records() == []
    every = s.records(include_invalid=True)
    assert len(every) == 1 and every[0]["valid_for_analysis"] is False
    assert every[0]["invalid_reason"] == "wrong_neuronx_cc_on_PATH"
    # an invalid row never counts as done, even with matching hashes
    assert s.resume_state("op", 0, "xla_nki", source_hash="h1", env_hash="e1") == "todo"
    assert latest_records(every) == {}


def test_new_row_supersedes_previous_valid_row(tmp_path):
    from tilebench.neuron_diag.report import latest_records
    s = RunStore("r7", root=tmp_path)
    s.append(_rec(source_hash="h1", wall_ms=2.0), env_hash="e1")
    s.append(_rec(source_hash="h1", wall_ms=1.0), env_hash="e1")
    every = s.records(include_invalid=True)
    assert [r["valid_for_analysis"] for r in every] == [False, True]
    assert every[0]["superseded_by"] == every[1]["row_id"]
    assert latest_records(s.records())[("op", 0, "xla_nki")]["wall_ms"] == 1.0
    assert len(s.path("records.jsonl").read_text().splitlines()) == 2  # history kept


def test_report_refuses_duplicate_valid_rows():
    from tilebench.neuron_diag.report import DuplicateValidRows, latest_records
    rows = [dict(operator="op", case_id=0, mode="xla_nki", valid_for_analysis=True, recorded_at=t)
            for t in ("a", "b")]
    with pytest.raises(DuplicateValidRows):
        latest_records(rows)


def test_resume_reruns_blocked_or_not_run(tmp_path):
    s = RunStore("r6", root=tmp_path)
    s.append(_rec(status="blocked_env", wall_ms=None, wall_timing_method=None,
                  verification_status="not_checked", source_hash="h"), env_hash="e")
    assert s.resume_state("op", 0, "xla_nki", source_hash="h", env_hash="e") == "todo"


def test_resume_requires_existing_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunStore("missing", root=tmp_path, resume=True)


# ---------------------------------------------------------------- timeout
def test_child_timeout_is_reported(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import neuron_diag

    st, rc = neuron_diag._run_child([sys.executable, "-c", "import time; time.sleep(30)"],
                                    overlay=tmp_path, env_extra={}, timeout=1,
                                    log_path=tmp_path / "log.txt")
    assert (st, rc) == ("timeout", None)


# ---------------------------------------------------------------- sources / cases
def test_import_patch_only_touches_old_layout_imports():
    src = "from core.nki_autotune import X\nfrom tilebench.core.a import b\nx = 'from core.'\n"
    out = sources.patch_imports(src)
    assert out.splitlines()[0] == "from tilebench.core.nki_autotune import X"
    assert out.splitlines()[1] == "from tilebench.core.a import b"
    assert out.splitlines()[2] == "x = 'from core.'"


def test_static_scan_marks_presence_not_cost():
    src = textwrap.dedent("""
        import torch
        try:
            import nki
        except ImportError:
            nki = None
        if nki is not None:
            @nki.jit
            def k(a):
                return a
        _kernel = k[_lnc_degree()] if nki is not None else None
        def run(x, autotune=False):
            y = x.reshape(128, -1).contiguous()
            return _kernel(y)[:10]
    """)
    s = sources.scan_impl_nki(src)
    assert s["kernels"] == ["k"]
    assert s["jit_guarded_by_nki_none"] and not s["module_level_unguarded_jit"]
    assert "_kernel" in s["launch_sites_in_run_static"]
    assert set(s["wrapper_ops_in_run_static"]) >= {"reshape", "contiguous", "slice"}


def test_case_selection_uses_config_indices():
    cfg = {"case_grid": {"n": [4, 1, 16, 8, 2], "dtype": ["fp16", "fp32"]}}
    idx = dict(cases.indexed_cases("vector_add", cfg))
    smoke = cases.smoke_cases("vector_add", cfg)
    assert {idx[i]["dtype"] for i in smoke} == {"fp16", "fp32"}
    assert all(idx[i]["n"] == 1 for i in smoke)
    pilot = cases.pilot_cases("vector_add", cfg)
    assert idx[pilot["fp16"]["small"]]["n"] == 1
    assert idx[pilot["fp16"]["medium"]]["n"] == 4
    assert idx[pilot["fp16"]["large"]]["n"] == 16
    assert idx[pilot["fp32"]["large"]]["dtype"] == "fp32"


# ---------------------------------------------------------------- report
def test_report_keeps_failures_visible(tmp_path):
    from tilebench.neuron_diag import report

    s = RunStore("r7", root=tmp_path)
    s.append(_rec(operator="vector_add", mode="native_torch_eager", status="correctness_failure",
                  verification_status="fail", wall_ms=None, wall_timing_method=None,
                  reason="mismatch"), env_hash="e")
    s.append(_rec(operator="mul2", mode="native_nki", status="blocked_env", wall_ms=None,
                  wall_timing_method=None, verification_status="not_checked",
                  reason="no native stack"), env_hash="e")
    s.append(_rec(operator="mul2", mode="native_torch_compiled", status="compile_failure", wall_ms=None,
                  wall_timing_method=None, verification_status="not_checked",
                  reason="compile-diagnostic-row"), env_hash="e")
    text = report.write_report(s).read_text()
    assert "correctness_failure" in text and "blocked_env" in text
    assert "mismatch" in text and "no native stack" in text
    assert "compile-diagnostic-row" not in text and "native_torch_compiled" not in text  # diagnostic only


def test_benchmark_report_excludes_legacy_xla_rows(tmp_path):
    from tilebench.neuron_diag import report

    s = RunStore("r8", root=tmp_path)
    s.append(_rec(operator="vector_add", mode="xla_torch", wall_ms=2.0), env_hash="e")
    s.append(_rec(operator="vector_add", mode="xla_nki", status="runtime_failure", wall_ms=None,
                  wall_timing_method=None, verification_status="not_checked",
                  reason="legacy-row"), env_hash="e")
    s.append(_rec(operator="vector_add", mode="native_torch_eager", wall_ms=0.5,
                  fallback_status="unable_to_determine"), env_hash="e")
    s.append(_rec(operator="vector_add", mode="native_nki", status="runtime_failure", wall_ms=None,
                  wall_timing_method=None, verification_status="not_checked",
                  reason="native-failure"), env_hash="e")
    assert set(k[2] for k in report.benchmark_records(s.records())) == {"native_torch_eager", "native_nki"}
    bench = report.write_report(s)
    text = bench.read_text()
    assert bench.name == "report.md" and bench.parent == s.dir
    assert "xla_torch" not in text and "xla_nki" not in text and "legacy-row" not in text
    assert "native-failure" in text
    legacy = report.write_report(s, legacy_xla=True)
    assert legacy == s.dir / "legacy_xla" / "report.md"
    ltext = legacy.read_text()
    assert "LEGACY XLA" in ltext and "legacy-row" in ltext and "native-failure" not in ltext


def test_archived_legacy_run_is_frozen(tmp_path):
    from tilebench.neuron_diag import report
    from tilebench.neuron_diag.store import ARCHIVE_MARKER

    s = RunStore("r9", root=tmp_path)
    s.append(_rec(mode="xla_nki"), env_hash="e")
    (s.dir / ARCHIVE_MARKER).write_text("{}")
    with pytest.raises(ValueError):
        s.append(_rec(mode="native_nki"), env_hash="e")
    with pytest.raises(ValueError):
        report.write_report(s)
    assert report.write_report(s, legacy_xla=True).is_file()


def test_run_refuses_legacy_modes_without_flag(tmp_path):
    sys.path.insert(0, str(REPO / "scripts"))
    import neuron_diag

    with pytest.raises(SystemExit, match="legacy XLA diagnostic modes"):
        neuron_diag.main(["run", "--run-id", "never-created", "--ops", "relu", "--modes", "native_nki,xla_nki"])
    with pytest.raises(SystemExit, match="torch.compile diagnostic"):
        neuron_diag.main(["run", "--run-id", "never-created", "--ops", "relu", "--modes",
                          "native_torch_eager,native_torch_compiled"])
    assert not (neuron_diag.DIAG_ROOT / "never-created").exists() if hasattr(neuron_diag, "DIAG_ROOT") else True


def test_host_scalar_reads_are_not_compute_fallback():
    from tilebench.neuron_diag import report
    from tilebench.neuron_diag.runtime import XlaRuntime

    rt = XlaRuntime.__new__(XlaRuntime)
    rt._fb_before = {"aten::_local_scalar_dense": 1}
    rt._aten_counters = lambda: {"aten::_local_scalar_dense": 6}
    state, ev = rt.fallback_end()
    assert state == "host_scalar_read" and "host round trips" in ev
    rt._aten_counters = lambda: {"aten::_local_scalar_dense": 6, "aten::sort": 5}
    state, ev = rt.fallback_end()
    assert state == "confirmed_cpu_fallback" and "aten::sort" in ev
    old_row = {"status": "cpu_fallback",
               "fallback_evidence": "torch_xla aten:: CPU-fallback counters grew: {'aten::_local_scalar_dense': 5}"}
    assert report.host_read_only(old_row)
    assert not report.host_read_only(dict(old_row, fallback_evidence="... {'aten::sort': 5}"))


def test_old_fallback_names_are_read_as_new_vocabulary():
    d = _rec().to_json()
    for old, new in (("none_found", "no_fallback_observed_with_evidence"),
                     ("undetermined", "unable_to_determine")):
        d["fallback_status"] = old
        assert Record.from_json(d).fallback_status == new
    with pytest.raises(RecordError):
        _rec(fallback_status="maybe").validate()


def test_error_metrics_report_max_errors_and_effective_tolerance():
    import torch

    from tilebench.neuron_diag.worker import _error_metrics

    out = torch.tensor([1.0, 2.0, 3.0])
    ref = torch.tensor([1.0, 2.5, 3.0])
    m = _error_metrics(out, ref, None, None)
    assert m["max_abs_err"] == pytest.approx(0.5)
    assert m["max_rel_err"] == pytest.approx(0.2)
    assert m["tolerance"][0]["atol"] == 1e-5      # fp32 default of the verifier
    assert _error_metrics(out, ref, 0.1, 0.01)["tolerance"][0]["rtol"] == 0.01
