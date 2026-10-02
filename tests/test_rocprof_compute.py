"""ROCm Compute Profiler pipeline: replay of the exact winner, kernel selection,
capture validation, the driver's coverage set and the uploader's plan.

No GPU and no profiler is needed: workloads and PC-sampling results are small
synthetic files with the layout rocprof-compute 3.7.0 writes.
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import tilebench.paths as paths
from tilebench.profiling import ncu_kernel_select as ks
from tilebench.profiling import replay
from tilebench.profiling import rocprof_compute as rc

TOOLS = Path(__file__).resolve().parents[1] / "scripts" / "profiling"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# replay: the exact winner, shared by the probe and both harnesses
# --------------------------------------------------------------------------

def test_resolve_params_injects_the_dtype():
    import torch
    assert replay.resolve_params({"n": 4}, "fp16") == {"n": 4, "dtype": torch.float16}
    assert replay.resolve_params({"n": 4, "dtype": "bf16"}, None)["dtype"] is torch.bfloat16
    with pytest.raises(RuntimeError):
        replay.resolve_params({"n": 4}, "fp4")


def test_apply_winner_routes_prefixed_keys_into_per_kernel_dicts():
    impl = ModuleType("impl")
    kv = {"BLOCK_M": 16, "num_warps": 4}
    impl._DEFAULT_KV_CONFIG = kv
    impl._DEFAULT_OUT_CONFIG = {"BLOCK_M": 16}
    replay.apply_winner(impl, {"kv_BLOCK_M": 32, "kv_num_warps": 8, "out_BLOCK_M": 64}, None)
    assert kv == {"BLOCK_M": 32, "num_warps": 8}                 # mutated in place
    assert impl._DEFAULT_OUT_CONFIG == {"BLOCK_M": 64}
    assert not hasattr(impl, "_DEFAULT_CONFIG")                  # nothing left over


def test_apply_winner_per_dtype_and_single_configs():
    import torch
    impl = ModuleType("impl")
    impl._DEFAULT_CONFIGS = {torch.float16: SimpleNamespace(BLOCK=1, num_warps=4)}
    replay.apply_winner(impl, {"BLOCK": 128}, torch.float16)
    assert vars(impl._DEFAULT_CONFIGS[torch.float16]) == {"BLOCK": 128, "num_warps": 4}

    single = ModuleType("impl")
    single._DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4}
    replay.apply_winner(single, {"BLOCK_SIZE": 512}, None)
    assert single._DEFAULT_CONFIG == {"BLOCK_SIZE": 512, "num_warps": 4}
    replay.apply_winner(single, None, None)                      # no winner: untouched
    assert single._DEFAULT_CONFIG == {"BLOCK_SIZE": 512, "num_warps": 4}


# --------------------------------------------------------------------------
# kernel classification and selection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["Memcpy DtoD (Device -> Device)", "__amd_rocclr_copyBuffer",
                                  "Memset (Device)", "void at::native::fill<float>(float*)"])
def test_runtime_and_aten_kernels_are_not_operator_kernels(name):
    assert ks.is_aux_kernel(name)


@pytest.mark.parametrize("name", ["add_kernel", "radix_scatter_kernel", "first_wave"])
def test_triton_kernels_are_operator_kernels(name):
    assert not ks.is_aux_kernel(name)


def test_kernel_filter_is_start_anchored_and_never_end_anchored():
    rgx = rc.kernel_filter_regex(["phi_kernel", "kv_gemm_kernel", "phi_kernel"])
    assert rgx == "^(?:kv_gemm_kernel|phi_kernel)"
    assert "$" not in rgx                       # a `$` disables rocprof-compute's filter
    with pytest.raises(ValueError):
        rc.kernel_filter_regex([])
    with pytest.raises(ValueError):
        rc.kernel_filter_regex(["void at::native::x<int>"])


def test_profiler_env_appends_the_rocm_lib_dir_and_sets_the_version(tmp_path):
    root = tmp_path / "core-7.14"
    (root / "bin").mkdir(parents=True)
    (root / ".info").mkdir()
    (root / ".info" / "version").write_text("7.14.0\n")
    tool = root / "bin" / "rocprof-compute"
    tool.write_text("")
    env = rc.profiler_env({"LD_LIBRARY_PATH": "/a:/b"}, tool)
    assert env["LD_LIBRARY_PATH"] == f"/a:/b:{root}/lib"      # appended, never prepended
    assert env["ROCM_VER"] == "7.14.0"
    again = rc.profiler_env(env, tool)
    assert again["LD_LIBRARY_PATH"] == env["LD_LIBRARY_PATH"]  # not appended twice
    assert rc.profiler_env({"ROCM_VER": "9"}, tool)["ROCM_VER"] == "9"


# --------------------------------------------------------------------------
# capture validation
# --------------------------------------------------------------------------

def _pass_file(path, launches):
    """A results_<pass>.csv: one row per (dispatch, counter), two counters."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Dispatch_ID", "Kernel_Name", "Start_Timestamp", "Counter_Name", "Counter_Value"])
        for i, name in enumerate(launches):
            for counter in ("SQ_WAVES", "GRBM_GUI_ACTIVE"):
                w.writerow([i, name, 1000 + 10 * i, counter, 1])


def _workload(tmp_path, passes, name="workload"):
    wl = tmp_path / name
    (wl / "perfmon").mkdir(parents=True)
    for f in ("profiling_config.yaml", "sysinfo.csv", "log.txt"):
        (wl / f).write_text("x")
    for p, launches in passes.items():
        _pass_file(wl / f"results_{p}.csv", launches)
    return wl


PIPELINE = ["kernel_A", "kernel_B", "kernel_B", "kernel_C"]


def test_repeated_launches_are_counted_not_deduplicated(tmp_path):
    wl = _workload(tmp_path, {"pmc_perf_0": PIPELINE, "pmc_perf_1": PIPELINE})
    assert rc.dispatch_sequence(wl / "results_pmc_perf_0.csv") == PIPELINE
    v = rc.validate_workload(wl, PIPELINE)
    assert v["ok"] and v["passes"] == 2 and v["captured_names"] == PIPELINE
    assert set(v["launches_per_pass"].values()) == {4}


def test_a_missing_launch_fails_even_when_every_name_is_right(tmp_path):
    wl = _workload(tmp_path, {"pmc_perf_0": PIPELINE, "pmc_perf_1": PIPELINE[:3]})
    v = rc.validate_workload(wl, PIPELINE)
    assert not v["ok"]
    assert any("results_pmc_perf_1.csv" in p and "captured 3 != expected 4" in p for p in v["problems"])


@pytest.mark.parametrize("intruder", ["__amd_rocclr_copyBuffer", "void at::native::fill<float>()"])
def test_a_non_operator_kernel_in_the_capture_fails(tmp_path, intruder):
    wl = _workload(tmp_path, {"pmc_perf_0": [intruder] + PIPELINE})
    v = rc.validate_workload(wl, PIPELINE)
    assert not v["ok"] and any("non-operator" in p for p in v["problems"])


def test_launch_order_is_checked(tmp_path):
    wl = _workload(tmp_path, {"pmc_perf_0": ["kernel_B", "kernel_A", "kernel_B", "kernel_C"]})
    v = rc.validate_workload(wl, PIPELINE)
    assert not v["ok"] and any("order differs at launch 0" in p for p in v["problems"])


def test_an_incomplete_or_empty_workload_fails(tmp_path):
    wl = _workload(tmp_path, {"pmc_perf_0": PIPELINE})
    (wl / "sysinfo.csv").unlink()
    assert not rc.validate_workload(wl, PIPELINE)["ok"]
    empty = _workload(tmp_path, {}, name="empty")
    v = rc.validate_workload(empty, PIPELINE)
    assert not v["ok"] and any("no results_" in p for p in v["problems"])
    assert rc.sequence_problems(PIPELINE, []) == ["nothing captured"]


def _pc_workload(tmp_path, kernels, samples):
    """ps_file_results.json: dispatches 1..n of `kernels`, `samples` as
    (dispatch_id, inst_index, issued)."""
    wl = tmp_path / "pc"
    wl.mkdir()
    symbols = [{"kernel_id": i, "kernel_name": k + ".kd", "formatted_kernel_name": k}
               for i, k in enumerate(sorted(set(kernels)))]
    kid = {s["formatted_kernel_name"]: s["kernel_id"] for s in symbols}
    tool = {"kernel_symbols": symbols,
            "strings": {"pc_sample_instructions": ["s_waitcnt vmcnt(0)", "v_add_f32 v1, v2, v3"],
                        "pc_sample_comments": ["impl_triton.py:16", ""]},
            "buffer_records": {
                "kernel_dispatch": [{"start_timestamp": 10 * i,
                                     "dispatch_info": {"dispatch_id": i, "kernel_id": kid[k]}}
                                    for i, k in enumerate(kernels, 1)],
                "pc_sample_stochastic": [
                    {"inst_index": idx, "record": {
                        "dispatch_id": d, "wave_issued": issued, "inst_type": "TYPE_VALU",
                        "pc": {"code_object_id": 4, "code_object_offset": 0x1768 + idx},
                        "snapshot": {"stall_reason": "REASON_WAITCNT"}}}
                    for d, idx, issued in samples]}}
    (wl / rc.PC_SAMPLING_RESULTS).write_text(json.dumps({"rocprofiler-sdk-tool": [tool]}))
    return wl


def test_pc_sampling_validation_counts_operator_samples_only(tmp_path):
    kernels = ["void at::native::normal<float>()", "kernel_A", "kernel_B", "kernel_B", "kernel_C"]
    wl = _pc_workload(tmp_path, kernels, [(1, 1, 1), (2, 0, 0), (3, 0, 0), (4, 1, 1), (9, 0, 0)])
    v = rc.validate_pc_sampling(wl, PIPELINE)
    assert v["ok"] and v["captured_names"] == PIPELINE
    assert v["samples_total"] == 5 and v["samples_operator"] == 3       # generator + uncorrelated out
    assert v["samples_with_source_line"] == 2 and v["correlation_available"]
    rows = rc.pc_sampling_instruction_rows(wl)
    op_rows = [r for r in rows if r["operator_kernel"]]
    assert sum(r["samples"] for r in op_rows) == 3
    stalled = next(r for r in op_rows if r["instruction"] == "s_waitcnt vmcnt(0)")
    assert stalled["source_line"] == "impl_triton.py:16"
    assert json.loads(stalled["stall_reasons"]) == {"WAITCNT": stalled["stalled"]}


def test_pc_sampling_with_a_wrong_dispatch_sequence_fails(tmp_path):
    wl = _pc_workload(tmp_path, ["kernel_A", "kernel_B", "kernel_C"], [(1, 0, 0)])
    v = rc.validate_pc_sampling(wl, PIPELINE)
    assert not v["ok"] and any("captured 3 != expected 4" in p for p in v["problems"])
    assert not rc.validate_pc_sampling(tmp_path / "absent", PIPELINE)["ok"]


def test_no_sample_in_an_operator_kernel_is_recorded_not_failed(tmp_path):
    wl = _pc_workload(tmp_path, ["kernel_A", "kernel_B", "kernel_B", "kernel_C"], [])
    v = rc.validate_pc_sampling(wl, PIPELINE)
    assert v["ok"] and not v["correlation_available"] and v["samples_operator"] == 0


# --------------------------------------------------------------------------
# paths, catalogue, probe, driver, uploader
# --------------------------------------------------------------------------

def test_rocprof_compute_artifacts_are_one_directory_per_hardware_and_pair():
    assert paths.rocprof_compute_output_dir("MI300X") == paths.OUTPUT_ROOT / "rocprof_compute" / "MI300X"
    assert paths.rocprof_compute_pair_dir("MI300X", "softmax", "triton", "fp16") == \
        paths.OUTPUT_ROOT / "rocprof_compute" / "MI300X" / "softmax" / "triton_fp16"
    with pytest.raises(ValueError):
        paths.rocprof_compute_output_dir("../B200")


@pytest.fixture
def mi300x(tmp_path, monkeypatch):
    """Temporary metadata, results and artifact roots with a Triton-only MI300X
    autotune log for two operators, one of them with an unsupported dtype."""
    for attr, sub in (("PROFILING_METADATA_ROOT", "outputs/profiling"),
                      ("ROCPROF_COMPUTE_OUTPUT_ROOT", "outputs/rocprof_compute"),
                      ("RESULTS_ROOT", "results")):
        monkeypatch.setattr(paths, attr, tmp_path / sub)
    logs = tmp_path / "results" / "MI300X" / "logs" / "autotune_logs"
    logs.mkdir(parents=True)
    from tilebench.profiling.ncu_catalogue import sweep_max_cases
    (logs / "mul2_autotune_triton.json").write_text(json.dumps(
        [{"dtype": dt, "params": {"n": 1 << 20}, "triton_autotune_cfg": {"BLOCK": 256}}
         for dt in sweep_max_cases("mul2")[0]]))                 # every dtype of mul2
    (logs / "matmul_fp32_fp16_fp8_autotune_triton.json").write_text(json.dumps(
        [{"dtype": dt, "params": {"M": 64, "N": 64, "K": 64}, "triton_autotune_cfg": {"BLOCK_M": 64}}
         for dt in ("fp16", "fp32")]))                            # fp8_e4m3fn: unsupported, no entry
    return tmp_path


def test_a_triton_only_catalogue_is_accepted(mi300x):
    ncu_catalogue = load_tool("ncu_catalogue")
    ncu_catalogue.main(["--gpu", "MI300X", "--tile-language", "triton",
                        "mul2", "matmul_fp32_fp16_fp8"])
    cat = {c["op"]: c for c in ks.load_catalogue("MI300X")}
    assert cat["mul2"]["autotune_winner_per_dtype"]["fp16"] == \
        {"params": {"n": 1 << 20}, "triton": {"BLOCK": 256}, "cutile": None}
    assert "fp8_e4m3fn" in cat["matmul_fp32_fp16_fp8"]["dtypes"]
    assert "fp8_e4m3fn" not in cat["matmul_fp32_fp16_fp8"]["autotune_winner_per_dtype"]
    with pytest.raises(SystemExit):                          # still needs a profiled backend
        ncu_catalogue.main(["--gpu", "MI300X", "--tile-language", "tilelang", "mul2"])


def test_the_probe_probes_only_the_selected_backends_and_skips_unsupported_dtypes(mi300x, monkeypatch):
    load_tool("ncu_catalogue").main(["--gpu", "MI300X", "--tile-language", "triton",
                                     "mul2", "matmul_fp32_fp16_fp8"])
    probe = load_tool("probe_kernel_count")
    calls = []
    monkeypatch.setattr(probe, "count_one", lambda op, be, params, cfg, dt: calls.append((op, be, dt, cfg)) or
                        {"count": 1, "names": ["k"], "excluded": [], "first_call_identical": True})
    monkeypatch.setattr(sys, "argv", ["probe", "--gpu", "MI300X", "--tile-language", "triton"])
    probe.main()
    assert {be for _, be, _, _ in calls} == {"triton"}
    assert ("matmul_fp32_fp16_fp8", "triton", "fp8_e4m3fn") not in {c[:3] for c in calls}
    rows = {(r["op"], r["dtype"]): r for r in
            json.loads(paths.kernel_counts_path("MI300X").read_text())}
    assert rows[("matmul_fp32_fp16_fp8", "fp8_e4m3fn")]["count"] is None
    assert "skipped" in rows[("matmul_fp32_fp16_fp8", "fp8_e4m3fn")]
    assert rows[("mul2", "fp16")] == {"op": "mul2", "dtype": "fp16", "backend": "triton", "count": 1,
                                      "names": ["k"], "excluded": [], "first_call_identical": True}
    monkeypatch.setattr(sys, "argv", ["probe", "--gpu", "MI300X", "--tile-language", "tilelang"])
    with pytest.raises(SystemExit):
        probe.main()


def test_the_driver_covers_every_valid_pair_and_lists_the_excluded_ones(mi300x):
    load_tool("ncu_catalogue").main(["--gpu", "MI300X", "--tile-language", "triton",
                                     "mul2", "matmul_fp32_fp16_fp8"])
    driver = load_tool("rocprof_compute_driver")
    pairs, excluded = driver.valid_pairs(ks.load_catalogue("MI300X"), set(), set())
    from tilebench.profiling.ncu_catalogue import sweep_max_cases
    assert sorted((op, dt) for op, dt, _, _ in pairs) == sorted(
        [("matmul_fp32_fp16_fp8", "fp16"), ("matmul_fp32_fp16_fp8", "fp32")]
        + [("mul2", dt) for dt in sweep_max_cases("mul2")[0]])
    assert [(e["op"], e["dtype"]) for e in excluded] == [("matmul_fp32_fp16_fp8", "fp8_e4m3fn")]
    only, _ = driver.valid_pairs(ks.load_catalogue("MI300X"), {"mul2"}, {"fp32"})
    assert [(op, dt) for op, dt, _, _ in only] == [("mul2", "fp32")]


def test_the_harness_is_started_next_to_the_driver():
    driver = load_tool("rocprof_compute_driver")
    assert driver.HARNESS == TOOLS / "rocprof_compute_harness.py" and driver.HARNESS.is_file()


def _sweep(root, entries):
    root.mkdir(parents=True, exist_ok=True)
    (root / "sweep_log.json").write_text(json.dumps(entries))


def test_the_uploader_sends_only_validated_pairs_as_whole_directories(mi300x, tmp_path):
    pytest.importorskip("huggingface_hub")
    up = load_tool("hf_upload_rocm_compute")
    root = paths.rocprof_compute_output_dir("MI300X")
    good = root / "mul2" / "triton_fp16"
    _workload(good, {"pmc_perf_0": ["mul2_kernel"]})
    _workload(root / "mul2" / "triton_fp32", {"pmc_perf_0": ["mul2_kernel", "mul2_kernel"]})
    (good / "analysis").mkdir()
    (good / "analysis" / "workload_report.txt").write_text("report")
    entry = {"op": "mul2", "backend": "triton", "pc_sampling": False, "expected_names": ["mul2_kernel"]}
    _sweep(root, [{**entry, "dtype": "fp16", "ok": True},
                  {**entry, "dtype": "fp32", "ok": True},                     # does not validate
                  {**entry, "dtype": "bf16", "ok": False}])
    pairs, refused = up.uploadable_pairs("MI300X")
    assert pairs == ["mul2/triton_fp16"]
    assert len(refused) == 2
    plan = up.upload_plan("MI300X", "AMD_MI300X", pairs)
    assert plan["path_in_repo"] == "AMD_MI300X" and plan["folder_path"] == str(root)
    assert plan["allow_patterns"] == ["mul2/triton_fp16/*", "sweep_log.json", "coverage.json"]
    assert not any(p.startswith(("NVIDIA", "Neuron")) for p in plan["allow_patterns"])
    files = up.local_files("MI300X", pairs)                  # every file, any extension
    assert "mul2/triton_fp16/workload/perfmon" not in files
    assert {"mul2/triton_fp16/workload/log.txt", "mul2/triton_fp16/analysis/workload_report.txt",
            "mul2/triton_fp16/workload/results_pmc_perf_0.csv"} <= set(files)
    assert up.uploadable_pairs("MI300X", "softmax") == ([], [])


def test_the_rocm_uploader_needs_a_folder_and_a_token(mi300x, monkeypatch, capsys):
    pytest.importorskip("huggingface_hub")
    up = load_tool("hf_upload_rocm_compute")
    monkeypatch.setattr(sys, "argv", ["up", "--gpu", "MI300X"])
    with pytest.raises(SystemExit) as e:
        up.main()
    assert e.value.code == 2 and "--hf-folder" in capsys.readouterr().err
    monkeypatch.delenv("HUGGING_FACE", raising=False)
    monkeypatch.setattr(sys, "argv", ["up", "--gpu", "MI300X", "--hf-folder", "AMD_MI300X"])
    with pytest.raises(SystemExit) as e:
        up.main()
    assert "HUGGING_FACE" in str(e.value)
    for bad in ("../x", "a/b", "NVIDIA GH200"):
        monkeypatch.setattr(sys, "argv", ["up", "--gpu", "MI300X", "--hf-folder", bad])
        with pytest.raises(SystemExit):
            up.main()


def test_the_ncu_uploader_stays_ncu_only():
    text = (TOOLS / "hf_upload.py").read_text()
    assert 'ALLOW_PATTERNS = ["*.ncu-rep", "**/*.ncu-rep"]' in text
    assert "rocprof" not in text
