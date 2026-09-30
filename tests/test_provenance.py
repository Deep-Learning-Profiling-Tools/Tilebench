"""tilebench.provenance and where the benchmark scripts record it.

The result JSON formats are unchanged: provenance goes to a sidecar
(run_bench.py) or to an added key of the run manifest (run_bench_all.py).
"""
import json
import shutil
import subprocess
import sys

import pytest

import tilebench.paths as paths
from tilebench import provenance

from test_results_layout import load_script, results, run_bench, fake_results  # noqa: F401

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@e",
                           *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    for rel in ("tilebench/core/engine.py", "scripts/run_bench.py", "results/B200/csv/op.csv"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("v1\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


# --------------------------------------------------------------------------
# source state
# --------------------------------------------------------------------------

@needs_git
def test_clean_checkout_records_the_exact_sha(repo):
    state = provenance.source_state(repo / "tilebench")      # any directory of the checkout
    assert state["git_sha"] == git(repo, "rev-parse", "HEAD") and len(state["git_sha"]) == 40
    assert state["dirty"] is False and state["dirty_files"] == [] and state["untracked_files"] == []


@needs_git
@pytest.mark.parametrize("rel", ["scripts/run_bench.py", "tilebench/core/engine.py"])
def test_a_modified_source_file_anywhere_in_the_checkout_is_dirty(repo, rel):
    (repo / rel).write_text("v2\n")
    state = provenance.source_state(repo / "tilebench")
    assert state["dirty"] is True and state["dirty_files"] == [rel]


@needs_git
def test_a_staged_change_is_dirty(repo):
    (repo / "scripts/run_bench.py").write_text("v2\n")
    git(repo, "add", "scripts/run_bench.py")
    assert provenance.source_state(repo)["dirty"] is True


@needs_git
def test_rewritten_summary_csvs_are_data_not_a_dirty_source(repo):
    """run_bench.py rewrites results/<gpu>/csv/: the next run of the same
    campaign must not report its source as dirty because of that."""
    (repo / "results/B200/csv/op.csv").write_text("new measurement\n")
    assert provenance.source_state(repo)["dirty"] is False


@needs_git
def test_untracked_files_are_listed_but_do_not_make_the_tree_dirty(repo):
    (repo / "tilebench/core/new_helper.py").write_text("x\n")
    state = provenance.source_state(repo)
    assert state["dirty"] is False and state["untracked_files"] == ["tilebench/core/new_helper.py"]


def test_no_checkout_is_recorded_not_raised(tmp_path):
    state = provenance.source_state(tmp_path)
    assert state["git_sha"] is None and state["dirty"] is None and state["error"]


def test_no_git_binary_is_recorded_not_raised(tmp_path, monkeypatch):
    def missing(*a, **kw):
        raise FileNotFoundError("git")
    monkeypatch.setattr(provenance.subprocess, "run", missing)
    state = provenance.source_state(tmp_path)
    assert state["git_sha"] is None and "FileNotFoundError" in state["error"]


# --------------------------------------------------------------------------
# software / device
# --------------------------------------------------------------------------

def test_software_versions():
    import torch
    sw = provenance.software_state()
    assert set(sw) == {"python", "torch", "torch_cuda", "torch_hip", "triton", "tilelang", "cuda_tile"}
    assert sw["python"] == ".".join(map(str, sys.version_info[:3]))
    assert sw["torch"] == torch.__version__
    assert sw["torch_cuda"] == torch.version.cuda and sw["torch_hip"] == torch.version.hip


def test_backends_that_are_not_installed_are_none(monkeypatch):
    monkeypatch.setattr(provenance.importlib.metadata, "packages_distributions", lambda: {})
    monkeypatch.setattr(provenance, "_distribution_version", lambda name: None)
    sw = provenance.software_state()
    assert sw["triton"] is None and sw["tilelang"] is None and sw["cuda_tile"] is None


def test_device_block_keeps_the_label_and_the_detected_device_apart(monkeypatch):
    from tilebench import hardware
    monkeypatch.setattr(hardware, "device_info",
                        lambda: hardware.DeviceInfo("nvidia", "NVIDIA B200", (10, 0), None))
    monkeypatch.setattr(hardware, "detect_arch", lambda: "blackwell")
    dev = provenance.device_state("GH200")                  # label and device disagree: both kept
    assert dev == {"requested_label": "GH200", "name": "NVIDIA B200", "vendor": "nvidia",
                   "arch": "blackwell", "compute_capability": [10, 0], "gcn_arch_name": None}


def test_no_gpu_device_block(monkeypatch):
    from tilebench import hardware
    monkeypatch.setattr(hardware, "device_info", lambda: None)
    monkeypatch.setattr(hardware, "detect_arch", lambda: None)
    dev = provenance.device_state("trn1")
    assert dev["requested_label"] == "trn1" and dev["name"] is None and dev["arch"] is None


def test_collect_is_json_ready():
    record = provenance.collect("SMOKE")
    assert record["schema"] == provenance.SCHEMA
    assert set(record) == {"schema", "created_at_utc", "source", "software", "device", "host"}
    assert record["device"]["requested_label"] == "SMOKE"
    json.dumps(record)


# --------------------------------------------------------------------------
# where the scripts record it
# --------------------------------------------------------------------------

def test_provenance_sidecar_path():
    assert paths.provenance_log_path("GH200", "mul2", "autotune", ["cutile", "triton"]) == \
        paths.RESULTS_ROOT / "GH200/logs/provenance/mul2_autotune_triton-cutile.json"


def test_run_bench_writes_a_sidecar_and_leaves_the_logs_unchanged(run_bench, results):
    run_bench("--gpu", "GH200", "--operator", "mul2", "--tile-language", "triton,cutile", "--autotune")
    name = "mul2_autotune_triton-cutile.json"
    timing = results / "GH200/logs/time_measurement_logs" / name
    autotune = results / "GH200/logs/autotune_logs" / name
    sidecar = results / "GH200/logs/provenance" / name

    # the logs keep their format: a plain list of cases, no metadata mixed in
    assert isinstance(json.loads(timing.read_text()), list)
    assert isinstance(json.loads(autotune.read_text()), list)

    record = json.loads(sidecar.read_text())
    assert record["schema"] == provenance.SCHEMA
    assert record["device"]["requested_label"] == "GH200"
    assert record["source"]["git_sha"] is None or len(record["source"]["git_sha"]) == 40
    assert record["run"] == {
        "script": "scripts/run_bench.py", "operator": "mul2", "mode": "autotune",
        "backends": ["triton", "cutile"], "overrides": {"autotune": True},
        "timing_log": str(timing), "autotune_log": str(autotune),
        "summary_csv": str(results / "GH200/csv/mul2_autotune.csv")}


def test_an_explicit_output_takes_the_sidecar_with_it(run_bench, results, tmp_path):
    out = tmp_path / "custom/t.json"
    run_bench("--gpu", "B200", "--operator", "mul2", "--output", str(out))
    record = json.loads((tmp_path / "custom/t.provenance.json").read_text())
    assert record["run"]["timing_log"] == str(out)
    assert not (results / "B200/logs/provenance").exists()


def test_run_bench_records_the_source_state_of_the_checkout(run_bench, results, monkeypatch):
    seen = []
    monkeypatch.setattr(provenance, "source_state",
                        lambda: seen.append(1) or {"git_sha": "a" * 40, "dirty": True,
                                                    "dirty_files": ["x.py"], "untracked_files": []})
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "tilelang")
    record = json.loads((results / "B200/logs/provenance/mul2_default_tilelang.json").read_text())
    assert seen and record["source"] == {"git_sha": "a" * 40, "dirty": True,
                                         "dirty_files": ["x.py"], "untracked_files": []}


def test_an_empty_run_writes_no_sidecar(run_bench, results, monkeypatch):
    monkeypatch.setattr(run_bench.module, "run_benchmark_suite", lambda *a, **kw: [])
    run_bench("--gpu", "B200", "--operator", "mul2")
    assert not (results / "B200/logs").exists()


def test_run_bench_all_manifest_gains_a_provenance_key(results, tmp_path, monkeypatch):
    mod = load_script("run_bench_all")
    monkeypatch.setattr(mod, "run_benchmark_suite",
                        lambda op, logs_dir=None: fake_results({"triton"}))
    monkeypatch.setattr(sys, "argv", ["run_bench_all.py", "--gpu", "MI300X",
                                      "--results-root", str(tmp_path / "runs"),
                                      "--operators", "mul2"])
    assert mod.main() == 0
    (summary,) = (tmp_path / "runs").glob("*/summary.json")
    manifest = json.loads(summary.read_text())
    # the existing manifest keys are all still there
    assert {"gpu", "created_at_utc", "hostname", "python_executable", "python_version",
            "operators_requested", "operators_succeeded", "operators_failed", "artifacts",
            "finished_at_utc"} <= set(manifest)
    assert manifest["provenance"]["schema"] == provenance.SCHEMA
    assert manifest["provenance"]["device"]["requested_label"] == "MI300X"
