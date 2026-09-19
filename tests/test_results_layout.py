"""Hardware-scoped results: results/<hardware>/{csv,logs,figures,aggregate,runs}/.

No test here needs a GPU: the benchmark engine is replaced by a stub, and every
write goes to a temporary results root.
"""
import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tilebench.paths as paths

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def results(tmp_path, monkeypatch):
    """Point the results tree at a temporary directory."""
    root = tmp_path / "results"
    monkeypatch.setattr(paths, "RESULTS_ROOT", root)
    return root


# --------------------------------------------------------------------------
# path helpers
# --------------------------------------------------------------------------

def test_results_helpers_map_into_the_hardware_namespace():
    assert paths.results_root("B200") == paths.REPO_ROOT / "results" / "B200"
    assert paths.results_csv_dir("B200") == paths.REPO_ROOT / "results" / "B200" / "csv"
    assert paths.results_csv_dir("GH200") == paths.REPO_ROOT / "results" / "GH200" / "csv"
    assert paths.results_logs_dir("B200").name == "logs"
    assert paths.results_figures_dir("B200").name == "figures"
    assert paths.results_aggregate_dir("B200").name == "aggregate"
    assert paths.results_runs_dir("B200").name == "runs"


@pytest.mark.parametrize("label", ["B200", "GH200", "MI300X", "MI325X", "H100-SXM5", "rtx_6000", "A100.80GB"])
def test_any_safe_label_is_accepted_without_a_device_list(label):
    assert paths.hardware_label(label) == label
    assert paths.results_csv_dir(label).parent.name == label


@pytest.mark.parametrize("label", ["", "..", ".", "../B200", "B200/..", "/", "/abs", "a/b", "a\\b",
                                   "B200/", ".hidden", "-rf", "B 200", "B200\n", None, 200])
def test_unsafe_labels_are_rejected(label):
    with pytest.raises(ValueError):
        paths.hardware_label(label)
    with pytest.raises(ValueError):
        paths.results_root(label)


# --------------------------------------------------------------------------
# .gitignore
# --------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_gitignore_tracks_only_the_per_hardware_summary_csvs(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    shutil.copy(REPO / ".gitignore", tmp_path / ".gitignore")

    def ignored(rel):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
        return subprocess.run(["git", "check-ignore", "-q", rel], cwd=tmp_path).returncode == 0

    for rel in ["results/B200/csv/x.csv", "results/GH200/csv/x.csv", "results/MI300X/csv/x.csv"]:
        assert not ignored(rel), rel
    for rel in ["results/B200/logs/x.json", "results/GH200/figures/x.png",
                "results/MI300X/aggregate/x.csv",       # a generated CSV, still ignored
                "results/B200/runs/x.json",
                "results/B200/csv/notes.txt",           # only *.csv is tracked
                "results/B200/x.csv", "results/x.csv",  # not under <hardware>/csv/
                "results/logs/x.json"]:                 # legacy layout
        assert ignored(rel), rel
    rules = (REPO / ".gitignore").read_text()
    assert not any(gpu in rules for gpu in ("B200", "GH200", "MI300X")), "no hardware name in .gitignore"


# --------------------------------------------------------------------------
# the committed B200 data
# --------------------------------------------------------------------------

def test_b200_csvs_are_byte_identical_to_the_pre_migration_files():
    """tests/data/b200_csv.sha256 was recorded from results/csv/ BEFORE the move
    to results/B200/csv/. The CSVs are frozen paper results: regenerate the
    manifest only when they are intentionally re-measured:
        (cd results/B200/csv && sha256sum *.csv) > tests/data/b200_csv.sha256
    """
    expected = dict(reversed(line.split()) for line in
                    (REPO / "tests/data/b200_csv.sha256").read_text().splitlines())
    csv_dir = REPO / "results" / "B200" / "csv"
    actual = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in csv_dir.glob("*.csv")}
    assert len(expected) == 90
    assert actual == expected
    assert not (REPO / "results" / "csv").exists(), "the pre-migration directory is gone"


def test_b200_csvs_hold_gpu_columns_only():
    for p in sorted((REPO / "results" / "B200" / "csv").glob("*.csv")):
        header = p.read_text().splitlines()[0].split(",")
        assert not [c for c in header if "nki" in c], p.name           # no cross-hardware columns
        assert not {"gpu", "device", "hardware"} & set(header), p.name  # the directory names the GPU


# --------------------------------------------------------------------------
# scripts/run_bench.py
# --------------------------------------------------------------------------

BACKENDS = ("triton", "cutile", "tilelang", "nki")


def fake_results(enabled, torch_ms=2.0):
    rows = []
    for n in (1024, 2048):
        r = {"params": {"n": n}, "problem_size": n, "dtype": "fp16",
             "torch_ms": torch_ms, "torch_stats": {"mean": torch_ms}}
        for b in BACKENDS:
            ms = 1.0 if b in enabled else float("nan")
            r.update({f"{b}_ms": ms, f"{b}_stats": {}, f"{b}_ok": b in enabled, f"{b}_err": None,
                      f"speedup_{b}": torch_ms / ms if b in enabled else 0.0,
                      f"{b}_autotune_cfg": None})
        rows.append(r)
    return rows


@pytest.fixture
def run_bench(results, tmp_path, monkeypatch):
    mod = load_script("run_bench")
    calls = {"backends": [], "archived": []}

    def engine(operator, benchmark_overrides=None, enabled_backends=None):
        calls["backends"].append(set(enabled_backends))
        return fake_results(enabled_backends)

    monkeypatch.setattr(mod, "run_benchmark_suite", engine)
    monkeypatch.setattr(mod, "_archive_logs", lambda gpu: calls["archived"].append(gpu))
    monkeypatch.setattr(mod, "_detected_device", lambda: None)
    monkeypatch.setattr(mod, "NKI_OUTPUT_ROOT", tmp_path / "outputs" / "nki")

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["run_bench.py", *argv])
        mod.main()
    run.calls = calls
    run.module = mod
    return run


def test_run_bench_default_paths_are_scoped_by_gpu(results):
    mod = load_script("run_bench")
    timing, autotune, summary = mod._default_paths("B200", "mul2", autotune=False)
    assert timing == results / "B200/logs/time_measurement_logs/mul2_results.json"
    assert autotune == results / "B200/logs/autotune_logs/mul2_autotune.json"
    assert summary == results / "B200/csv/mul2_default.csv"
    assert mod._default_paths("B200", "mul2", autotune=True)[2] == results / "B200/csv/mul2_autotune.csv"
    assert mod._default_paths("GH200", "mul2", autotune=False)[2] == results / "GH200/csv/mul2_default.csv"


def test_run_bench_requires_gpu_and_has_no_default(run_bench, results, capsys):
    with pytest.raises(SystemExit) as e:
        run_bench("--operator", "mul2")
    assert e.value.code == 2 and "--gpu is required" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_bench("--gpu", "../B200", "--operator", "mul2")
    assert not results.exists() and run_bench.calls["backends"] == []


def test_run_bench_writes_into_the_gpu_namespace_and_archives_only_that_gpu(run_bench, results, capsys):
    run_bench("--gpu", "B200", "--operator", "mul2")
    assert (results / "B200/csv/mul2_default.csv").is_file()
    assert (results / "B200/logs/time_measurement_logs/mul2_results.json").is_file()
    assert (results / "B200/logs/autotune_logs/mul2_autotune.json").is_file()
    assert sorted(p.name for p in results.iterdir()) == ["B200"]
    assert run_bench.calls["archived"] == ["B200"]
    assert "GPU/result namespace: B200" in capsys.readouterr().out
    # the default backend set is the GPU one: NKI is not part of it
    assert run_bench.calls["backends"] == [{"triton", "cutile", "tilelang"}]
    header = (results / "B200/csv/mul2_default.csv").read_text().splitlines()[0]
    assert "nki" not in header and "tilelang_ms" in header


def test_same_operator_on_two_gpus_does_not_collide(run_bench, results):
    run_bench("--gpu", "B200", "--operator", "mul2", "--no-archive")
    b200 = results / "B200/csv/mul2_default.csv"
    before = b200.read_bytes()
    run_bench("--gpu", "GH200", "--operator", "mul2", "--autotune", "--no-archive")
    assert b200.read_bytes() == before
    assert (results / "GH200/csv/mul2_autotune.csv").is_file()
    assert (results / "GH200/logs/time_measurement_logs/mul2_results.json").is_file()
    assert not (results / "GH200/csv/mul2_default.csv").exists()


def test_explicit_output_paths_win_but_the_csv_stays_in_the_namespace(run_bench, results, tmp_path):
    out, tune = tmp_path / "custom/t.json", tmp_path / "custom/a.json"
    run_bench("--gpu", "B200", "--operator", "mul2", "--no-archive",
              "--output", str(out), "--autotune-log", str(tune))
    assert out.is_file() and tune.is_file()
    assert not (results / "B200/logs").exists()
    assert (results / "B200/csv/mul2_default.csv").is_file()


def test_tilelang_merge_touches_only_the_current_gpu(run_bench, results):
    for gpu in ("B200", "GH200"):
        run_bench("--gpu", gpu, "--operator", "mul2", "--tile-language", "triton,cutile", "--no-archive")
    b200, gh200 = (results / g / "csv/mul2_default.csv" for g in ("B200", "GH200"))
    gh200_before = gh200.read_bytes()
    frozen = [r["triton_ms"] for r in csv.DictReader(b200.open())]

    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "tilelang", "--no-archive")

    rows = list(csv.DictReader(b200.open()))
    assert all(r["tilelang_ms"] == "1.0000" for r in rows)       # merged into B200
    assert [r["triton_ms"] for r in rows] == frozen               # frozen columns untouched
    assert gh200.read_bytes() == gh200_before                     # GH200 untouched


@pytest.mark.parametrize("selection", ["nki", "triton,nki", "tilelang,nki"])
def test_nki_never_enters_a_gpu_namespace(run_bench, results, selection):
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "triton,cutile", "--no-archive")
    frozen = (results / "B200/csv/mul2_default.csv").read_bytes()
    with pytest.raises(SystemExit) as e:
        run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", selection, "--no-archive")
    assert e.value.code == 2
    assert (results / "B200/csv/mul2_default.csv").read_bytes() == frozen


def test_nki_alone_writes_outside_results(run_bench, results, tmp_path):
    run_bench("--operator", "mul2", "--tile-language", "nki")
    nki = tmp_path / "outputs" / "nki"
    assert (nki / "csv/mul2_default.csv").read_text().splitlines()[0] == \
        "params,dtype,torch_ms,nki_ms,speedup_nki"
    assert (nki / "logs/time_measurement_logs/mul2_results.json").is_file()
    assert not results.exists()                                   # nothing under results/<gpu>/
    assert run_bench.calls["archived"] == []                      # GPU log archive is not involved


def test_all_means_every_gpu_backend(run_bench):
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "all", "--no-archive")
    assert run_bench.calls["backends"] == [{"triton", "cutile", "tilelang"}]


def test_run_bench_passes_its_gpu_to_the_archive_script(monkeypatch):
    mod = load_script("run_bench")
    seen = {}
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **kw: seen.update(cmd=cmd) or
                        subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr=""))
    mod._archive_logs("GH200")
    assert seen["cmd"][-2:] == ["--gpu", "GH200"] and seen["cmd"][1].endswith("archive_logs.sh")


# --------------------------------------------------------------------------
# visualize / aggregate / plot_sweep_max / run_bench_all
# --------------------------------------------------------------------------

def test_visualize_reads_and_writes_inside_the_gpu_namespace(results, monkeypatch):
    mod = load_script("visualize")
    src = results / "GH200/logs/time_measurement_logs/mul2_results.json"
    src.parent.mkdir(parents=True)
    src.write_text(json.dumps([{k: v for k, v in r.items() if "nki" not in k}
                               for r in fake_results({"triton", "cutile"})]))
    monkeypatch.setattr(sys, "argv", ["visualize.py", "--gpu", "GH200", "--operator", "mul2",
                                      "--metrics", "latency_ms"])
    mod.main()
    assert list((results / "GH200/figures/mul2").glob("*.png"))
    assert sorted(p.name for p in results.iterdir()) == ["GH200"]

    monkeypatch.setattr(sys, "argv", ["visualize.py", "--operator", "mul2"])
    with pytest.raises(SystemExit):                               # --gpu is required
        mod.main()


def test_aggregate_reads_csv_and_writes_aggregate_of_one_gpu(results):
    from tilebench.profiling import aggregate_results
    src = results / "B200/csv/mul2_default.csv"
    src.parent.mkdir(parents=True)
    src.write_text("params,dtype,torch_ms,triton_ms,cutile_ms\nn=1,fp16,2.0,1.0,1.0\nn=2,fp16,2.0,1.0,1.0\n")
    aggregate_results.main(["--gpu", "B200"])
    out = results / "B200/aggregate/mul2.csv"
    assert list(csv.DictReader(out.open()))[0]["speedup_triton"]
    assert sorted(p.name for p in results.iterdir()) == ["B200"]
    with pytest.raises(SystemExit):
        aggregate_results.main([])


def test_plot_sweep_max_selects_the_gpu(results):
    mod = load_script("plot_sweep_max")
    with pytest.raises(FileNotFoundError) as e:                   # no GH200 data in the temp tree
        mod.sweep_max_rows("GH200")
    assert str(results / "GH200" / "csv") in str(e.value.filename)
    with pytest.raises(SystemExit):
        mod.main([])


def test_plot_sweep_max_finds_every_committed_b200_operator():
    rows = load_script("plot_sweep_max").sweep_max_rows("B200")   # the real, migrated CSVs
    assert len(rows) == 45


def test_plot_sweep_max_default_output_cannot_clobber_the_readme_figure(results, monkeypatch):
    mod = load_script("plot_sweep_max")
    monkeypatch.setattr(mod, "sweep_max_rows", lambda gpu: [
        (f"op{i}", {"torch_ms": 3.0 + i, "triton_ms": 2.0, "cutile_ms": 1.0}) for i in range(4)])
    mod.main(["--gpu", "GH200"])
    assert (results / "GH200/figures/sweep_max_latency.png").is_file()


def test_run_bench_all_scopes_its_runs_and_records_the_gpu(results, monkeypatch, tmp_path):
    mod = load_script("run_bench_all")
    seen = []
    monkeypatch.setattr(mod, "run_benchmark_suite",
                        lambda op, enabled_backends=None: seen.append(enabled_backends) or [{"op": op}])
    monkeypatch.chdir(tmp_path)                                   # main() chdirs; restored on teardown
    monkeypatch.setattr(sys, "argv", ["run_bench_all.py", "--gpu", "GH200", "--operators", "mul2"])
    assert mod.main() == 0
    (summary,) = (results / "GH200/runs").glob("*/summary.json")
    assert json.loads(summary.read_text())["gpu"] == "GH200"
    assert seen == [{"triton", "cutile", "tilelang"}]
    assert sorted(p.name for p in results.iterdir()) == ["GH200"]

    monkeypatch.setattr(sys, "argv", ["run_bench_all.py", "--operators", "mul2"])
    with pytest.raises(SystemExit):
        mod.main()
