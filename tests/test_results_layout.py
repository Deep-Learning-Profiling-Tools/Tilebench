"""Hardware-scoped results: results/<hardware>/{csv,logs,figures,aggregate,runs}/.

No test here needs a GPU: the benchmark engine is replaced by a stub, and every
write goes to a temporary results root.
"""
import csv
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
    spec = importlib.util.spec_from_file_location(f"_script_{name.replace('/', '_')}", SCRIPTS / f"{name}.py")
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

#: Columns measured on the GPU of the namespace. NKI_COLUMNS are the one legal
#: extension: cross-hardware measurements from AWS Trainium, merged in by
#: `run_bench.py --tile-language nki`, with speedup_nki = torch_nki_ms / nki_ms.
FROZEN = ["params", "dtype", "torch_ms", "triton_ms", "cutile_ms", "speedup_triton",
          "speedup_cutile", "triton_vs_cutile", "tilelang_ms", "speedup_tilelang"]
NKI_COLUMNS = ["torch_nki_ms", "nki_ms", "speedup_nki"]


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def b200_csvs():
    return sorted((REPO / "results" / "B200" / "csv").glob("*.csv"))


def test_b200_csv_schema_allows_only_the_nki_extension():
    for p in b200_csvs():
        header, rows = read_csv(p)
        assert header[:len(FROZEN)] == FROZEN, p.name
        assert header[len(FROZEN):] in ([], NKI_COLUMNS), p.name       # nothing else may be appended
        assert not {"gpu", "device", "hardware"} & set(header), p.name  # the directory names the GPU
        if header[len(FROZEN):]:
            for r in rows:                                              # never torch_ms / nki_ms
                if "nan" not in (r["torch_nki_ms"], r["nki_ms"]):
                    assert float(r["speedup_nki"]) == pytest.approx(
                        float(r["torch_nki_ms"]) / float(r["nki_ms"]), abs=0.006), (p.name, r["params"])


# --------------------------------------------------------------------------
# scripts/run_bench.py
# --------------------------------------------------------------------------

BACKENDS = ("triton", "cutile", "tilelang", "nki")


def fake_results(enabled, torch_ms=2.0):
    """What the engine returns. For an NKI run torch_ms is the torch baseline
    timed on the Neuron device, and nki_ms is half of it (speedup_nki = 2)."""
    rows = []
    for n in (1024, 2048):
        r = {"params": {"n": n}, "problem_size": n, "dtype": "fp16",
             "torch_ms": torch_ms, "torch_stats": {"mean": torch_ms}}
        for b in BACKENDS:
            ms = (torch_ms / 2 if b == "nki" else 1.0) if b in enabled else float("nan")
            r.update({f"{b}_ms": ms, f"{b}_stats": {}, f"{b}_ok": b in enabled, f"{b}_err": None,
                      f"speedup_{b}": torch_ms / ms if b in enabled else 0.0,
                      f"{b}_autotune_cfg": None})
        rows.append(r)
    return rows


@pytest.fixture
def run_bench(results, tmp_path, monkeypatch):
    mod = load_script("run_bench")
    calls = {"backends": [], "logs_dir": [], "torch_ms": 2.0}

    def engine(operator, benchmark_overrides=None, enabled_backends=None, logs_dir=None):
        calls["backends"].append(set(enabled_backends))
        calls["logs_dir"].append(logs_dir)
        return fake_results(enabled_backends, calls["torch_ms"])

    monkeypatch.setattr(mod, "run_benchmark_suite", engine)
    monkeypatch.setattr(mod, "_detected_device", lambda: None)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["run_bench.py", *argv])
        mod.main()
    run.calls = calls
    run.module = mod
    return run


def test_run_bench_default_paths_are_scoped_by_gpu(results):
    mod = load_script("run_bench")
    timing, autotune, summary = mod._default_paths("B200", "mul2", False, ["triton", "cutile"])
    assert timing == results / "B200/logs/time_measurement_logs/mul2_default_triton-cutile.json"
    assert autotune == results / "B200/logs/autotune_logs/mul2_default_triton-cutile.json"
    assert summary == results / "B200/csv/mul2_default.csv"
    assert mod._default_paths("B200", "mul2", True, ["nki"])[2] == results / "B200/csv/mul2_autotune.csv"
    assert mod._default_paths("GH200", "mul2", False, [])[2] == results / "GH200/csv/mul2_default.csv"


def test_run_bench_requires_gpu_and_has_no_default(run_bench, results, capsys):
    with pytest.raises(SystemExit) as e:
        run_bench("--operator", "mul2")
    assert e.value.code == 2 and "required: --gpu" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_bench("--gpu", "../B200", "--operator", "mul2")
    assert not results.exists() and run_bench.calls["backends"] == []


def test_run_bench_writes_only_into_the_gpu_namespace(run_bench, results, capsys):
    run_bench("--gpu", "B200", "--operator", "mul2")
    assert (results / "B200/csv/mul2_default.csv").is_file()
    assert (results / "B200/logs/time_measurement_logs/mul2_default_triton-cutile-tilelang.json").is_file()
    assert (results / "B200/logs/autotune_logs/mul2_default_triton-cutile-tilelang.json").is_file()
    assert sorted(p.name for p in results.iterdir()) == ["B200"]
    assert "GPU/result namespace: B200" in capsys.readouterr().out
    # the default backend set is the GPU one: NKI is not part of it
    assert run_bench.calls["backends"] == [{"triton", "cutile", "tilelang"}]
    header = (results / "B200/csv/mul2_default.csv").read_text().splitlines()[0]
    assert "nki" not in header and "tilelang_ms" in header       # NKI only runs when named


def test_a_benchmark_run_only_writes_results(run_bench, results, monkeypatch, capsys):
    """The runner produces files under results/<gpu>/ and does nothing else: it
    starts no process and performs no Git operation. Backing results up is not
    part of running a benchmark."""
    def forbidden(*args, **kwargs):
        raise AssertionError(f"run_bench.py started a process: {args}")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    run_bench("--gpu", "B200", "--operator", "mul2")
    assert (results / "B200/csv/mul2_default.csv").is_file()
    for flag in ("--no-archive", "--archive"):                     # removed, not kept as a no-op
        with pytest.raises(SystemExit) as e:
            run_bench("--gpu", "B200", "--operator", "mul2", flag)
        assert e.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err


def test_same_operator_on_two_gpus_does_not_collide(run_bench, results):
    run_bench("--gpu", "B200", "--operator", "mul2")
    b200 = results / "B200/csv/mul2_default.csv"
    before = b200.read_bytes()
    run_bench("--gpu", "GH200", "--operator", "mul2", "--autotune")
    assert b200.read_bytes() == before
    assert (results / "GH200/csv/mul2_autotune.csv").is_file()
    assert (results / "GH200/logs/time_measurement_logs/mul2_autotune_triton-cutile-tilelang.json").is_file()
    assert not (results / "GH200/csv/mul2_default.csv").exists()


def test_explicit_output_paths_win_but_the_csv_stays_in_the_namespace(run_bench, results, tmp_path):
    out, tune = tmp_path / "custom/t.json", tmp_path / "custom/a.json"
    run_bench("--gpu", "B200", "--operator", "mul2",
              "--output", str(out), "--autotune-log", str(tune))
    assert out.is_file() and tune.is_file()
    assert not (results / "B200/logs").exists()
    assert (results / "B200/csv/mul2_default.csv").is_file()


def test_tilelang_merge_touches_only_the_current_gpu(run_bench, results):
    for gpu in ("B200", "GH200"):
        run_bench("--gpu", gpu, "--operator", "mul2", "--tile-language", "triton,cutile")
    b200, gh200 = (results / g / "csv/mul2_default.csv" for g in ("B200", "GH200"))
    gh200_before = gh200.read_bytes()
    frozen = [r["triton_ms"] for r in csv.DictReader(b200.open())]

    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "tilelang")

    rows = list(csv.DictReader(b200.open()))
    assert all(r["tilelang_ms"] == "1.0000" for r in rows)       # merged into B200
    assert [r["triton_ms"] for r in rows] == frozen               # frozen columns untouched
    assert gh200.read_bytes() == gh200_before                     # GH200 untouched


def frozen_view(path):
    _, rows = read_csv(path)
    return [[r[c] for c in ("params", "dtype", "torch_ms", "triton_ms", "cutile_ms",
                            "speedup_triton", "speedup_cutile", "triton_vs_cutile")] for r in rows]


@pytest.fixture
def campaign(run_bench, results):
    """B200 and GH200 each hold a frozen torch/triton/cutile CSV (torch_ms = 2.0)."""
    for gpu in ("B200", "GH200"):
        run_bench("--gpu", gpu, "--operator", "mul2", "--tile-language", "triton,cutile")
    return results / "B200/csv/mul2_default.csv", results / "GH200/csv/mul2_default.csv"


def test_nki_merges_into_the_csv_of_its_campaign(run_bench, results, campaign):
    b200, gh200 = campaign
    frozen, gh200_before = frozen_view(b200), gh200.read_bytes()

    run_bench.calls["torch_ms"] = 5.0          # torch on the Neuron device, not the B200's 2.0
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "nki")

    header, rows = read_csv(b200)
    assert header[-3:] == NKI_COLUMNS                              # appended, names unchanged
    for r in rows:
        assert (r["torch_nki_ms"], r["nki_ms"], r["speedup_nki"]) == ("5.0000", "2.5000", "2.00")
        assert r["torch_ms"] == "2.0000"                           # the B200 baseline is a different number
        assert float(r["speedup_nki"]) == float(r["torch_nki_ms"]) / float(r["nki_ms"])
        assert float(r["speedup_nki"]) != float(r["torch_ms"]) / float(r["nki_ms"])    # never torch_ms / nki_ms
    # nki_ms is written as measured: B200 drift scaling (2.0 / 5.0) would have given 1.0
    assert frozen_view(b200) == frozen                             # frozen GPU columns untouched
    assert gh200.read_bytes() == gh200_before                      # another namespace is untouched
    assert sorted(p.name for p in results.iterdir()) == ["B200", "GH200"]


def test_nki_logs_live_in_the_campaign_namespace(run_bench, results, campaign):
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "nki")
    timing = json.loads((results / "B200/logs/time_measurement_logs/mul2_default_nki.json").read_text())
    assert "nki_ms" in timing[0]
    assert (results / "B200/logs/autotune_logs/mul2_default_nki.json").is_file()
    # the engine is told where the Neuron profiling flow keeps its artifacts
    assert run_bench.calls["logs_dir"][-1] == results / "B200" / "logs"


def test_nki_columns_survive_a_later_tilelang_merge(run_bench, campaign):
    b200, _ = campaign
    run_bench.calls["torch_ms"] = 5.0
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "nki")
    nki = [[r[c] for c in NKI_COLUMNS] for r in read_csv(b200)[1]]
    run_bench.calls["torch_ms"] = 2.0
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "tilelang")
    header, rows = read_csv(b200)
    assert [[r[c] for c in NKI_COLUMNS] for r in rows] == nki
    assert "tilelang_ms" in header


def test_nki_runs_only_when_named(run_bench):
    run_bench("--gpu", "B200", "--operator", "mul2")
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "all")
    assert all("nki" not in b for b in run_bench.calls["backends"])
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "nki")
    assert run_bench.calls["backends"][-1] == {"nki"}


def test_nki_profiling_artifacts_are_placed_under_the_namespace_logs():
    import inspect
    from tilebench.core import engine, nki_orchestrator
    logs = paths.results_logs_dir("B200")
    assert engine._nki_artifact_paths(logs) == {
        "base_dir": str(logs / "nki_profiles"),
        "index_path": str(logs / "nki_neff_manifest.jsonl")}
    with pytest.raises(ValueError):
        engine._nki_artifact_paths(None)                          # no silent fallback location
    params = inspect.signature(nki_orchestrator.profile_case_on_neuron).parameters
    assert params["base_dir"].default is inspect.Parameter.empty
    assert params["index_path"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------
# raw JSON names: one file per (GPU, operator, mode, backend selection)
# --------------------------------------------------------------------------

def test_backend_tag_is_canonical_and_order_free():
    from tilebench.backends import backend_tag, parse_backends
    assert backend_tag(["cutile", "triton"]) == backend_tag(["triton", "cutile"]) == "triton-cutile"
    assert backend_tag(["nki", "tilelang", "cutile", "triton"]) == "triton-cutile-tilelang-nki"
    assert backend_tag(["tilelang", "tilelang"]) == "tilelang" and backend_tag([]) == "torch"
    assert parse_backends("cutile, Triton,torch") == ["triton", "cutile"]
    assert parse_backends(None) == parse_backends("all") == ["triton", "cutile", "tilelang"]
    with pytest.raises(ValueError):
        parse_backends("triton,cuda")
    assert paths.timing_log_path("B200", "mul2", "default", ["cutile", "triton"]) == \
        paths.timing_log_path("B200", "mul2", "default", ["triton", "cutile"])
    with pytest.raises(ValueError):
        paths.timing_log_path("B200", "mul2", "latest", ["triton"])       # no such mode


def test_runs_of_one_operator_never_overwrite_each_others_raw_json(run_bench, results):
    runs = [("B200", "triton,cutile", False), ("B200", "triton,cutile", True),
            ("B200", "tilelang", False), ("B200", "nki", False), ("GH200", "triton,cutile", False)]
    expected = {}
    for gpu, backends, autotune in runs:
        run_bench("--gpu", gpu, "--operator", "mul2", "--tile-language", backends,
                  *(["--autotune"] if autotune else []))
        name = f"mul2_{'autotune' if autotune else 'default'}_{backends.replace(',', '-')}.json"
        for kind in ("time_measurement_logs", "autotune_logs"):
            path = results / gpu / "logs" / kind / name
            expected[path] = path.read_bytes()                     # as written by its own run
    assert len(expected) == 10
    for path, content in expected.items():                         # ...and untouched by every later run
        assert path.read_bytes() == content, path
    # NKI's raw JSON belongs to the B200 campaign namespace
    assert (results / "B200/logs/time_measurement_logs/mul2_default_nki.json").is_file()
    written = sorted(p.name for p in (results / "B200/logs/time_measurement_logs").iterdir())
    assert written == ["mul2_autotune_triton-cutile.json", "mul2_default_nki.json",
                       "mul2_default_tilelang.json", "mul2_default_triton-cutile.json"]
    assert "latest" not in " ".join(written)


def test_backend_order_on_the_command_line_does_not_change_the_file(run_bench, results):
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "cutile,triton")
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "triton,cutile")
    assert [p.name for p in (results / "B200/logs/time_measurement_logs").iterdir()] == \
        ["mul2_default_triton-cutile.json"]


def test_a_repeated_tilelang_run_with_other_cases_is_not_refused(run_bench, results, monkeypatch):
    """The autotune log is one file per selection, so a TileLang run no longer
    merges into (and can no longer be refused by) a combined log."""
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "tilelang")
    monkeypatch.setattr(run_bench.module, "run_benchmark_suite",
                        lambda *a, **kw: fake_results({"tilelang"})[:1])     # a --case-indices subset
    run_bench("--gpu", "B200", "--operator", "mul2", "--tile-language", "tilelang")
    log = json.loads((results / "B200/logs/autotune_logs/mul2_default_tilelang.json").read_text())
    assert len(log) == 1


# --------------------------------------------------------------------------
# visualize / aggregate / plot_sweep_max / run_bench_all
# --------------------------------------------------------------------------

def test_visualize_reads_and_writes_inside_the_gpu_namespace(results, monkeypatch):
    mod = load_script("visualize")
    src = results / "GH200/logs/time_measurement_logs/mul2_autotune_triton-cutile.json"
    src.parent.mkdir(parents=True)
    src.write_text(json.dumps([{k: v for k, v in r.items() if "nki" not in k}
                               for r in fake_results({"triton", "cutile"})]))
    argv = ["visualize.py", "--gpu", "GH200", "--operator", "mul2", "--metrics", "latency_ms"]
    # the run is named explicitly (order-free); nothing is picked by glob or mtime
    monkeypatch.setattr(sys, "argv", argv + ["--mode", "autotune", "--tile-language", "cutile,triton"])
    mod.main()
    assert list((results / "GH200/figures/mul2").glob("*.png"))
    assert sorted(p.name for p in results.iterdir()) == ["GH200"]
    monkeypatch.setattr(sys, "argv", argv)                        # the default-mode, GPU-backend run does not exist
    with pytest.raises(FileNotFoundError) as e:
        mod.main()
    assert e.value.filename.endswith("mul2_default_triton-cutile-tilelang.json")

    monkeypatch.setattr(sys, "argv", ["visualize.py", "--operator", "mul2"])
    with pytest.raises(SystemExit):                               # --gpu is required
        mod.main()


def test_aggregate_reads_csv_and_writes_aggregate_of_one_gpu(results):
    aggregate_results = load_script("aggregate_results")
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
    with pytest.raises(FileNotFoundError) as e:                   # no GH200 results: B200's are not borrowed
        mod.sweep_max_rows("GH200")
    assert str(results / "GH200" / "csv") in str(e.value)
    # one GH200 result: that operator, at the sweep-max case of its config.yaml, and nothing from B200
    real = next((REPO / "results" / "B200" / "csv").glob("mul2_autotune.csv"))
    (results / "GH200" / "csv").mkdir(parents=True)
    shutil.copy(real, results / "GH200" / "csv" / real.name)
    assert [label for label, _ in mod.sweep_max_rows("GH200")] == ["mul2"]
    with pytest.raises(SystemExit):
        mod.main([])


def test_plot_sweep_max_needs_only_the_configs_and_the_committed_csvs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PROFILING_METADATA_ROOT", tmp_path / "no-metadata")
    rows = load_script("plot_sweep_max").sweep_max_rows("B200")   # the real B200 CSVs
    assert len(rows) == 45
    assert not (tmp_path / "no-metadata").exists()                # no NCU catalogue is read or written


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
                        lambda op, **kw: seen.append(kw) or [{"op": op}])
    monkeypatch.chdir(tmp_path)                                   # main() chdirs; restored on teardown
    monkeypatch.setattr(sys, "argv", ["run_bench_all.py", "--gpu", "GH200", "--operators", "mul2"])
    assert mod.main() == 0
    (summary,) = (results / "GH200/runs").glob("*/summary.json")
    assert json.loads(summary.read_text())["gpu"] == "GH200"
    assert seen == [{"logs_dir": results / "GH200" / "logs"}]      # backend selection is left to the engine
    assert sorted(p.name for p in results.iterdir()) == ["GH200"]

    monkeypatch.setattr(sys, "argv", ["run_bench_all.py", "--operators", "mul2"])
    with pytest.raises(SystemExit):
        mod.main()
