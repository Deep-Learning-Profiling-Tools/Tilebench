"""Hardware-scoped NCU profiling metadata and outputs, both generated data:
    outputs/profiling/<hardware>/{ncu_catalogue,kernel_counts}.json
    outputs/ncu/<hardware>/
No test here needs a GPU or the ncu binary.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tilebench.paths as paths
from tilebench.profiling import ncu_kernel_select as ks

REPO = Path(__file__).resolve().parents[1]

@pytest.fixture
def metadata(tmp_path, monkeypatch):
    """A temporary metadata tree holding B200 metadata (a catalogue built from
    the operator configs, and probed-looking kernel counts), with temporary
    results and NCU output roots. Nothing is committed for any GPU."""
    from tilebench.profiling import ncu_catalogue
    root = tmp_path / "outputs" / "profiling"
    monkeypatch.setattr(paths, "PROFILING_METADATA_ROOT", root)
    monkeypatch.setattr(paths, "NCU_OUTPUT_ROOT", tmp_path / "outputs" / "ncu")
    monkeypatch.setattr(paths, "RESULTS_ROOT", tmp_path / "results")
    ncu_catalogue.main(["--gpu", "B200"])
    rows = [{"op": c["op"], "dtype": dt, "backend": be, "count": 1, "names": [f"{c['op']}_kernel"]}
            for c in json.loads((root / "B200" / "ncu_catalogue.json").read_text())
            for dt in c["dtypes"] for be in ("triton", "cutile")]
    (root / "B200" / "kernel_counts.json").write_text(json.dumps(rows, indent=2))
    return root


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def test_metadata_and_ncu_outputs_are_one_directory_per_hardware():
    base = paths.REPO_ROOT / "outputs" / "profiling"           # generated data, outside the package
    assert paths.PACKAGE_ROOT not in base.parents
    assert paths.ncu_catalogue_path("B200") == base / "B200" / "ncu_catalogue.json"
    assert paths.kernel_counts_path("B200") == base / "B200" / "kernel_counts.json"
    assert paths.ncu_catalogue_path("GH200") == base / "GH200" / "ncu_catalogue.json"
    assert paths.kernel_counts_path("MI300X").parent == paths.profiling_metadata_dir("MI300X")
    assert paths.ncu_output_dir("B200") == paths.REPO_ROOT / "outputs" / "ncu" / "B200"
    assert paths.ncu_output_dir("B200") != paths.ncu_output_dir("GH200")
    for helper in (paths.profiling_metadata_dir, paths.ncu_catalogue_path,
                   paths.kernel_counts_path, paths.ncu_output_dir):
        for label in ("../B200", "B200/..", "", "/", "a\\b"):
            with pytest.raises(ValueError):
                helper(label)


def test_the_source_package_holds_no_profiling_data_or_cluster_scripts():
    assert not hasattr(paths, "NCU_CATALOGUE") and not hasattr(paths, "KERNEL_COUNTS")
    stray = [p.relative_to(REPO).as_posix() for p in (REPO / "tilebench" / "profiling").rglob("*")
             if p.is_file() and p.suffix in {".json", ".sh", ".sbatch"}]
    assert stray == []


# --------------------------------------------------------------------------
# readers: a GPU without metadata is an error, never a fallback to B200
# --------------------------------------------------------------------------

def test_b200_metadata_loads(metadata):
    counts, names = ks.load_kernel_counts("B200")
    assert counts and names
    assert len(ks.load_catalogue("B200")) == 45


def test_missing_gpu_metadata_fails_loudly_without_falling_back(metadata):
    with pytest.raises(ks.MissingKernelCountsError) as e:
        ks.load_kernel_counts("GH200")
    assert "GH200" in str(e.value) and "never used as a fallback" in str(e.value)
    assert "--gpu GH200" in str(e.value)                      # how to produce it, for THIS gpu
    with pytest.raises(ks.MissingProfilingMetadataError) as e:
        ks.load_catalogue("GH200")
    assert "GH200" in str(e.value) and "never used as a fallback" in str(e.value)


@pytest.mark.parametrize("module", ["ncu_one", "ncu_driver", "ncu_writeup"])
def test_ncu_tools_require_a_gpu_and_refuse_one_without_metadata(metadata, monkeypatch, module, capsys):
    import importlib
    mod = importlib.import_module(f"tilebench.profiling.{module}")
    extra = ["mul2"] if module == "ncu_one" else []
    monkeypatch.setattr(sys, "argv", [module, *extra])
    with pytest.raises(SystemExit) as e:                      # no B200 default
        mod.main()
    assert e.value.code == 2 and "--gpu" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [module, "--gpu", "GH200", *extra])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "GH200" in str(e.value.code) and "fallback" in str(e.value.code)
    assert not (metadata / "GH200").exists()                  # nothing was written for it
    assert not (metadata.parent / "ncu").exists()


def test_ncu_reports_of_two_gpus_cannot_collide(metadata):
    from tilebench.profiling import ncu_driver
    b200 = ncu_driver.out_path(paths.ncu_output_dir("B200"), "mul2", "triton", "fp16")
    gh200 = ncu_driver.out_path(paths.ncu_output_dir("GH200"), "mul2", "triton", "fp16")
    assert b200 != gh200 and b200.name == gh200.name == "triton_fp16.ncu-rep"
    assert b200.parts[-4:-1] == ("ncu", "B200", "mul2")


# --------------------------------------------------------------------------
# writers: each GPU's files only
# --------------------------------------------------------------------------

def _autotune_log(gpu, op, mode, backends, triton_cfg):
    entry = next(c for c in ks.load_catalogue("B200") if c["op"] == op)
    rows = [{"params": entry["default_params_per_dtype"][dt], "problem_size": 1, "dtype": dt,
             "triton_autotune_cfg": triton_cfg, "cutile_autotune_cfg": {"occupancy": 1}}
            for dt in entry["dtypes"]]
    path = paths.autotune_log_path(gpu, op, mode, backends)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return path


def test_catalogue_generation_is_per_gpu_and_reads_one_named_log(metadata):
    from tilebench.profiling import ncu_catalogue
    b200 = {p.name: digest(p) for p in (metadata / "B200").iterdir()}
    wanted = _autotune_log("GH200", "mul2", "autotune", ["cutile", "triton"], {"BLOCK": 1024})
    # decoys that a glob or an mtime guess could pick up instead; written later, so newer
    _autotune_log("GH200", "mul2", "default", ["triton", "cutile"], {"BLOCK": 1})
    _autotune_log("GH200", "mul2", "autotune", ["tilelang"], {"BLOCK": 2})
    _autotune_log("GH200", "mul2", "autotune", ["triton", "cutile", "tilelang"], {"BLOCK": 3})
    _autotune_log("B200", "mul2", "autotune", ["triton", "cutile"], {"BLOCK": 4})
    assert wanted.name == "mul2_autotune_triton-cutile.json"

    ncu_catalogue.main(["--gpu", "GH200", "mul2"])

    (entry,) = json.loads((metadata / "GH200" / "ncu_catalogue.json").read_text())
    winners = {w["triton"]["BLOCK"] for w in entry["autotune_winner_per_dtype"].values()}
    assert winners == {1024}
    assert {p.name: digest(p) for p in (metadata / "B200").iterdir()} == b200     # B200 untouched

    ncu_catalogue.main(["--gpu", "GH200", "--tile-language", "tilelang,cutile,triton", "mul2"])
    (entry,) = json.loads((metadata / "GH200" / "ncu_catalogue.json").read_text())
    assert {w["triton"]["BLOCK"] for w in entry["autotune_winner_per_dtype"].values()} == {3}

    gh200 = digest(metadata / "GH200" / "ncu_catalogue.json")
    ncu_catalogue.main(["--gpu", "B200", "mul2"])                                 # and the other way round
    assert digest(metadata / "GH200" / "ncu_catalogue.json") == gh200
    for bad in (["mul2"], ["--gpu", "GH200", "--tile-language", "tilelang", "mul2"]):
        with pytest.raises(SystemExit):                       # no GPU default; winners need triton and cutile
            ncu_catalogue.main(bad)


def test_kernel_count_probe_is_per_gpu(metadata, monkeypatch):
    from tilebench.profiling import probe_kernel_count as probe
    shutil.copy(metadata / "B200" / "ncu_catalogue.json", (metadata / "GH200").mkdir() or metadata / "GH200")
    b200 = digest(metadata / "B200" / "kernel_counts.json")
    monkeypatch.setattr(probe, "count_one", lambda *a, **kw: {"count": 7, "names": ["gh200_kernel"]})
    monkeypatch.setenv("ONLY_OP", "mul2")

    monkeypatch.setattr(sys, "argv", ["probe", "--gpu", "GH200"])
    probe.main()
    rows = json.loads((metadata / "GH200" / "kernel_counts.json").read_text())
    assert rows and {r["op"] for r in rows} == {"mul2"} and {r["count"] for r in rows} == {7}
    assert digest(metadata / "B200" / "kernel_counts.json") == b200               # B200 untouched

    gh200 = digest(metadata / "GH200" / "kernel_counts.json")
    monkeypatch.setattr(sys, "argv", ["probe", "--gpu", "B200"])
    probe.main()                                              # ONLY_OP merge into B200's own file
    assert digest(metadata / "GH200" / "kernel_counts.json") == gh200
    merged = json.loads((metadata / "B200" / "kernel_counts.json").read_text())
    assert {r["op"] for r in merged} == {c["op"] for c in ks.load_catalogue("B200")}

    monkeypatch.setattr(sys, "argv", ["probe", "--gpu", "MI300X"])
    with pytest.raises(SystemExit) as e:                      # no catalogue for it: not B200's
        probe.main()
    assert "MI300X" in str(e.value.code)
    monkeypatch.setattr(sys, "argv", ["probe"])
    with pytest.raises(SystemExit):
        probe.main()


# --------------------------------------------------------------------------
# packaging: the installed package is source only
# --------------------------------------------------------------------------

def test_a_regular_install_ships_source_without_experiment_data_or_cluster_scripts(tmp_path):
    """Not an editable install: build from a copy of the sources, install into a
    private directory and use the package from outside the repository."""
    src, site = tmp_path / "src", tmp_path / "site"
    shutil.copytree(REPO / "tilebench", src / "tilebench",
                    ignore=shutil.ignore_patterns("__pycache__", "llm_generated", "*.pyc"))
    for name in ("pyproject.toml", "README.md", "requirements.txt"):
        shutil.copy(REPO / name, src / name)
    build = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                            "--no-build-isolation", "--target", str(site), str(src)],
                           capture_output=True, text=True, cwd=tmp_path)
    if build.returncode != 0 and "No module named pip" in build.stderr:
        pytest.skip("pip is not available")
    assert build.returncode == 0, build.stderr[-2000:]

    installed = [p.relative_to(site).as_posix() for p in (site / "tilebench").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts]      # pip byte-compiles on install
    assert "tilebench/profiling/ncu_one.py" in installed and "tilebench/paths.py" in installed
    assert [f for f in installed if f.endswith((".sh", ".sbatch"))] == []
    assert [f for f in installed if f.startswith("tilebench/profiling/") and not f.endswith(".py")] == []
    # the resources the framework does need at run time are still there
    assert "tilebench/data/peak_performance/B200.json" in installed
    assert "tilebench/benchmarks/operators/mul2/config.yaml" in installed

    code = ("import json, tilebench, tilebench.paths as p\n"
            "print(json.dumps({'pkg': tilebench.__file__, 'meta': str(p.ncu_catalogue_path('B200')),"
            " 'ops': len(p.list_operators())}))\n")
    env = {k: v for k, v in os.environ.items() if k != "TILEBENCH_REPO_ROOT"}
    env["PYTHONPATH"] = str(site)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    info = json.loads(out.stdout.strip().splitlines()[-1])
    assert info["pkg"].startswith(str(site)) and info["ops"] == 45
    assert "/tilebench/" not in info["meta"].replace(str(site), "")     # generated data is not inside the package
