"""Hardware-scoped NCU profiling metadata and outputs:
    tilebench/profiling/metadata/<hardware>/{ncu_catalogue,kernel_counts}.json
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

#: The committed B200 files, recorded BEFORE they moved from tilebench/profiling/
#: to tilebench/profiling/metadata/B200/. Update only when they are regenerated.
B200_SHA256 = {
    "ncu_catalogue.json": "814bb30d1c951396739b992d5db92c4ed1ec14fd500a85dae46b3d2be256913d",
    "kernel_counts.json": "cfe0687e48665968e09e62ac3def2e86ae3bed14aa4a12a46175dcce79b666a6",
}


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    """A temporary metadata tree with a copy of the committed B200 files, and
    temporary results and NCU output roots."""
    root = tmp_path / "metadata"
    shutil.copytree(paths.PROFILING_METADATA_ROOT / "B200", root / "B200")
    monkeypatch.setattr(paths, "PROFILING_METADATA_ROOT", root)
    monkeypatch.setattr(paths, "NCU_OUTPUT_ROOT", tmp_path / "outputs" / "ncu")
    monkeypatch.setattr(paths, "RESULTS_ROOT", tmp_path / "results")
    return root


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def test_metadata_and_ncu_outputs_are_one_directory_per_hardware():
    base = paths.PACKAGE_ROOT / "profiling" / "metadata"
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


def test_there_is_no_global_metadata_any_more():
    assert not hasattr(paths, "NCU_CATALOGUE") and not hasattr(paths, "KERNEL_COUNTS")
    assert not (paths.PACKAGE_ROOT / "profiling" / "ncu_catalogue.json").exists()
    assert not (paths.PACKAGE_ROOT / "profiling" / "kernel_counts.json").exists()


def test_committed_b200_metadata_is_byte_identical_to_the_pre_move_files():
    for name, sha in B200_SHA256.items():
        assert digest(paths.profiling_metadata_dir("B200") / name) == sha, name


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
    assert not (metadata.parent / "outputs").exists()         # nothing was written for it


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
# packaging and portability
# --------------------------------------------------------------------------

def test_metadata_ships_with_a_regular_install(tmp_path):
    """Not an editable install: build from a copy of the sources, install into a
    private directory and read the metadata from outside the repository."""
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

    code = ("import json, tilebench, tilebench.paths as p\n"
            "from tilebench.profiling import ncu_kernel_select as ks\n"
            "counts, names = ks.load_kernel_counts('B200')\n"
            "print(json.dumps({'pkg': tilebench.__file__, 'catalogue': str(p.ncu_catalogue_path('B200')),"
            " 'ops': len(ks.load_catalogue('B200')), 'pairs': len(counts)}))\n")
    env = {k: v for k, v in os.environ.items() if k != "TILEBENCH_REPO_ROOT"}
    env["PYTHONPATH"] = str(site)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    info = json.loads(out.stdout.strip().splitlines()[-1])
    assert info["pkg"].startswith(str(site)) and info["catalogue"].startswith(str(site))
    assert info["ops"] == 45 and info["pairs"] > 0
    for name, sha in B200_SHA256.items():                     # the installed copies are the committed bytes
        assert digest(site / "tilebench/profiling/metadata/B200" / name) == sha


@pytest.mark.parametrize("script", ["run_batch.sh", "rerun_timeout.sh", "batch1_launch.sh",
                                    "supplemental_launch.sh", "batch2.sbatch"])
def test_profiling_shell_scripts_are_machine_independent(script):
    text = (REPO / "tilebench" / "profiling" / script).read_text()
    assert "/projects/" not in text and "kzhou6" not in text and "bcui2" not in text
    assert "PYTHONPATH" not in text and "_summary.csv" not in text and "rename_outputs" not in text


def test_run_batch_works_from_outside_the_repository(tmp_path):
    """A stub `python` records how run_batch.sh calls the benchmark."""
    record = tmp_path / "calls.txt"
    stub = tmp_path / "bin" / "python"
    stub.parent.mkdir()
    stub.write_text(f'#!/usr/bin/env bash\necho "$PWD|${{PYTHONPATH:-unset}}|$*" >> "{record}"\n')
    stub.chmod(0o755)
    batch = f"pytest_{os.getpid()}"
    env = {**os.environ, "PATH": f"{stub.parent}{os.pathsep}{os.environ['PATH']}", "GPU": "GH200"}
    env.pop("PYTHONPATH", None)
    try:
        r = subprocess.run(["bash", str(REPO / "tilebench/profiling/run_batch.sh"), batch, "mul2"],
                           cwd=tmp_path, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        calls = [line.split("|") for line in record.read_text().splitlines()]
        assert len(calls) == 2                                # default, then autotune
        for cwd, pythonpath, args in calls:
            assert cwd == str(tmp_path) and pythonpath == "unset"
            assert args.startswith(f"{REPO}/scripts/run_bench.py --gpu GH200 --operator mul2")
        assert calls[1][2].endswith("--autotune") and not calls[0][2].endswith("--autotune")
    finally:
        shutil.rmtree(REPO / "outputs" / batch, ignore_errors=True)
    missing = subprocess.run(["bash", str(REPO / "tilebench/profiling/run_batch.sh"), batch, "mul2"],
                             cwd=tmp_path, env={k: v for k, v in env.items() if k != "GPU"},
                             capture_output=True, text=True)
    assert missing.returncode != 0 and "GPU" in missing.stderr      # no default label
