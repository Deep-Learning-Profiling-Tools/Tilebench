"""scripts/run_bench_all.py --tile-language: the same selection rules as
run_bench.py, passed to the engine and recorded in the run manifest.

The engine is replaced by a stub; nothing here needs a GPU."""
import json
import sys

import pytest

from tilebench.backends import parse_backends
from tilebench.core.engine import DEFAULT_ENABLED_BACKENDS

from test_results_layout import fake_results, load_script, results  # noqa: F401


@pytest.fixture
def run_all(results, tmp_path, monkeypatch):
    mod = load_script("run_bench_all")
    calls = []

    def engine(op, enabled_backends="<not passed>", logs_dir=None):
        calls.append({"op": op, "enabled_backends": enabled_backends, "logs_dir": logs_dir})
        if enabled_backends == "<not passed>":
            enabled_backends = None
        return fake_results(enabled_backends or set(DEFAULT_ENABLED_BACKENDS))

    monkeypatch.setattr(mod, "run_benchmark_suite", engine)
    runs = tmp_path / "runs"

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["run_bench_all.py", "--gpu", "MI300X",
                                          "--results-root", str(runs), "--operators", "mul2", "relu",
                                          *argv])
        code = mod.main()
        summaries = sorted(runs.glob("*/summary.json"))
        return code, json.loads(summaries[-1].read_text()) if summaries else None
    run.calls = calls
    run.runs = runs
    return run


def test_triton_only_reaches_the_engine_for_every_operator(run_all):
    code, manifest = run_all("--tile-language", "triton")
    assert code == 0
    assert [c["enabled_backends"] for c in run_all.calls] == [{"triton"}, {"triton"}]
    assert manifest["provenance"]["run"] == {"script": "scripts/run_bench_all.py",
                                             "tile_language": "triton", "backends": ["triton"]}


@pytest.mark.parametrize("flag", ["triton", "cutile,triton", "Triton, torch", "all", "tilelang", "nki",
                                  "triton,cutile,tilelang,nki"])
def test_selection_is_parsed_exactly_like_run_bench(run_all, flag):
    code, manifest = run_all("--tile-language", flag)
    assert code == 0
    expected = parse_backends(flag)                        # the run_bench.py parser itself
    assert all(c["enabled_backends"] == set(expected) for c in run_all.calls)
    assert manifest["provenance"]["run"]["backends"] == expected
    assert manifest["provenance"]["run"]["tile_language"] == flag


def test_without_the_flag_the_engine_default_is_kept(run_all):
    code, manifest = run_all()
    assert code == 0
    # the engine is called exactly as before: no enabled_backends argument at all
    assert [c["enabled_backends"] for c in run_all.calls] == ["<not passed>", "<not passed>"]
    assert manifest["provenance"]["run"]["tile_language"] is None
    assert manifest["provenance"]["run"]["backends"] == ["triton", "cutile", "tilelang", "nki"]


def test_the_engine_default_is_unchanged():
    assert DEFAULT_ENABLED_BACKENDS == ("triton", "cutile", "tilelang", "nki")


def test_logs_dir_is_still_the_namespace(run_all, results):
    run_all("--tile-language", "triton")
    assert all(c["logs_dir"] == results / "MI300X" / "logs" for c in run_all.calls)


@pytest.mark.parametrize("flag", ["triton,cuda", "hip"])
def test_an_invalid_selection_is_rejected_before_anything_runs(run_all, flag, capsys):
    with pytest.raises(SystemExit) as e:
        run_all("--tile-language", flag)
    assert e.value.code == 2 and "--tile-language" in capsys.readouterr().err
    assert run_all.calls == [] and not run_all.runs.exists()


def test_an_empty_selection_is_torch_only_as_in_run_bench(run_all):
    code, manifest = run_all("--tile-language", "torch")
    assert code == 0 and parse_backends("torch") == []
    assert all(c["enabled_backends"] == set() for c in run_all.calls)
    assert manifest["provenance"]["run"]["backends"] == []
