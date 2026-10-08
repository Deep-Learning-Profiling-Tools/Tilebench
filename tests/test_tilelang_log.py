"""tilebench.core.tilelang_log: only the autotuner output of the current run is
collected, never a stale autotuner.log.

TileLang is replaced by a stand-in module that behaves like TileLang 0.1.11's
tilelang.autotuner.tuner: its first AutoTuner.run() in a process opens
autotuner.log in the working directory with FileHandler(mode="w") and keeps it
open; every later tuning in the process appends to it.
"""
import json
import logging
import sys
import types
import uuid

import pytest

from tilebench.core import tilelang_log

from test_results_layout import fake_results, load_script, results, run_bench  # noqa: F401


class FakeTuner:
    def __init__(self, monkeypatch):
        self.module = types.ModuleType("tilelang.autotuner.tuner")
        self.module.logger = logging.getLogger(f"fake-tilelang-{uuid.uuid4().hex}")
        self.module.logger.setLevel(logging.DEBUG)
        self.module.logger.propagate = False
        self.module._logger_handlers_initialized = False
        monkeypatch.setitem(sys.modules, "tilelang.autotuner.tuner", self.module)

    def run(self, *lines):
        """One AutoTuner.run(): initialise the handlers once, then log."""
        if not self.module._logger_handlers_initialized:
            self.module.logger.addHandler(logging.FileHandler("autotuner.log", mode="w"))
            self.module._logger_handlers_initialized = True
        for line in lines:
            self.module.logger.warning(line)

    def close(self):
        for h in list(self.module.logger.handlers):
            h.close()
            self.module.logger.removeHandler(h)


@pytest.fixture
def tuner(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = FakeTuner(monkeypatch)
    yield t
    t.close()


def test_a_process_that_never_tuned_collects_nothing_and_leaves_no_stale_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "tilelang.autotuner.tuner", raising=False)
    (tmp_path / "autotuner.log").write_text("operator A, an earlier process\n")
    dest = tmp_path / "logs/B_default_tilelang.log"
    dest.parent.mkdir()
    dest.write_text("left by an earlier run\n")
    start = tilelang_log.mark()
    path, note = tilelang_log.collect(start, dest)
    assert path is None and "did not run" in note
    assert not dest.exists()                                      # the slot holds only this run


def test_imported_but_never_initialised_is_not_ours(tuner, tmp_path):
    (tmp_path / "autotuner.log").write_text("stale\n")
    path, note = tilelang_log.collect(tilelang_log.mark(), tmp_path / "out.log")
    assert path is None and "did not run" in note


def test_the_first_use_truncates_a_stale_file_and_only_this_run_is_collected(tuner, tmp_path):
    (tmp_path / "autotuner.log").write_text("STALE operator A output\n")
    start = tilelang_log.mark()                                   # before TileLang opened the log
    assert start == 0
    tuner.run("B tuning 1", "B tuning 2")
    path, note = tilelang_log.collect(start, tmp_path / "B.log")
    assert note is None and path == tmp_path / "B.log"
    text = path.read_text()
    assert "B tuning 1" in text and "B tuning 2" in text and "STALE" not in text


def test_operators_in_one_process_each_get_their_own_part(tuner, tmp_path):
    a = tilelang_log.mark()
    tuner.run("A tuning")
    path_a, _ = tilelang_log.collect(a, tmp_path / "A.log")
    b = tilelang_log.mark()
    tuner.run("B tuning")
    path_b, _ = tilelang_log.collect(b, tmp_path / "B.log")
    assert path_a.read_text().strip() == "A tuning"
    assert path_b.read_text().strip() == "B tuning"               # nothing of A
    assert (tmp_path / "autotuner.log").read_text().splitlines() == ["A tuning", "B tuning"]


def test_a_run_that_wrote_nothing_collects_nothing_even_with_an_older_log(tuner, tmp_path):
    tuner.run("A tuning")
    dest = tmp_path / "B.log"
    dest.write_text("previous run of B\n")
    start = tilelang_log.mark()                                   # B hits the autotune cache
    path, note = tilelang_log.collect(start, dest)
    assert path is None and "cache hit" in note and not dest.exists()


def test_a_foreign_writer_is_detected(tuner, tmp_path):
    start = tilelang_log.mark()
    tuner.run("ours")
    with open(tmp_path / "autotuner.log", "a") as other:          # e.g. another process in this directory
        other.write("theirs\n")
    path, note = tilelang_log.collect(start, tmp_path / "out.log")
    assert path is None and "another process" in note


def test_a_replaced_file_is_detected(tuner, tmp_path):
    start = tilelang_log.mark()
    tuner.run("ours")
    (tmp_path / "autotuner.log").unlink()
    (tmp_path / "autotuner.log").write_text("ours\n")             # same bytes, different file
    path, note = tilelang_log.collect(start, tmp_path / "out.log")
    assert path is None and "replaced" in note


def test_collection_is_atomic_and_leaves_no_temporary(tuner, tmp_path):
    start = tilelang_log.mark()
    tuner.run("x")
    path, _ = tilelang_log.collect(start, tmp_path / "logs/deep/out.log")
    assert path.read_text().strip() == "x"
    assert [p.name for p in path.parent.iterdir()] == ["out.log"]


def test_collection_does_not_disturb_the_tuner(tuner, tmp_path):
    start = tilelang_log.mark()
    tuner.run("first")
    tilelang_log.collect(start, tmp_path / "out.log")
    tuner.run("second")                                           # the handler keeps working
    assert (tmp_path / "autotuner.log").read_text().splitlines() == ["first", "second"]


# --------------------------------------------------------------------------
# the runners
# --------------------------------------------------------------------------

def test_run_bench_collects_into_the_hardware_namespace(run_bench, results, tuner, tmp_path, monkeypatch):
    (tmp_path / "autotuner.log").write_text("STALE\n")
    def engine(operator, benchmark_overrides=None, enabled_backends=None, logs_dir=None):
        tuner.run(f"{operator} tuning")
        return fake_results(enabled_backends)
    monkeypatch.setattr(run_bench.module, "run_benchmark_suite", engine)
    run_bench("--gpu", "GH200", "--operator", "mul2", "--tile-language", "tilelang", "--autotune")
    log = results / "GH200/logs/tilelang_autotuner/mul2_autotune_tilelang.log"
    assert log.read_text().strip() == "mul2 tuning"
    record = json.loads((results / "GH200/logs/provenance/mul2_autotune_tilelang.json").read_text())
    assert record["run"]["tilelang_autotuner_log"] == str(log)
    assert record["run"]["tilelang_autotuner_log_note"] is None


def test_run_bench_without_tilelang_collects_nothing(run_bench, results, tuner):
    tuner.run("from an earlier operator in this process")
    run_bench("--gpu", "GH200", "--operator", "mul2", "--tile-language", "triton")
    assert not (results / "GH200/logs/tilelang_autotuner").exists()
    record = json.loads((results / "GH200/logs/provenance/mul2_default_triton.json").read_text())
    assert record["run"]["tilelang_autotuner_log"] is None


def test_run_bench_all_collects_one_file_per_operator(results, tuner, tmp_path, monkeypatch):
    mod = load_script("run_bench_all")
    tuner.run("before the suite")        # open the log here: main() chdirs to the repository

    def engine(op, logs_dir=None, enabled_backends=None):
        if op == "relu":
            tuner.run("relu tuning")                              # mul2 hits the cache
        return fake_results({"tilelang"})
    monkeypatch.setattr(mod, "run_benchmark_suite", engine)
    monkeypatch.setattr(sys, "argv", ["run_bench_all.py", "--gpu", "GH200", "--results-root",
                                      str(tmp_path / "runs"), "--operators", "mul2", "relu",
                                      "--tile-language", "tilelang"])
    assert mod.main() == 0
    (summary,) = (tmp_path / "runs").glob("*/summary.json")
    run_id = summary.parent.name
    artifacts = json.loads(summary.read_text())["artifacts"]["tilelang_autotuner"]
    relu_log = results / f"GH200/logs/tilelang_autotuner/relu_{run_id}.log"
    assert artifacts["relu"] == {"path": str(relu_log), "note": None}
    assert relu_log.read_text().strip() == "relu tuning"
    assert artifacts["mul2"]["path"] is None
    assert not (results / f"GH200/logs/tilelang_autotuner/mul2_{run_id}.log").exists()
