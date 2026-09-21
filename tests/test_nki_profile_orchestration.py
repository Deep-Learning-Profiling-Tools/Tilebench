"""CPU-only tests for parent-side NKI orchestration (fake workers, no Neuron)."""
import json
import os
import sys
import types

import pytest
import torch

from tilebench.core.nki_orchestrator import NkiOrchestrationError, profile_case_on_neuron

WINNER_TRACE = [{"tuner_name": "tilebench.benchmarks.operators.vector_add.impl_nki.add_kernel",
                 "shape_key": [[128, 8192], "torch.float16"],
                 "config": {"free_tile_size": 2048}}]


def fake_impl_nki():
    m = types.ModuleType("fake_impl_nki")

    def run(x, y, block_size=1024, autotune=False, **kwargs):
        return x + y

    m.run = run
    return m


def artifact_record(workdir, stem, marker):
    """Create a REAL fake pair on disk (the parent re-validates its SHA256s)."""
    from tilebench.core.nki_artifact import NKI_HLO_MARKER
    from tilebench.core.nki_profile_spec import sha256_file
    os.makedirs(workdir, exist_ok=True)
    neff = os.path.join(workdir, f"{stem}.neff")
    hlo = os.path.join(workdir, f"{stem}.hlo_module.pb")
    with open(neff, "wb") as f:
        f.write(b"fake-neff-" + stem.encode())
    with open(hlo, "wb") as f:
        f.write(b"hlo " + (NKI_HLO_MARKER if marker else b"plain") + stem.encode())
    return {"stem": stem, "neff_path": neff, "hlo_path": hlo,
            "neff_sha256": sha256_file(neff), "hlo_sha256": sha256_file(hlo),
            "hlo_nki_marker": marker}


def fake_profiler(neff_path, *, warmup, out_dir, tag):
    """Capture/view stand-in: only the explicit $NKI_NEFF_PATH override uses it."""
    mean = 0.09 if tag == "nki" else 0.03
    return {"mean": mean, "total_ms": mean, "repeat": 1,
            "method": "neuron_explorer_capture", "neff": neff_path,
            "profile_nth_exec": max(2, warmup + 1),
            "profile_input_mode": "tool_default"}


# Fake runtime trace: the fake worker writes, per timed iteration, the
# executions it "ran" (model_id, start, end) next to the runtime NEFF copies.
def fake_executions_loader(session_dir, *, data_path, display_name):
    with open(os.path.join(session_dir, "fake_executions.json")) as f:
        execs = json.load(f)
    return execs, os.path.join(data_path, "profiles", "global", display_name + "@latest")


class FakeRunner:
    """Stands in for the subprocess launches; records calls, writes results."""

    def __init__(self, selector_verify_ok=True, profile_error=None):
        self.calls = []
        self.selector_verify_ok = selector_verify_ok
        self.profile_error = profile_error
        self.torch_verify_ok = True
        self.torch_artifact = True
        self.nki_verify_ok = True
        # NKI run(): device µs per executed graph, per iteration
        self.nki_graphs = {"MODULE_N": 90.0}
        # simulate dropped trace events / varying graph structure on iteration 2
        self.drop_second_iteration_graphs = False
        # None -> the NKI phase raised before identity was established (artifacts: null);
        # [] -> it compiled nothing (artifacts: [])
        self.nki_phase_artifacts_override = "unset"
        # executions torch-xla submitted that the trace does not hold (or vice versa)
        self.extra_xla_executions = 0
        self.clock = 1_000_000

    def __call__(self, cmd, *, cwd, env_overrides):
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1)
                if isinstance(cmd[i], str) and cmd[i].startswith("--")}
        mode = args["--mode"]
        self.calls.append({"mode": mode, "args": args, "env": env_overrides})
        if mode == "selector":
            result = {"ok": self.selector_verify_ok, "mode": "selector",
                      "verify_ok": self.selector_verify_ok,
                      "verify_error": None if self.selector_verify_ok else "mismatch",
                      "tuning_trace": WINNER_TRACE,
                      "last_config": {"free_tile_size": 2048}}
        elif self.profile_error:
            result = {"ok": False, "mode": "profile",
                      "error": self.profile_error, "traceback": "..."}
        else:
            wd = args["--workdir"]
            with open(args["--spec"]) as f:
                spec = json.load(f)
            session = os.path.join(args["--inspect-dir"], "i-fake_pid_1", "123")
            os.makedirs(session, exist_ok=True)
            execs = []
            self.xla_executions = 0

            def phase(graphs_us, repeat, drop_after_first=False):
                """Timed windows (XLA execution-index ranges) + trace executions."""
                windows = []
                for it in range(repeat):
                    begin = len(execs)
                    for stem, us in graphs_us.items():
                        if drop_after_first and it > 0 and stem != next(iter(graphs_us)):
                            continue
                        execs.append({"flow_id": [self.clock], "model_id": "M_" + stem,
                                      "start_ns": self.clock,
                                      "end_ns": self.clock + int(us * 1000), "pcores": 2})
                        self.clock += int(us * 1000) + 1000
                    windows.append([begin, max(len(execs), begin + 1)])
                self.xla_executions = len(execs)
                return windows

            def artifacts(stems_marker):
                out = []
                for stem, marker in stems_marker:
                    rec = artifact_record(wd, stem, marker)
                    # the runtime writes back the NEFF it executed, byte-identical
                    import shutil
                    shutil.copyfile(rec["neff_path"],
                                    os.path.join(session, f"neff_M_{stem}_vnc_0.neff"))
                    out.append(rec)
                return out

            torch_entry = {"verify_ok": self.torch_verify_ok,
                           "verify_error": None if self.torch_verify_ok else "xla mismatch",
                           "windows": None, "artifacts": None}
            if self.torch_artifact:
                torch_entry["artifacts"] = artifacts([("MODULE_T", False)])
            else:
                torch_entry["artifacts"] = []          # compiled nothing (e.g. unsupported op)
            if self.torch_verify_ok and self.torch_artifact:
                torch_entry["windows"] = phase({"MODULE_T": 30.0}, spec["repeat"])
            nki_entry = None
            if spec["nki_enabled"]:
                nki_entry = {"verify_ok": self.nki_verify_ok,
                             "verify_error": None if self.nki_verify_ok else "mismatch on new inputs",
                             "replay_installed": bool(spec["autotune_replay"]),
                             "replay_consumed": bool(spec["autotune_replay"]),
                             "executed_trace": spec["autotune_replay"],
                             "windows": None, "artifacts": None}
                nki_entry["artifacts"] = artifacts(
                    [(stem, stem.startswith("MODULE_N")) for stem in self.nki_graphs])
                if self.nki_phase_artifacts_override != "unset":
                    nki_entry["artifacts"] = self.nki_phase_artifacts_override
                    nki_entry["verify_ok"] = False
                    nki_entry["verify_error"] = "phase failed: RuntimeError: boom"
                elif self.nki_verify_ok:
                    nki_entry["windows"] = phase(self.nki_graphs, spec["repeat"],
                                                 self.drop_second_iteration_graphs)
            with open(os.path.join(session, "ntrace.pb"), "wb") as f:
                f.write(b"fake")
            with open(os.path.join(session, "fake_executions.json"), "w") as f:
                json.dump(execs, f)
            result = {"ok": True, "mode": "profile", "torch": torch_entry, "nki": nki_entry,
                      "xla_executions": self.xla_executions + self.extra_xla_executions}
            if spec["nki_enabled"] and not nki_entry["verify_ok"]:
                result["ok"] = False
        with open(args["--result"], "w") as f:
            json.dump(result, f)
        return 0 if result["ok"] else 1


def orchestrate(tmp_path, runner, autotune=True):
    inputs = (torch.zeros(8, dtype=torch.float16), torch.zeros(8, dtype=torch.float16))
    return profile_case_on_neuron(
        operator="vector_add", params={"n": 8}, dtype_str="fp16",
        inputs=inputs, ref_output=torch.zeros(8, dtype=torch.float16),
        impl_nki=fake_impl_nki(), block_size=1024, autotune=autotune,
        verify_atol=None, verify_rtol=None, warmup=2, repeat=3,
        base_dir=str(tmp_path / "profiles"),
        index_path=str(tmp_path / "index.jsonl"),
        runner=runner, profiler=fake_profiler,
        executions_loader=fake_executions_loader)


def test_selector_runs_and_exits_before_profile(tmp_path):
    runner = FakeRunner()
    orchestrate(tmp_path, runner)
    assert [c["mode"] for c in runner.calls] == ["selector", "profile"]


def test_profile_receives_exact_winner_trace(tmp_path):
    runner = FakeRunner()
    orchestrate(tmp_path, runner)
    profile_call = runner.calls[1]
    with open(profile_call["args"]["--spec"]) as f:
        spec = json.load(f)
    assert spec["autotune_replay"] == WINNER_TRACE
    assert spec["autotune_enabled"] is True


def test_profile_uses_distinct_private_cwd_and_cache(tmp_path):
    runner = FakeRunner()
    res = orchestrate(tmp_path, runner)
    sel, prof = runner.calls
    assert sel["args"]["--workdir"] != prof["args"]["--workdir"]
    assert res["spec_id"][:16] in prof["args"]["--workdir"]
    assert prof["args"]["--workdir"].endswith("artifacts")
    assert prof["args"]["--cache-dir"].endswith("cache")
    assert res["spec_id"][:16] in prof["args"]["--cache-dir"]
    assert prof["args"]["--inspect-dir"].endswith("inspect")
    assert "--inspect-dir" not in sel["args"]          # the selector is never timed


def test_worker_failure_propagates(tmp_path):
    runner = FakeRunner(profile_error="NkiArtifactIdentityError: ambiguous")
    with pytest.raises(NkiOrchestrationError, match="ambiguous"):
        orchestrate(tmp_path, runner)


def test_failed_selector_prevents_profile_execution(tmp_path):
    runner = FakeRunner(selector_verify_ok=False)
    with pytest.raises(NkiOrchestrationError, match="selector verification failed"):
        orchestrate(tmp_path, runner)
    assert [c["mode"] for c in runner.calls] == ["selector"]


def test_failed_replay_prevents_artifact_profiling(tmp_path):
    runner = FakeRunner(profile_error="NkiAutotuneReplayError: no longer exists")
    with pytest.raises(NkiOrchestrationError, match="NkiAutotuneReplayError"):
        orchestrate(tmp_path, runner)


def test_parent_orchestration_does_not_import_torch_xla(tmp_path):
    orchestrate(tmp_path, FakeRunner())
    assert "torch_xla" not in sys.modules


def test_manifest_and_index_written_with_identity(tmp_path):
    res = orchestrate(tmp_path, FakeRunner())
    with open(res["manifest_path"]) as f:
        manifest = json.load(f)
    assert manifest["spec_id"] == res["spec_id"]
    assert manifest["identity_source"] == "private_debug_dump"
    assert manifest["identity_status"] == "validated"
    assert manifest["autotune_replay"] == WINNER_TRACE
    nki_t, torch_t = manifest["targets"]["nki"], manifest["targets"]["torch"]
    assert [a["hlo_nki_marker"] for a in nki_t["artifacts"]] == [True]
    assert [a["hlo_nki_marker"] for a in torch_t["artifacts"]] == [False]
    # every executed NEFF is attributed to a validated artifact
    assert [e["stem"] for e in nki_t["executed"]] == ["MODULE_N"]
    assert nki_t["executed"][0]["neff_sha256"] == nki_t["artifacts"][0]["neff_sha256"]
    assert nki_t["executed"][0]["count_per_iteration"] == 1
    assert manifest["profile_input_mode"] == "runtime_inspect_real_inputs"
    assert manifest["compiler_environment"]["NEURON_RT_INSPECT_DEVICE_PROFILE"] == "1"
    lines = open(tmp_path / "index.jsonl").read().splitlines()
    last = json.loads(lines[-1])
    assert last["spec_id"] == res["spec_id"]
    assert last["nki_neff_sha256s"] == [nki_t["artifacts"][0]["neff_sha256"]]
    # engine-facing fields: mean over the 3 timed iterations (real inputs)
    assert res["nki_ok"] and res["nki_ms"] == pytest.approx(0.09)
    assert res["torch_ms"] == pytest.approx(0.03)
    assert res["nki_stats"]["repeat"] == 3 and res["nki_stats"]["method"] == "neuron_rt_inspect"
    assert res["nki_stats"]["spec_id"] == res["spec_id"]
    assert res["nki_stats"]["neff_sha256s"] == [nki_t["artifacts"][0]["neff_sha256"]]
    assert nki_t["stats"]["mean"] == pytest.approx(0.09)


def test_multi_graph_run_is_timed_as_the_sum_of_its_graphs(tmp_path):
    """radix_sort shape: several kernel graphs + an XLA helper graph per run()."""
    runner = FakeRunner()
    runner.nki_graphs = {"MODULE_N_PASS_A": 20.0, "MODULE_N_PASS_B": 20.0, "MODULE_XLA": 0.5}
    res = orchestrate(tmp_path, runner, autotune=False)
    assert res["nki_ok"] and res["nki_ms"] == pytest.approx(0.0405)
    assert res["nki_stats"]["executions_per_iteration"] == 3
    with open(res["manifest_path"]) as f:
        nki_t = json.load(f)["targets"]["nki"]
    assert sorted(e["stem"] for e in nki_t["executed"]) == \
        ["MODULE_N_PASS_A", "MODULE_N_PASS_B", "MODULE_XLA"]
    assert {e["stem"]: e["hlo_nki_marker"] for e in nki_t["executed"]}["MODULE_XLA"] is False


def test_nki_run_that_executes_no_kernel_graph_fails(tmp_path):
    runner = FakeRunner()
    runner.nki_graphs = {"MODULE_XLA": 5.0}       # worker: phase compiled a marker pair...
    # ...but the artifact list still needs one marker pair for the parent to accept it,
    # so mimic a kernel that was compiled yet never executed in the timed windows.
    real_artifacts = runner.__class__.__call__

    def call(cmd, *, cwd, env_overrides):
        rc = real_artifacts(runner, cmd, cwd=cwd, env_overrides=env_overrides)
        args = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
        if args["--mode"] == "profile":
            with open(args["--result"]) as f:
                result = json.load(f)
            result["nki"]["artifacts"].append(artifact_record(args["--workdir"], "MODULE_N", True))
            with open(args["--result"], "w") as f:
                json.dump(result, f)
        return rc

    with pytest.raises(Exception, match="executed no marker-bearing"):
        orchestrate(tmp_path, call, autotune=False)


def test_execution_count_mismatch_is_not_published(tmp_path):
    """The trace must hold exactly the executions torch-xla submitted, or the
    index windows cannot be attributed (dropped trace events, foreign executions)."""
    runner = FakeRunner()
    runner.extra_xla_executions = 1
    with pytest.raises(Exception, match="cannot be attributed"):
        orchestrate(tmp_path, runner, autotune=False)


def test_varying_execution_pattern_is_not_published(tmp_path):
    runner = FakeRunner()
    runner.nki_graphs = {"MODULE_N": 20.0, "MODULE_XLA": 0.5}
    runner.drop_second_iteration_graphs = True
    with pytest.raises(Exception, match="pattern differs"):
        orchestrate(tmp_path, runner, autotune=False)


def test_no_autotune_means_untuned_profile(tmp_path):
    runner = FakeRunner()
    res = orchestrate(tmp_path, runner, autotune=False)
    assert [c["mode"] for c in runner.calls] == ["profile"]
    with open(runner.calls[0]["args"]["--spec"]) as f:
        spec = json.load(f)
    assert spec["autotune_replay"] == [] and spec["autotune_enabled"] is False
    assert res["nki_ok"]


def test_identical_spec_reuses_validated_artifact_but_reverifies(tmp_path):
    runner = FakeRunner()
    first = orchestrate(tmp_path, runner, autotune=False)
    assert [c["mode"] for c in runner.calls] == ["profile"]
    second = orchestrate(tmp_path, runner, autotune=False)
    # the worker runs again (inputs change per run -> correctness re-verified)
    assert [c["mode"] for c in runner.calls] == ["profile", "profile"]
    assert second["spec_id"] == first["spec_id"]
    assert second["identity_source"] == "validated_manifest_reuse"
    assert second["nki_stats"]["neff_sha256s"] == first["nki_stats"]["neff_sha256s"]
    # private artifacts/cache were kept for the reuse run (only wiped on rebuild)
    with open(first["manifest_path"]) as f:
        first_neff = json.load(f)["targets"]["nki"]["artifacts"][0]["neff_path"]
    assert os.path.isfile(first_neff)
    with open(runner.calls[1]["args"]["--spec"]) as f:
        assert json.load(f)["expected_stems"] == {"torch": ["MODULE_T"], "nki": ["MODULE_N"]}


def test_reuse_with_failing_reverification_is_not_published(tmp_path):
    runner = FakeRunner()
    orchestrate(tmp_path, runner, autotune=False)
    runner.nki_verify_ok = False           # a data-dependent bug on the new inputs
    second = orchestrate(tmp_path, runner, autotune=False)
    assert second["identity_source"] == "validated_manifest_reuse"
    assert second["nki_ok"] is False and "verification failed" in second["nki_err"]
    assert second["nki_ms"] != second["nki_ms"]  # nan
    with open(second["manifest_path"]) as f:
        nki_t = json.load(f)["targets"]["nki"]
    assert nki_t["stats"] is None and nki_t["executed"] is None
    assert [a["stem"] for a in nki_t["artifacts"]] == ["MODULE_N"]   # identity still recorded
    # ...so the spec recovers on the next run instead of being stuck forever (Codex P1)
    runner.nki_verify_ok = True
    third = orchestrate(tmp_path, runner, autotune=False)
    assert third["identity_source"] == "validated_manifest_reuse"
    assert third["nki_ok"] and third["nki_ms"] == pytest.approx(0.09)
    with open(runner.calls[2]["args"]["--spec"]) as f:
        assert json.load(f)["expected_stems"]["nki"] == ["MODULE_N"]


def test_nki_phase_failure_without_identity_triggers_rebuild(tmp_path):
    runner = FakeRunner()
    orchestrate(tmp_path, runner, autotune=False)
    runner.nki_phase_artifacts_override = None       # phase raised before identity
    second = orchestrate(tmp_path, runner, autotune=False)
    assert second["nki_ok"] is False
    runner.nki_phase_artifacts_override = "unset"
    third = orchestrate(tmp_path, runner, autotune=False)
    assert third["identity_source"] == "private_debug_dump"          # rebuilt, not reused
    assert third["nki_ok"] and third["nki_ms"] == pytest.approx(0.09)


def test_nki_phase_that_compiled_nothing_triggers_rebuild(tmp_path):
    runner = FakeRunner()
    orchestrate(tmp_path, runner, autotune=False)
    runner.nki_phase_artifacts_override = []         # e.g. kernel failed to compile
    second = orchestrate(tmp_path, runner, autotune=False)
    assert second["nki_ok"] is False
    runner.nki_phase_artifacts_override = "unset"
    third = orchestrate(tmp_path, runner, autotune=False)
    assert third["identity_source"] == "private_debug_dump"
    assert third["nki_ok"]


def test_tampered_reused_artifact_triggers_rebuild_not_reuse(tmp_path):
    runner = FakeRunner()
    first = orchestrate(tmp_path, runner, autotune=False)
    with open(first["manifest_path"]) as f:
        first_neff = json.load(f)["targets"]["nki"]["artifacts"][0]["neff_path"]
    with open(first_neff, "ab") as f:
        f.write(b"tampered")
    second = orchestrate(tmp_path, runner, autotune=False)
    assert [c["mode"] for c in runner.calls] == ["profile", "profile"]  # rebuilt
    assert second["identity_source"] == "private_debug_dump"


def test_torch_verification_failure_is_not_published_as_baseline(tmp_path):
    runner = FakeRunner()
    runner.torch_verify_ok = False
    calls = []

    def spy_profiler(neff_path, *, warmup, out_dir, tag):
        calls.append(tag)
        return fake_profiler(neff_path, warmup=warmup, out_dir=out_dir, tag=tag)

    inputs = (torch.zeros(8, dtype=torch.float16), torch.zeros(8, dtype=torch.float16))
    res = profile_case_on_neuron(
        operator="vector_add", params={"n": 8}, dtype_str="fp16",
        inputs=inputs, ref_output=torch.zeros(8, dtype=torch.float16),
        impl_nki=fake_impl_nki(), block_size=1024, autotune=False,
        verify_atol=None, verify_rtol=None, warmup=2, repeat=3,
        base_dir=str(tmp_path / "profiles"), index_path=str(tmp_path / "index.jsonl"),
        runner=runner, profiler=spy_profiler, executions_loader=fake_executions_loader)
    assert res["torch_stats"] is None and res["torch_ms"] != res["torch_ms"]  # nan
    assert "verification failed" in res["torch_err"]
    assert calls == []                   # capture path unused; the trace timed only NKI
    with open(res["manifest_path"]) as f:
        manifest = json.load(f)
    assert manifest["targets"]["torch"]["stats"] is None
    assert res["nki_ok"] and manifest["targets"]["nki"]["stats"]["mean"] == pytest.approx(0.09)


def test_explicit_override_is_reported_unverified(tmp_path, monkeypatch):
    from tilebench.core.nki_artifact import NEFF_PATH_ENV
    override_dir = tmp_path / "override"
    art = artifact_record(str(override_dir), "MODULE_OVERRIDE", True)
    monkeypatch.setenv(NEFF_PATH_ENV, art["neff_path"])
    runner = FakeRunner()
    res = orchestrate(tmp_path, runner, autotune=False)
    with open(runner.calls[0]["args"]["--spec"]) as f:
        assert json.load(f)["nki_enabled"] is False   # worker timed torch only
    assert res["identity_source"] == "explicit_override"
    assert res["nki_ok"] is False and "NOT verified" in res["nki_err"]
    assert res["nki_ms"] != res["nki_ms"]              # nan: never a benchmark number
    assert res["nki_stats"]["verified"] is False and res["nki_stats"]["mean"] == pytest.approx(0.09)
    assert res["nki_stats"]["method"] == "neuron_explorer_capture"   # tool-default inputs
    with open(res["manifest_path"]) as f:
        nki_t = json.load(f)["targets"]["nki"]
    assert nki_t["verify_ok"] is None and nki_t["executed"] is None


def test_torch_phase_failure_keeps_nki_result(tmp_path):
    """A torch-baseline failure (e.g. torch.sort unsupported on trn2) is reported
    as torch_err while the NKI target is still profiled — and the validated NKI
    artifacts are still reused on the next run (the baseline is re-run fresh)."""
    runner = FakeRunner()
    runner.torch_verify_ok = False
    runner.torch_artifact = False
    res = orchestrate(tmp_path, runner, autotune=False)
    assert res["torch_ms"] != res["torch_ms"] and res["torch_err"]
    assert res["nki_ok"] and res["nki_ms"] == pytest.approx(0.09)
    second = orchestrate(tmp_path, runner, autotune=False)
    assert second["identity_source"] == "validated_manifest_reuse"
    with open(runner.calls[1]["args"]["--spec"]) as f:
        assert json.load(f)["expected_stems"] == {"nki": ["MODULE_N"]}
