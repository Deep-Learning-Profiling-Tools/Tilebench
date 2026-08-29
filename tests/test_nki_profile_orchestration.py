"""CPU-only tests for parent-side NKI orchestration (fake workers, no Neuron)."""
import json
import os
import sys
import types

import pytest
import torch

from core.nki_orchestrator import NkiOrchestrationError, profile_case_on_neuron

WINNER_TRACE = [{"tuner_name": "benchmarks.operators.vector_add.impl_nki.add_kernel",
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
    from core.nki_artifact import NKI_HLO_MARKER
    from core.nki_profile_spec import sha256_file
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
    mean = 0.09 if tag == "nki" else 0.03
    return {"mean": mean, "total_ms": mean, "repeat": 1,
            "method": "neuron_profile", "neff": neff_path,
            "profile_nth_exec": max(2, warmup + 1),
            "profile_input_mode": "tool_default"}


class FakeRunner:
    """Stands in for the subprocess launches; records calls, writes results."""

    def __init__(self, selector_verify_ok=True, profile_error=None):
        self.calls = []
        self.selector_verify_ok = selector_verify_ok
        self.profile_error = profile_error
        self.torch_verify_ok = True

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
            result = {"ok": True, "mode": "profile",
                      "torch": {"verify_ok": self.torch_verify_ok,
                                "verify_error": None if self.torch_verify_ok else "xla mismatch",
                                "artifact": artifact_record(wd, "MODULE_T", False)},
                      "nki": ({"verify_ok": True, "verify_error": None,
                               "replay_installed": bool(spec["autotune_replay"]),
                               "replay_consumed": bool(spec["autotune_replay"]),
                               "executed_trace": spec["autotune_replay"],
                               "artifact": artifact_record(wd, "MODULE_N", True)}
                              if spec["nki_enabled"] else None)}
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
        runner=runner, profiler=fake_profiler)


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
    assert manifest["targets"]["nki"]["hlo_nki_marker"] is True
    assert manifest["targets"]["torch"]["hlo_nki_marker"] is False
    lines = open(tmp_path / "index.jsonl").read().splitlines()
    assert json.loads(lines[-1])["spec_id"] == res["spec_id"]
    # engine-facing fields
    assert res["nki_ok"] and res["nki_ms"] == pytest.approx(0.09)
    assert res["torch_ms"] == pytest.approx(0.03)
    assert res["nki_stats"]["spec_id"] == res["spec_id"]
    assert res["nki_stats"]["neff_sha256"] == manifest["targets"]["nki"]["neff_sha256"]
    assert manifest["targets"]["nki"]["stats"]["mean"] == pytest.approx(0.09)


def test_no_autotune_means_untuned_profile(tmp_path):
    runner = FakeRunner()
    res = orchestrate(tmp_path, runner, autotune=False)
    assert [c["mode"] for c in runner.calls] == ["profile"]
    with open(runner.calls[0]["args"]["--spec"]) as f:
        spec = json.load(f)
    assert spec["autotune_replay"] == [] and spec["autotune_enabled"] is False
    assert res["nki_ok"]


def test_identical_spec_reuses_validated_manifest_without_worker(tmp_path):
    runner = FakeRunner()
    first = orchestrate(tmp_path, runner, autotune=False)
    assert [c["mode"] for c in runner.calls] == ["profile"]
    second = orchestrate(tmp_path, runner, autotune=False)
    assert [c["mode"] for c in runner.calls] == ["profile"]  # no new worker launch
    assert second["spec_id"] == first["spec_id"]
    assert second["identity_source"] == "validated_manifest_reuse"
    assert second["nki_stats"]["neff_sha256"] == first["nki_stats"]["neff_sha256"]


def test_tampered_reused_artifact_triggers_rebuild_not_reuse(tmp_path):
    runner = FakeRunner()
    first = orchestrate(tmp_path, runner, autotune=False)
    with open(first["nki_stats"]["neff"], "ab") as f:
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
        runner=runner, profiler=spy_profiler)
    assert res["torch_stats"] is None and res["torch_ms"] != res["torch_ms"]  # nan
    assert "verification failed" in res["torch_err"]
    assert "torch" not in calls          # never timed
    assert res["nki_ok"] and calls == ["nki"]


def test_explicit_override_is_reported_unverified(tmp_path, monkeypatch):
    from core.nki_artifact import NEFF_PATH_ENV
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
    with open(res["manifest_path"]) as f:
        assert json.load(f)["targets"]["nki"]["verify_ok"] is None
