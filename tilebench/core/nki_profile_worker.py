"""Fresh-process worker for NKI case execution on Trainium.

Two modes, both launched by ``core/nki_orchestrator.py`` via
``sys.executable -m core.nki_profile_worker`` (never os.fork — the whole point
is a clean interpreter with empty torch-xla / PJRT / computation-cache state):

selector
    Runs ``impl_nki.run(..., autotune=True)`` on the saved case inputs to
    determine the autotune winner. Exports the complete canonical
    ``NkiAutotuner`` winner trace and verifies the output against the saved
    torch reference. Its compiled artifacts are NEVER profiled — the process
    exists only to pin down the winner identity, then exits.

profile
    The process whose executions are actually timed. Runs in a PRIVATE working
    directory with a PRIVATE Neuron compile cache and with the Neuron runtime's
    inspect facility enabled (``RUNTIME_INSPECT_ENV`` + a private output dir),
    installs the selector's exact winner trace, and replays it (zero candidate
    timings — a replay mismatch fails before anything is timed). Per target
    (torch baseline, then NKI) it: runs ``run()`` once and verifies the output;
    runs ``warmup`` untimed and ``repeat`` timed iterations, recording the
    range of XLA execution indices every timed iteration covered; and
    records EVERY NEFF/HLO pair the phase compiled (an operator may compile
    several graphs per run()) — see core/nki_artifact.py. The runtime writes
    the executed NEFFs + the per-execution system trace into the inspect dir.
    The worker then EXITS; the parent orchestrator (core/nki_orchestrator.py)
    ingests the trace, sums the device time of the executions inside each
    range and matches every executed NEFF to the recorded pairs by SHA256
    (core/nki_timer.py).

Only stdlib is imported at module level; torch / torch-xla imports happen
inside main() AFTER the Neuron debug + private-cache + inspect environment is
set.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback

from tilebench.core.nki_profile_spec import RUNTIME_INSPECT_ENV, atomic_write_json, sha256_file


def _snapshot_stems(root: str) -> set[str]:
    stems = set()
    for dirpath, _dirs, files in os.walk(root, followlinks=False):
        for fn in files:
            for suffix in (".neff", ".hlo_module.pb"):
                if fn.endswith(suffix):
                    stems.add(fn[: -len(suffix)])
    return stems


def _artifact_record(pair) -> dict:
    return {
        "stem": pair.stem,
        "neff_path": os.path.abspath(pair.neff_path),
        "hlo_path": os.path.abspath(pair.hlo_path),
        "neff_sha256": pair.neff_sha256,
        "hlo_sha256": pair.hlo_sha256,
        "hlo_nki_marker": pair.has_marker,
    }


def _run_selector(spec: dict, bundle: dict) -> dict:
    import importlib

    from tilebench.core import nki_autotune as na
    from tilebench.core.nki_timer import to_cpu, to_xla_device
    from tilebench.core.verifier import verify
    from torch_xla.core import xla_model as xm

    impl_nki = importlib.import_module(
        f"tilebench.benchmarks.operators.{spec['operator']}.impl_nki")

    na.clear_tuning_trace()
    xla_inputs = to_xla_device(tuple(bundle["inputs"]))
    kw = dict(spec["run_kwargs"])
    if spec["run_accepts_autotune"]:
        kw["autotune"] = True
    out = impl_nki.run(*xla_inputs, **kw)
    xm.mark_step()
    xm.wait_device_ops()
    ok, err = verify(to_cpu(out), bundle["reference_output"],
                     atol=spec["verify_atol"], rtol=spec["verify_rtol"])
    last_cfg = None
    get_last = getattr(impl_nki, "get_last_config", None)
    if callable(get_last):
        last_cfg = get_last()
    return {
        "ok": bool(ok),
        "mode": "selector",
        "verify_ok": bool(ok),
        "verify_error": None if ok else str(err),
        "tuning_trace": na.export_tuning_trace(),
        "last_config": last_cfg,
    }


def _xla_execution_count() -> int:
    """Number of XLA graph executions this process has submitted so far
    (torch-xla's ``ExecuteTime`` metric count). One per runtime execution, so
    it indexes the runtime trace's execution list without any clock."""
    import torch_xla.debug.metrics as met

    data = met.metric_data("ExecuteTime")
    return int(data[0]) if data else 0


def _timed_windows(run_once, *, warmup: int, repeat: int) -> list[list[int]]:
    """``warmup`` untimed + ``repeat`` timed executions of one run(); each timed
    iteration is recorded as the half-open range ``[begin, end)`` of XLA
    execution indices it covered — the parent slices the trace's execution
    list (ordered by device start) with it, so no wall-clock is ever compared
    against the trace's timebase."""
    for _ in range(warmup):
        run_once()
    windows = []
    for _ in range(repeat):
        begin = _xla_execution_count()
        run_once()
        end = _xla_execution_count()
        if end <= begin:
            raise RuntimeError("timed run() submitted no XLA execution")
        windows.append([begin, end])
    return windows


def _new_pair_records(cwd: str, pre_stems: set[str]) -> list[dict]:
    """Every valid pair that appeared since ``pre_stems`` (no marker checks —
    used to record what a FAILED phase still dumped, so a reuse run knows
    those pairs exist)."""
    from tilebench.core.nki_artifact import discover_pairs

    pairs, _rejections = discover_pairs([cwd])
    return [_artifact_record(p) for p in sorted(pairs, key=lambda p: p.stem)
            if p.stem not in pre_stems]


def _run_phase(*, fn, kw, xla_inputs, reference, spec, expected_stems, nki_phase,
               cwd, ctx) -> dict:
    """Run one target: verify one run(), record the phase's pairs, time
    warmup+repeat iterations when verification passed.

    Artifact identity is recorded whether or not verification passes (it is a
    compile fact, and a reuse run must know every pair this spec dumps); only
    a verified run is timed. A run() that raises (unsupported op, compile
    error) is reported in ``verify_error`` with whatever pairs it still dumped
    (``[]`` when it compiled nothing).
    """
    from tilebench.core.nki_artifact import resolve_expected_pairs, resolve_phase_pairs
    from tilebench.core.nki_timer import to_cpu
    from tilebench.core.verifier import verify
    from torch_xla.core import xla_model as xm

    def run_once():
        out = fn(*xla_inputs, **kw)   # keep the output alive until the graph ran
        xm.mark_step()
        xm.wait_device_ops()
        return out

    entry = {"verify_ok": False, "verify_error": None, "windows": None,
             "artifacts": None, "phase_failed": False}
    pre_stems = _snapshot_stems(cwd)
    try:
        out = run_once()
    except Exception as e:  # noqa: BLE001 — reported, never silently timed
        entry["phase_failed"] = True
        entry["verify_error"] = (f"phase failed: {type(e).__name__}: "
                                 f"{str(e).splitlines()[0][:300]}")
        entry["artifacts"] = _new_pair_records(cwd, pre_stems)
        return entry
    ok, err = verify(to_cpu(out), reference,
                     atol=spec["verify_atol"], rtol=spec["verify_rtol"])
    entry["verify_ok"] = bool(ok)
    entry["verify_error"] = None if ok else str(err)
    if expected_stems:
        pairs = resolve_expected_pairs([cwd], stems=expected_stems, nki_phase=nki_phase,
                                       pre_stems=pre_stems, context=ctx)
    else:
        pairs = resolve_phase_pairs([cwd], nki_phase=nki_phase, exclude_stems=pre_stems,
                                    context=ctx)
    entry["artifacts"] = [_artifact_record(p) for p in pairs]
    if ok:
        entry["windows"] = _timed_windows(run_once, warmup=int(spec["warmup"]),
                                          repeat=int(spec["repeat"]))
    return entry


def _run_profile(spec: dict, bundle: dict) -> dict:
    import importlib

    from tilebench.core import nki_autotune as na
    from tilebench.core.nki_timer import to_xla_device
    from torch_xla.core import xla_model as xm

    cwd = os.getcwd()
    ctx = {"operator": spec["operator"], "case_label": spec["case_label"],
           "spec_id": spec.get("spec_id", ""), "worker": "profile"}
    result: dict = {"ok": True, "mode": "profile", "torch": None, "nki": None}
    reference = bundle["reference_output"]

    impl_torch = importlib.import_module(
        f"tilebench.benchmarks.operators.{spec['operator']}.impl_torch")

    # Reuse mode (parent found a validated manifest for this spec_id): the
    # private artifacts/cache were kept, so compiles are cache hits and dump
    # nothing new. Identity then means "the expected pairs are still there and
    # nothing new appeared"; correctness is still verified on THIS run's inputs.
    expected = spec.get("expected_stems") or {}

    # ---- device transfers first, flushed, so their graphlets are excluded ----
    xla_inputs = to_xla_device(tuple(bundle["inputs"]))
    xm.mark_step()
    xm.wait_device_ops()

    # ---- torch baseline -------------------------------------------------------
    # A failure here (unsupported op on trn2, XLA numeric mismatch, identity
    # ambiguity) is recorded as the torch baseline's own error and must NOT
    # abort the NKI phase: the NKI result stands on its own.
    try:
        result["torch"] = _run_phase(
            fn=impl_torch.run, kw={}, xla_inputs=xla_inputs, reference=reference, spec=spec,
            expected_stems=expected.get("torch"), nki_phase=False, cwd=cwd,
            ctx=dict(ctx, target="torch"))
    except Exception as e:  # noqa: BLE001 — identity errors: reported, never timed
        result["torch"] = {
            "verify_ok": False,
            "verify_error": f"torch baseline phase failed: {type(e).__name__}: "
                            f"{str(e).splitlines()[0][:300]}",
            "windows": None, "artifacts": None, "phase_failed": True,
        }
    if not result["torch"]["verify_ok"]:
        # Drain the failed graph only AFTER its except block has ended: the
        # traceback kept the dead lazy tensors alive, and syncing those trips
        # torch-xla's "Check failed: tensor_data" and would take the NKI phase
        # down with it.
        gc.collect()
        try:
            xm.mark_step()
            xm.wait_device_ops()
        except Exception as e:  # noqa: BLE001
            result["torch"]["verify_error"] += (
                f" (device drain after failure also raised: {type(e).__name__})")

    # ---- NKI: exact winner replay ---------------------------------------------
    if spec["nki_enabled"]:
        impl_nki = importlib.import_module(
            f"tilebench.benchmarks.operators.{spec['operator']}.impl_nki")
        replay = spec["autotune_replay"]
        kw = dict(spec["run_kwargs"])
        if replay:
            na.install_tuning_replay(replay, strict=True)
            kw["autotune"] = True
        elif spec["run_accepts_autotune"]:
            kw["autotune"] = False
        na.clear_tuning_trace()
        entry = _run_phase(
            fn=impl_nki.run, kw=kw, xla_inputs=xla_inputs, reference=reference, spec=spec,
            expected_stems=expected.get("nki"), nki_phase=True, cwd=cwd,
            ctx=dict(ctx, target="nki"))
        if replay and not entry["phase_failed"]:
            na.assert_tuning_replay_consumed()
        entry.update({
            "replay_installed": bool(replay),
            "replay_consumed": bool(replay),
            "executed_trace": na.export_tuning_trace(),
        })
        if not entry["verify_ok"]:
            result["ok"] = False
        result["nki"] = entry

    # The parent checks this against the number of executions in the runtime
    # trace: the windows above are only meaningful if the two lists line up.
    result["xla_executions"] = _xla_execution_count()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tilebench.core.nki_profile_worker")
    parser.add_argument("--mode", required=True, choices=("selector", "profile"))
    parser.add_argument("--spec", required=True, help="path to spec.json")
    parser.add_argument("--bundle", required=True, help="path to input_bundle.pt")
    parser.add_argument("--result", required=True, help="path to write result JSON")
    parser.add_argument("--workdir", required=True, help="private CWD for this worker")
    parser.add_argument("--cache-dir", default=None,
                        help="private NEURON_COMPILE_CACHE_URL for this worker")
    parser.add_argument("--inspect-dir", default=None,
                        help="profile mode: private NEURON_RT_INSPECT_OUTPUT_DIR")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)

    # Absolutize every path argument before the chdir below invalidates
    # relative interpretations.
    for attr in ("spec", "bundle", "result", "workdir",
                 "cache_dir", "inspect_dir", "repo_root"):
        val = getattr(args, attr)
        if val:
            setattr(args, attr, os.path.abspath(val))

    # Environment must be final BEFORE torch/torch-xla are imported: the debug
    # dumps land in CWD, the compile cache and the runtime inspect output are
    # resolved from the env.
    os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ["NEURON_COMPILE_CACHE_URL"] = args.cache_dir
    if args.inspect_dir:
        os.makedirs(args.inspect_dir, exist_ok=True)
        os.environ.update(RUNTIME_INSPECT_ENV)
        os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = args.inspect_dir
    os.makedirs(args.workdir, exist_ok=True)
    os.chdir(args.workdir)
    if args.repo_root and args.repo_root not in sys.path:
        sys.path.insert(0, args.repo_root)

    result: dict
    try:
        with open(args.spec) as f:
            spec = json.load(f)

        import torch  # noqa: F401  (after env setup, before torch_xla use)
        bundle = torch.load(args.bundle, weights_only=False)
        bundle_sha = sha256_file(args.bundle)

        if args.mode == "selector":
            result = _run_selector(spec, bundle)
        else:
            result = _run_profile(spec, bundle)
        result["input_bundle_sha256"] = bundle_sha
        result["workdir"] = os.path.abspath(args.workdir)
        result["cache_dir"] = os.path.abspath(args.cache_dir) if args.cache_dir else None
        result["inspect_dir"] = os.path.abspath(args.inspect_dir) if args.inspect_dir else None
    except BaseException as e:  # report crashes as structured results too
        result = {"ok": False, "mode": args.mode, "error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()}
        atomic_write_json(args.result, result)
        return 1

    atomic_write_json(args.result, result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
