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
    The process whose artifacts are actually timed. Runs in a PRIVATE working
    directory with a PRIVATE Neuron compile cache, installs the selector's
    exact winner trace, and replays it (zero candidate timings — a replay
    mismatch fails before anything is profiled). Exactly one torch graph and
    (when NKI is enabled) exactly one NKI graph are compiled here, so artifact
    identity reduces to "the exactly-one validated pair in this private root"
    (marker-separated) — see core/nki_artifact.py. The worker records the
    validated pair paths + SHA256s in its result and EXITS; the parent
    orchestrator re-validates those hashes and hardware-times the exact NEFFs
    with core/nki_timer.profile_neff AFTER this process has released the
    NeuronCores (neuron-explorer capture needs the cores; capturing while the
    worker's PJRT client is still alive races NRT allocation — observed as
    "Logical Neuron Core(s) not available ... cores busy" on trn2).

Only stdlib is imported at module level; torch / torch-xla imports happen
inside main() AFTER the Neuron debug + private-cache environment is set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from core.nki_profile_spec import atomic_write_json, sha256_file


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

    from core import nki_autotune as na
    from core.nki_timer import to_cpu, to_xla_device
    from core.verifier import verify
    from torch_xla.core import xla_model as xm

    impl_nki = importlib.import_module(
        f"benchmarks.operators.{spec['operator']}.impl_nki")

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


def _run_profile(spec: dict, bundle: dict) -> dict:
    import importlib

    from core import nki_autotune as na
    from core.nki_artifact import resolve_unique
    from core.nki_timer import to_cpu, to_xla_device
    from core.verifier import verify
    from torch_xla.core import xla_model as xm

    cwd = os.getcwd()
    ctx = {"operator": spec["operator"], "case_label": spec["case_label"],
           "spec_id": spec.get("spec_id", ""), "worker": "profile"}
    result: dict = {"ok": True, "mode": "profile", "torch": None, "nki": None}

    impl_torch = importlib.import_module(
        f"benchmarks.operators.{spec['operator']}.impl_torch")

    # ---- device transfers first, flushed, so their graphlets are excluded ----
    xla_inputs = to_xla_device(tuple(bundle["inputs"]))
    xm.mark_step()
    xm.wait_device_ops()
    pre_compute_stems = _snapshot_stems(cwd)

    # ---- torch baseline: exactly one new non-NKI graph ----------------------
    torch_out = impl_torch.run(*xla_inputs)
    xm.mark_step()
    xm.wait_device_ops()
    torch_pair = resolve_unique(
        [cwd], expect_marker=False, exclude_stems=pre_compute_stems,
        context=dict(ctx, target="torch"))
    torch_ok, torch_err = verify(to_cpu(torch_out), bundle["reference_output"],
                                 atol=spec["verify_atol"], rtol=spec["verify_rtol"])
    result["torch"] = {
        "verify_ok": bool(torch_ok),
        "verify_error": None if torch_ok else str(torch_err),
        "artifact": _artifact_record(torch_pair),
    }

    # ---- NKI: exact winner replay, exactly one marker-bearing graph ---------
    if spec["nki_enabled"]:
        impl_nki = importlib.import_module(
            f"benchmarks.operators.{spec['operator']}.impl_nki")
        replay = spec["autotune_replay"]
        kw = dict(spec["run_kwargs"])
        if replay:
            na.install_tuning_replay(replay, strict=True)
            kw["autotune"] = True
        elif spec["run_accepts_autotune"]:
            kw["autotune"] = False
        na.clear_tuning_trace()
        nki_out = impl_nki.run(*xla_inputs, **kw)
        xm.mark_step()
        xm.wait_device_ops()
        if replay:
            na.assert_tuning_replay_consumed()
        nki_ok, nki_err = verify(to_cpu(nki_out), bundle["reference_output"],
                                 atol=spec["verify_atol"], rtol=spec["verify_rtol"])
        nki_entry: dict = {
            "verify_ok": bool(nki_ok),
            "verify_error": None if nki_ok else str(nki_err),
            "replay_installed": bool(replay),
            "replay_consumed": bool(replay),
            "executed_trace": na.export_tuning_trace(),
            "artifact": None,
        }
        if nki_ok:
            nki_pair = resolve_unique([cwd], expect_marker=True,
                                      context=dict(ctx, target="nki"))
            nki_entry["artifact"] = _artifact_record(nki_pair)
        else:
            result["ok"] = False
        result["nki"] = nki_entry

    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="core.nki_profile_worker")
    parser.add_argument("--mode", required=True, choices=("selector", "profile"))
    parser.add_argument("--spec", required=True, help="path to spec.json")
    parser.add_argument("--bundle", required=True, help="path to input_bundle.pt")
    parser.add_argument("--result", required=True, help="path to write result JSON")
    parser.add_argument("--workdir", required=True, help="private CWD for this worker")
    parser.add_argument("--cache-dir", default=None,
                        help="private NEURON_COMPILE_CACHE_URL for this worker")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)

    # Absolutize every path argument before the chdir below invalidates
    # relative interpretations.
    for attr in ("spec", "bundle", "result", "workdir",
                 "cache_dir", "repo_root"):
        val = getattr(args, attr)
        if val:
            setattr(args, attr, os.path.abspath(val))

    # Environment must be final BEFORE torch/torch-xla are imported: the debug
    # dumps land in CWD and the compile cache is resolved from the env.
    os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
    os.environ["XLA_IR_DEBUG"] = "1"
    os.environ["XLA_HLO_DEBUG"] = "1"
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ["NEURON_COMPILE_CACHE_URL"] = args.cache_dir
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
    except BaseException as e:  # report crashes as structured results too
        result = {"ok": False, "mode": args.mode, "error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()}
        atomic_write_json(args.result, result)
        return 1

    atomic_write_json(args.result, result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
