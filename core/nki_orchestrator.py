"""Parent-side orchestration of exact-identity NKI profiling on Trainium.

Flow (see the module docstrings of nki_profile_worker / nki_artifact for the
identity rules):

    PARENT (never imports torch-xla)
      |-- save exact CPU input + torch-reference bundle
      |-- if autotune requested and the operator has an NKI impl:
      |     SELECTOR subprocess: tunes, verifies, exports the canonical
      |     winner trace, exits. Its artifacts are never profiled.
      |-- build the canonical NkiProfileSpec (winner trace included)
      |     -> spec_id -> private per-spec directory
      |-- if a validated manifest for this spec_id already exists:
      |     re-validate hashes/marker, KEEP the private cache/artifacts
      |     (identity_source = "validated_manifest_reuse")
      |   else wipe the private space (fresh compile)
      |-- PROFILE subprocess (always — input values change per run, so
      |     correctness is re-verified every time): private CWD + private
      |     compile cache + private runtime-inspect dir, exact winner replay
      |     (no candidate timing), verify, then warmup + repeat timed
      |     iterations per target, each recorded as a range of XLA execution
      |     indices; every pair the phase compiled is recorded (on reuse:
      |     must be the manifest's)
      |-- PARENT re-validates SHA256s, ingests the runtime trace (execution
      |     count must equal the worker's), sums the device time of the
      |     executions inside each range, and matches
      |     every executed NEFF (runtime-written, byte-identical to the
      |     compiler dump) to a recorded pair by SHA256
      |-- write <spec_dir>/manifest.json (authoritative) + append the global
          audit index results/logs/nki_neff_manifest.jsonl

No step ever selects an artifact by mtime, sequence number, glob order, or
"most recent compiler event".
"""
from __future__ import annotations

import dataclasses
import datetime
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
from typing import Callable

from core.nki_artifact import (NkiArtifactIdentityError, resolve_explicit_override,
                               validate_manifest_reuse, validate_pair)
from core.nki_profile_spec import (MANIFEST_SCHEMA_VERSION, RUNTIME_INSPECT_ENV,
                                   NkiProfileSpec, append_jsonl_locked,
                                   atomic_write_json, describe_inputs, make_case_label,
                                   sha256_file, spec_lock)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_HARNESS_FILES = (
    "core/nki_autotune.py",
    "core/nki_timer.py",
    "core/nki_artifact.py",
    "core/nki_profile_spec.py",
    "core/nki_profile_worker.py",
    "core/nki_orchestrator.py",
)

_VERSION_PKGS = ("torch", "torch-xla", "torch-neuronx", "nki", "neuronx-cc",
                 "libneuronxla")


class NkiOrchestrationError(RuntimeError):
    """A worker phase failed; no latency may be reported for this case."""


def _software_versions() -> dict:
    out = {}
    for pkg in _VERSION_PKGS:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "absent"
    return out


def _source_hashes(paths) -> dict:
    out = {}
    for p in paths:
        ap = p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)
        out[os.path.relpath(ap, _REPO_ROOT)] = sha256_file(ap)
    return out


_SOURCE_ROOTS = tuple(os.path.join(_REPO_ROOT, d) + os.sep
                      for d in ("benchmarks", "core", "data"))


def _in_repo_source_file(module) -> str | None:
    f = getattr(module, "__file__", None)
    if not f or not f.endswith(".py"):
        return None
    f = os.path.abspath(f)
    return f if f.startswith(_SOURCE_ROOTS) else None


def _operator_source_files(operator: str, impl_modules) -> list:
    """Every in-repo .py file the operator's impl modules transitively pull in.

    Walks module globals (functions, classes, imported modules) so an operator
    that reuses another operator's kernel — e.g. streamk_matmul importing from
    matmul_fp32_fp16_fp8/impl_nki.py — gets that file into its spec identity;
    otherwise editing the helper would leave spec_id (and manifest reuse)
    unchanged while the compiled graph changed.
    """
    import types

    files = {os.path.join(_REPO_ROOT, "benchmarks", "operators", operator, "impl_torch.py")}
    seen: set = set()
    queue = [m for m in impl_modules if m is not None]
    while queue:
        mod = queue.pop()
        if id(mod) in seen:
            continue
        seen.add(id(mod))
        f = _in_repo_source_file(mod)
        if f is None:
            continue
        files.add(f)
        for v in list(vars(mod).values()):
            dep = v if isinstance(v, types.ModuleType) else inspect.getmodule(v)
            if dep is not None and id(dep) not in seen and _in_repo_source_file(dep):
                queue.append(dep)
    return sorted(os.path.relpath(f, _REPO_ROOT) for f in files if os.path.isfile(f))


def _run_kwargs_for(fn, block_size) -> tuple[dict, bool]:
    """Mirror engine._run_kwargs, but report `autotune` acceptance separately —
    the workers decide its value (selector: True; profile: replay-driven)."""
    sig = inspect.signature(fn).parameters
    kw = {}
    if "block_size" in sig:
        kw["block_size"] = block_size
    return kw, "autotune" in sig


def _default_runner(cmd: list[str], *, cwd: str, env_overrides: dict) -> int:
    env = dict(os.environ)
    env.update(env_overrides)
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def _read_result(path: str, phase: str) -> dict:
    if not os.path.isfile(path):
        raise NkiOrchestrationError(f"{phase} worker wrote no result file ({path})")
    with open(path) as f:
        res = json.load(f)
    if res.get("error"):
        raise NkiOrchestrationError(
            f"{phase} worker failed: {res['error']}\n{res.get('traceback', '')[-2000:]}")
    return res


def _launch_worker(runner: Callable, *, mode: str, spec_path: str, bundle_path: str,
                   result_path: str, workdir: str,
                   cache_dir: str | None, inspect_dir: str | None, python: str) -> dict:
    cmd = [python, "-m", "core.nki_profile_worker",
           "--mode", mode, "--spec", spec_path, "--bundle", bundle_path,
           "--result", result_path, "--workdir", workdir,
           "--repo-root", _REPO_ROOT]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]
    if inspect_dir:
        cmd += ["--inspect-dir", inspect_dir]
    env_overrides = {"PYTHONPATH": _REPO_ROOT + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    rc = runner(cmd, cwd=_REPO_ROOT, env_overrides=env_overrides)
    res = _read_result(result_path, mode)
    if rc != 0 and res.get("ok"):
        raise NkiOrchestrationError(f"{mode} worker exit={rc} but result claims ok")
    return res


def dataclasses_asdict(pair) -> dict:
    return dataclasses.asdict(pair)


def _executed_shas(rec: dict | None) -> list[str]:
    """SHA256s of the NEFFs a target's timed iterations actually executed."""
    return sorted(e["neff_sha256"] for e in ((rec or {}).get("executed") or []))


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def profile_case_on_neuron(
    *,
    operator: str,
    params: dict,
    dtype_str: str,
    inputs,
    ref_output,
    impl_nki,
    block_size,
    autotune: bool,
    verify_atol,
    verify_rtol,
    warmup: int,
    repeat: int,
    base_dir: str = "results/logs/nki_profiles",
    index_path: str = "results/logs/nki_neff_manifest.jsonl",
    python: str | None = None,
    runner: Callable | None = None,
    profiler: Callable | None = None,
    executions_loader: Callable | None = None,
) -> dict:
    """Time the torch baseline (and, when available, the NKI backend) for one
    benchmark case on Trainium with exact artifact identity.

    Returns the fields core/engine.py consumes: torch_stats/torch_ms,
    nki_stats/nki_ms/nki_ok/nki_err/nki_cfg, plus spec_id/manifest_path.
    ``profiler`` (capture/view of one NEFF) is used only for the explicit
    $NKI_NEFF_PATH override; ``executions_loader`` ingests the runtime trace.
    """
    import torch  # parent needs plain torch only (bundle serialization)

    from core.nki_timer import (NkiTraceError, find_session_dir, session_neffs,
                                time_windows)

    runner = runner or _default_runner
    python = python or sys.executable
    if profiler is None:
        from core.nki_timer import profile_neff as profiler
    if executions_loader is None:
        from core.nki_timer import load_session_executions as executions_loader

    nki_enabled = impl_nki is not None
    case_label = make_case_label(params, dtype_str)
    # Absolute: the workers chdir into private CWDs, so every path handed to
    # them must survive that.
    case_root = os.path.abspath(os.path.join(base_dir, operator, case_label))
    index_path = os.path.abspath(index_path)
    os.makedirs(case_root, exist_ok=True)
    ctx = {"operator": operator, "case_label": case_label}

    if nki_enabled:
        run_kwargs, accepts_autotune = _run_kwargs_for(impl_nki.run, block_size)
    else:
        run_kwargs, accepts_autotune = {}, False
    impl_torch = sys.modules.get(f"benchmarks.operators.{operator}.impl_torch")
    operator_sources = _operator_source_files(operator, (impl_nki, impl_torch))

    def cpu(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        if isinstance(x, (tuple, list)):
            return type(x)(cpu(v) for v in x)
        return x

    bundle = {"schema_version": 1,
              "inputs": [cpu(x) for x in inputs],
              "reference_output": cpu(ref_output)}

    # ---- selector phase: pin down the exact autotune winner -----------------
    selector_result = None
    autotune_replay: list = []
    nki_cfg = None
    if autotune and nki_enabled and accepts_autotune:
        sel_root = os.path.join(case_root, f"selector_pid{os.getpid()}")
        shutil.rmtree(sel_root, ignore_errors=True)
        os.makedirs(sel_root)
        sel_bundle = os.path.join(sel_root, "input_bundle.pt")
        torch.save(bundle, sel_bundle)
        sel_spec = {
            "operator": operator, "case_label": case_label,
            "run_kwargs": run_kwargs, "run_accepts_autotune": accepts_autotune,
            "verify_atol": verify_atol, "verify_rtol": verify_rtol,
        }
        sel_spec_path = os.path.join(sel_root, "selector_spec.json")
        atomic_write_json(sel_spec_path, sel_spec)
        selector_result = _launch_worker(
            runner, mode="selector", spec_path=sel_spec_path,
            bundle_path=sel_bundle,
            result_path=os.path.join(sel_root, "selector_result.json"),
            workdir=os.path.join(sel_root, "work"),
            cache_dir=None, inspect_dir=None, python=python)
        if not selector_result.get("verify_ok"):
            raise NkiOrchestrationError(
                f"selector verification failed for {operator}/{case_label}: "
                f"{selector_result.get('verify_error')}")
        autotune_replay = selector_result.get("tuning_trace") or []
        nki_cfg = selector_result.get("last_config")

    # §6.7: --autotune with no NKI-specific tuner records → profile untuned.
    autotune_enabled = bool(autotune_replay)

    # ---- canonical launch spec ----------------------------------------------
    spec = NkiProfileSpec(
        operator=operator,
        case_label=case_label,
        case_params=dict(params),
        dtype=dtype_str,
        block_size=block_size,
        run_kwargs=run_kwargs,
        input_specs=describe_inputs(inputs),
        autotune_enabled=autotune_enabled,
        autotune_replay=autotune_replay,
        verify_atol=verify_atol,
        verify_rtol=verify_rtol,
        warmup=int(warmup),
        repeat=int(repeat),
        nki_enabled=nki_enabled,
        operator_source_sha256=_source_hashes(operator_sources),
        harness_source_sha256=_source_hashes(_HARNESS_FILES),
        neuron_target=os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", ""),
        logical_nc_config=os.environ.get("NEURON_LOGICAL_NC_CONFIG", ""),
        neuron_cc_flags=os.environ.get("NEURON_CC_FLAGS", ""),
        software_versions=_software_versions(),
    )
    spec_id = spec.spec_id
    spec_dir = os.path.abspath(os.path.join(case_root, spec_id[:16]))
    artifacts_dir = os.path.join(spec_dir, "artifacts")
    cache_dir = os.path.join(spec_dir, "cache")
    inspect_dir = os.path.join(spec_dir, "inspect")
    profile_dir = os.path.join(spec_dir, "profile")
    manifest_path = os.path.join(spec_dir, "manifest.json")
    spec_extra = dict(spec.to_dict(), spec_id=spec_id)

    with spec_lock(spec_dir):
        atomic_write_json(os.path.join(spec_dir, "spec.json"), spec_extra)
        bundle_path = os.path.join(spec_dir, "input_bundle.pt")
        torch.save(bundle, bundle_path)
        bundle_sha = sha256_file(bundle_path)
        if selector_result is not None:
            atomic_write_json(os.path.join(spec_dir, "selector_result.json"),
                              selector_result)

        identity_source = "private_debug_dump"
        targets: dict = {}
        worker_result = None

        override_pair = resolve_explicit_override(dict(ctx, spec_id=spec_id))

        # ---- exact reuse of a previously validated manifest ------------------
        # Reuse means "same validated artifact, skip recompilation" — NOT "skip
        # verification": the case's input VALUES are regenerated per run and
        # are deliberately outside spec_id, so the profile worker always runs
        # and verifies the current outputs. On reuse the private cache and
        # artifacts are kept (compile-cache hits, no new dumps) and the
        # worker-identified artifact must hash-match the manifest.
        reuse_pairs = None
        if override_pair is None and os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                old_manifest = json.load(f)
            try:
                reuse_pairs = validate_manifest_reuse(
                    old_manifest, spec_id=spec_id, allowed_roots=[spec_dir])
            except NkiArtifactIdentityError as e:
                # Safe rebuild: this spec's private artifact space is wiped as a
                # unit (artifacts + cache + manifest) and rebuilt from scratch.
                print(f"  NKI manifest reuse invalid — rebuilding spec "
                      f"{spec_id[:16]}: {str(e).splitlines()[0]}")
                os.remove(manifest_path)
                reuse_pairs = None
        # The runtime trace is per run: its dir is wiped in both modes.
        shutil.rmtree(inspect_dir, ignore_errors=True)
        if reuse_pairs is not None:
            identity_source = "validated_manifest_reuse"
            shutil.rmtree(profile_dir, ignore_errors=True)
        else:
            # Fresh run: artifacts + cache are wiped together so the private
            # CWD is guaranteed to receive this run's dumps (a stale private
            # cache with an empty artifacts dir could otherwise cache-hit and
            # dump nothing).
            for p in (artifacts_dir, profile_dir):
                shutil.rmtree(p, ignore_errors=True)
            if override_pair is None:
                shutil.rmtree(cache_dir, ignore_errors=True)

        if True:  # (kept flat: fresh and reuse share the worker + validation path)
            worker_spec = dict(spec_extra,
                               run_accepts_autotune=accepts_autotune,
                               nki_enabled=nki_enabled and override_pair is None,
                               expected_stems=({t: [p.stem for p in ps]
                                                for t, ps in reuse_pairs.items()}
                                               if reuse_pairs is not None else {}))
            worker_spec_path = os.path.join(spec_dir, "worker_spec.json")
            atomic_write_json(worker_spec_path, worker_spec)
            worker_result = _launch_worker(
                runner, mode="profile", spec_path=worker_spec_path,
                bundle_path=bundle_path,
                result_path=os.path.join(spec_dir, "worker_result.json"),
                workdir=artifacts_dir,
                cache_dir=cache_dir, inspect_dir=inspect_dir, python=python)

            # ---- re-validate every artifact the worker recorded --------------
            # A graph whose output failed verification is never timed: an
            # invalid torch baseline would otherwise become the speedup
            # denominator.
            pairs_by_sha: dict = {}
            timed_targets = []
            for target in ("torch", "nki"):
                entry = worker_result.get(target)
                if not entry:
                    continue
                rec = {"verify_ok": entry["verify_ok"],
                       "verify_error": entry["verify_error"],
                       "artifacts": entry.get("artifacts"),   # identity even when not timed
                       "executed": None, "stats": None}
                targets[target] = rec
                if not (entry.get("artifacts") and entry["verify_ok"]):
                    continue
                pairs = []
                for art in entry["artifacts"]:
                    pair = validate_pair(art["neff_path"], require_marker=False,
                                         allowed_roots=[spec_dir],
                                         context=dict(ctx, spec_id=spec_id, target=target))
                    if pair.neff_sha256 != art["neff_sha256"] or \
                            pair.hlo_sha256 != art["hlo_sha256"] or \
                            pair.has_marker != bool(art["hlo_nki_marker"]):
                        raise NkiArtifactIdentityError(
                            f"artifact {art['stem']!r} changed between worker "
                            f"identification and parent validation for target {target!r}",
                            context=dict(ctx, spec_id=spec_id, target=target),
                            roots=[spec_dir])
                    pairs.append(pair)
                if reuse_pairs is not None and target in reuse_pairs:
                    got = {(p.stem, p.neff_sha256, p.hlo_sha256) for p in pairs}
                    want = {(p.stem, p.neff_sha256, p.hlo_sha256) for p in reuse_pairs[target]}
                    if got != want:
                        raise NkiArtifactIdentityError(
                            f"identical spec_id produced different {target!r} artifacts "
                            f"than the validated manifest (non-deterministic compile or "
                            f"environment drift)",
                            context=dict(ctx, spec_id=spec_id, target=target),
                            roots=[spec_dir],
                            candidates=[dataclasses_asdict(p) for p in pairs]
                            + [dataclasses_asdict(p) for p in reuse_pairs[target]])
                for p in pairs:
                    pairs_by_sha[p.neff_sha256] = p
                timed_targets.append(target)

            # ---- runtime trace: windows -> device time, executed NEFF -> pair --
            session_dir = parquet_dir = None
            if timed_targets:
                session_dir = find_session_dir(inspect_dir)
                executions, parquet_dir = executions_loader(
                    session_dir, data_path=os.path.join(profile_dir, "ne"),
                    display_name=f"{operator}-{case_label}-{spec_id[:16]}")
                if len(executions) != worker_result.get("xla_executions"):
                    raise NkiTraceError(
                        f"runtime trace holds {len(executions)} execution(s) but the worker "
                        f"submitted {worker_result.get('xla_executions')} XLA execution(s) — "
                        f"the timed windows cannot be attributed (trace incomplete: raise "
                        f"NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC; or executions the "
                        f"framework did not submit)")
                runtime_neffs = session_neffs(session_dir)
                runtime_sha = {m: sha256_file(p) for m, p in runtime_neffs.items()}
            for target in timed_targets:
                rec = targets[target]
                tctx = dict(ctx, spec_id=spec_id, target=target)
                stats = time_windows(executions, worker_result[target]["windows"], tag=target)
                per_model = stats.pop("per_model")
                executed = []
                for model_id, info in sorted(per_model.items()):
                    if model_id not in runtime_sha:
                        raise NkiArtifactIdentityError(
                            f"executed model {model_id} has no runtime-written NEFF in "
                            f"{session_dir} (NEURON_RT_INSPECT_DEVICE_PROFILE must be 1)",
                            context=tctx, roots=[inspect_dir])
                    pair = pairs_by_sha.get(runtime_sha[model_id])
                    if pair is None:
                        raise NkiArtifactIdentityError(
                            f"executed NEFF (model {model_id}, sha256 {runtime_sha[model_id]}) "
                            f"is not among the artifacts this run compiled",
                            context=tctx, roots=[spec_dir, inspect_dir],
                            candidates=[dataclasses_asdict(p) for p in pairs_by_sha.values()])
                    ntff = os.path.join(session_dir, f"{model_id}_vnc_0.ntff")
                    executed.append({
                        "model_id": model_id, "stem": pair.stem,
                        "neff_sha256": pair.neff_sha256, "hlo_sha256": pair.hlo_sha256,
                        "hlo_nki_marker": pair.has_marker,
                        "count_per_iteration": info["count_per_iteration"],
                        "mean_ms": info["mean_ms"],
                        "runtime_neff": runtime_neffs[model_id],
                        "ntff": ntff if os.path.isfile(ntff) else None,
                    })
                marked = [e for e in executed if e["hlo_nki_marker"]]
                if target == "nki" and not marked:
                    raise NkiArtifactIdentityError(
                        "the timed NKI iterations executed no marker-bearing (NKI) graph",
                        context=tctx, roots=[spec_dir], candidates=executed)
                if target == "torch" and marked:
                    raise NkiArtifactIdentityError(
                        "the torch baseline iterations executed an NKI graph",
                        context=tctx, roots=[spec_dir], candidates=executed)
                rec["executed"] = executed
                rec["stats"] = dict(stats, warmup=int(warmup), session_dir=session_dir,
                                    parquet_dir=parquet_dir)

            if override_pair is not None:
                identity_source = "explicit_override"
                stats = profiler(override_pair.neff_path, warmup=int(warmup),
                                 out_dir=profile_dir, tag="nki")
                targets["nki"] = {
                    "verify_ok": None,
                    "verify_error": "verification skipped: explicit $NKI_NEFF_PATH override",
                    "artifacts": [{
                        "stem": override_pair.stem,
                        "neff_path": override_pair.neff_path,
                        "hlo_path": override_pair.hlo_path,
                        "neff_sha256": override_pair.neff_sha256,
                        "hlo_sha256": override_pair.hlo_sha256,
                        "hlo_nki_marker": override_pair.has_marker,
                    }],
                    "executed": None,
                    "stats": stats,
                }

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "spec_id": spec_id,
            "manifest_path": manifest_path,
            "operator": operator,
            "case_label": case_label,
            "case_params": dict(params),
            "dtype": dtype_str,
            "autotune_enabled": autotune_enabled,
            "autotune_replay": autotune_replay,
            "identity_source": identity_source,
            "identity_status": "validated",
            "private_cwd": artifacts_dir,
            "private_cache_dir": cache_dir,
            "targets": targets,
            "runtime_session_dir": session_dir,
            "runtime_parquet_dir": parquet_dir,
            "profile_input_mode": next(
                (t["stats"]["profile_input_mode"] for t in targets.values()
                 if t.get("stats")), None),
            "input_bundle_sha256": bundle_sha,
            "software_versions": spec.software_versions,
            "compiler_environment": {
                "NEURON_PLATFORM_TARGET_OVERRIDE": spec.neuron_target,
                "NEURON_LOGICAL_NC_CONFIG": spec.logical_nc_config,
                "NEURON_CC_FLAGS": spec.neuron_cc_flags,
                "NEURON_RT_NUM_CORES": os.environ.get("NEURON_RT_NUM_CORES", ""),
                "NEURON_COMPILE_CACHE_URL": cache_dir,
                "NEURON_FRAMEWORK_DEBUG": "1",
                **RUNTIME_INSPECT_ENV,
                "NEURON_RT_INSPECT_OUTPUT_DIR": inspect_dir,
            },
            "created_at_utc": _utcnow(),
        }
        atomic_write_json(manifest_path, manifest)
        append_jsonl_locked(index_path, {
            "spec_id": spec_id, "operator": operator, "case_label": case_label,
            "dtype": dtype_str, "identity_source": identity_source,
            "manifest_path": manifest_path,
            "nki_neff_sha256s": _executed_shas(targets.get("nki")),
            "torch_neff_sha256s": _executed_shas(targets.get("torch")),
            "nki_ms": (targets.get("nki", {}).get("stats") or {}).get("mean"),
            "torch_ms": (targets.get("torch", {}).get("stats") or {}).get("mean"),
            "created_at_utc": manifest["created_at_utc"],
        })

    # ---- map to the engine's fields -----------------------------------------
    audit = {"spec_id": spec_id, "manifest_path": manifest_path,
             "identity_source": identity_source, "identity_status": "validated"}

    torch_rec = targets.get("torch")
    torch_err = ""
    if torch_rec is None:
        torch_stats = None
        torch_err = "profile worker produced no torch baseline"
    elif not torch_rec.get("verify_ok") or not torch_rec.get("stats"):
        torch_stats = None
        torch_err = (f"torch-on-Neuron verification failed: {torch_rec.get('verify_error')}"
                     if torch_rec.get("verify_ok") is False else "torch baseline not timed")
    else:
        torch_stats = dict(torch_rec["stats"], **audit,
                           neff_sha256s=_executed_shas(torch_rec))
    torch_ms = torch_stats["mean"] if torch_stats else float("nan")

    nki_rec = targets.get("nki")
    if not nki_enabled:
        nki_ok, nki_err, nki_stats = False, \
            "NKI not available (no impl or Neuron SDK not installed)", None
    elif nki_rec is None:
        nki_ok, nki_err, nki_stats = False, "NKI profile worker produced no NKI result", None
    elif nki_rec.get("verify_ok") is False:
        nki_ok, nki_err, nki_stats = False, \
            f"verification failed: {nki_rec.get('verify_error')}", None
    elif nki_rec.get("verify_ok") is None:
        # Explicit $NKI_NEFF_PATH override: the file was validated and timed,
        # but it was never executed against this case's inputs, so it must
        # not be reported as a verified result. The latency stays available
        # in nki_stats / the manifest for audit only.
        nki_ok = False
        nki_err = ("explicit NKI_NEFF_PATH override: latency measured but "
                   "correctness NOT verified for this case (see manifest)")
        nki_stats = dict(nki_rec["stats"], **audit,
                         neff_sha256s=[a["neff_sha256"] for a in nki_rec["artifacts"]],
                         verified=False)
    else:
        nki_ok, nki_err = True, ""
        nki_stats = dict(nki_rec["stats"], **audit,
                         neff_sha256s=_executed_shas(nki_rec))
    nki_ms = nki_stats["mean"] if (nki_stats and nki_ok) else float("nan")

    return {
        "torch_stats": torch_stats, "torch_ms": torch_ms, "torch_err": torch_err,
        "nki_stats": nki_stats, "nki_ms": nki_ms,
        "nki_ok": nki_ok, "nki_err": nki_err, "nki_cfg": nki_cfg,
        "spec_id": spec_id, "manifest_path": manifest_path,
        "identity_source": identity_source,
    }
