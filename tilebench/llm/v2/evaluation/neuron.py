"""Trainium2 (native PyTorch Neuron + NKI 0.7) evaluation for the LLM protocol.

Timing adapter `neuron_native_trace` (study.yaml support_matrix.Trn2). The candidate is evaluated by
`evaluation.neuron_worker`, which reuses the per-case logic of `evaluation.worker.run_case` unchanged
(interface reads, three numerical checks, 1 warmup + 3 timed launches, suite rules) and supplies only the
device-specific pieces defined here:

- device: `torch.device("neuron")` of the native torch-neuronx stack (tilebench.core.neuron_native: same launch
  settings as the formal Trn2 benchmark: NEURON_LOGICAL_NC_CONFIG / --lnc from the host, --target trn2,
  NEURON_RT_NUM_CORES=1, i.e. ONE logical NeuronCore = 2 physical NeuronCore-v3 under LNC=2). No torch-xla, no
  torch.compile (the whole suite runs inside neuron_native.torch_compile_disabled()).
- inputs: generated on the CPU with the case seed (the operator's generator), then moved to the device; the
  reference is the operator's impl_torch.run on CPU copies of the same inputs (the formal Trn2 benchmark's
  definition: candidate and PyTorch eager are both verified against the CPU reference), compared on the CPU
  with the job's tolerance.
- timing (`measure_neuron`): `warmup` synchronized calls, then `repeat` calls, each inside its own
  torch.profiler session with NeuronConfig(RUNTIME); the latency of one call is the device BUSY SUM of the
  `nc_exec_running` executions of that call (all executions of one run(); the per-physical-core copies of one
  execution are merged by (model_id, flow_id) first, so LNC=2 parallel execution is never double counted;
  neuron_native._device_call + neuron_intervals.aggregate). The first execution (compile) is outside it (run_case
  stage 1). A call with no device execution, or different executables between timed calls, makes the sample
  unavailable (timing_error); host wall-clock is never substituted. No cache flush exists on Neuron (inputs live
  in HBM; SBUF/PSUM are software managed): `flush_before_each_launch` is False and recorded; no CUDA graph
  (timing_execution_mode "eager").
- host fallback: every candidate call runs between torch_neuronx.clear_op_tracking() and
  get_fallback_ops()/get_executed_ops(); a CPU fallback op or a device->host copy (neuron_to_cpu) inside run()
  is a contract_violation (the tensor computation left the device).
- isolation (`NeuronSubprocessEvaluator`): bwrap when it works; on hosts where unprivileged user namespaces are
  blocked (Ubuntu 24.04 AppArmor `kernel.apparmor_restrict_unprivileged_userns=1`, no sudo) a docker container
  with the same allowlist (system dirs, runtime prefix and the venv's editable package roots read-only; the
  repository read-only with the same data masks; only the task's impl_torch.py re-bound; /home, /root, /tmp,
  /var/tmp private tmpfs; no network; own pid namespace; all capabilities dropped, no-new-privileges; uid of the
  caller; /dev/neuron* passed through; scrubbed environment; per-candidate compile caches inside the sandbox).
  The isolation probe checks the same restricted paths as the bwrap probe plus the manual impl_nki.py."""
from __future__ import annotations

import collections
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tilebench.paths import REPO_ROOT

from tilebench.llm.v2.evaluation import launcher as L

ADAPTER = "neuron_native_trace"
WORKER_MODULE = "tilebench.llm.v2.evaluation.neuron_worker"
# sources that decide verdicts/timing on Trn2, on top of the shared EVALUATION_SOURCES (fingerprint.py)
NEURON_EVALUATION_SOURCES = ("evaluation/neuron.py", "evaluation/neuron_worker.py")
NEURON_CORE_SOURCES = ("neuron_native.py", "neuron_intervals.py", "nki_timer.py")
NEURON_PACKAGES = ("torch", "torch-neuronx", "nki", "neuronx-cc")
DOCKER_IMAGE = "tilebench-sandbox-empty:1"     # an empty image: every binary comes from the read-only host binds
CONTAINER_MEMORY = "100g"                      # bound a runaway candidate/compile below the host's 125 GiB


# --------------------------------------------------------------------------
# host / device facts
# --------------------------------------------------------------------------

def neuron_present() -> bool:
    return bool(glob.glob("/dev/neuron*"))


def neuron_host_arch() -> str | None:
    """The study's arch label of a Trainium2 host ('trainium2'), None elsewhere (detect_arch() is CUDA/HIP only)."""
    if not neuron_present():
        return None
    try:
        from tilebench.core.neuron_native import hardware_identity
        ident = hardware_identity() or {}
    except Exception:  # noqa: BLE001
        ident = {}
    return "trainium2" if str(ident.get("instance_type", "")).startswith("trn2") else None


def host_arch() -> str | None:
    from tilebench import hardware
    return hardware.detect_arch() or neuron_host_arch()


def editable_roots(prefix: str | None = None) -> list[str]:
    """Source roots of editable installs in the runtime prefix (e.g. torch-neuronx from a local checkout): outside
    the prefix, so they must be bound read-only into the sandbox for the worker to import them."""
    prefix = prefix or sys.prefix
    roots: set[str] = set()
    for finder in glob.glob(os.path.join(prefix, "lib", "python*", "site-packages", "__editable___*_finder.py")):
        try:
            text = Path(finder).read_text()
        except OSError:
            continue
        start = text.find("MAPPING")
        if start < 0:
            continue
        line = text[start:].split("\n", 1)[0]
        try:
            mapping = json.loads(line.split("=", 1)[1].strip().replace("'", '"'))
        except (ValueError, IndexError):
            continue
        for path in mapping.values():
            p = Path(path)
            # bind the checkout root that contains the package (e.g. .../torch_neuron_eager)
            roots.add(str(p.parent if p.name in ("torch_neuronx",) else p))
    out = []
    for r in sorted(roots):
        if not r.startswith(prefix.rstrip("/") + "/") and os.path.isdir(r):
            out.append(r)
    # keep only the outermost roots
    return [r for r in out if not any(r != o and r.startswith(o.rstrip("/") + "/") for o in out)]


# --------------------------------------------------------------------------
# host-fallback detection (torch_neuronx op tracking)
# --------------------------------------------------------------------------

def _fallback_violation_class():
    from tilebench.llm.v2.evaluation.worker import PrecisionTampered

    class CpuFallbackViolation(PrecisionTampered):
        """Raised by a candidate call whose tensor computation left the device. Subclasses the worker's
        PrecisionTampered so that run_case closes the case as contract_violation (suite stop), unchanged."""

        def __init__(self, fallback_ops: list, host_copies: list):
            RuntimeError.__init__(self, "tensor computation of run() left the Neuron device: "
                                        f"CPU fallback ops {fallback_ops}, device->host copies {host_copies} "
                                        "(torch_neuronx op tracking during the candidate call)")
            self.where = "host fallback"
            self.diffs = {"fallback_ops": fallback_ops, "neuron_to_cpu": host_copies}

    return CpuFallbackViolation


def track_host_fallback(fn, tn, log: list | None = None):
    """Wrap a candidate call: torch_neuronx op tracking is cleared before and read after the call."""
    cls = _fallback_violation_class()

    def run(*a, **k):
        tn.clear_op_tracking()
        out = fn(*a, **k)
        fallback = sorted(set(tn.get_fallback_ops()))
        executed = sorted(set(tn.get_executed_ops()))
        host = [o for o in executed if "neuron_to_cpu" in o]
        if log is not None and len(log) < 4:
            log.append({"fallback_ops": fallback, "neuron_to_cpu": host, "executed_ops": executed[:40]})
        if fallback or host:
            raise cls(fallback, host)
        return out
    return run


# --------------------------------------------------------------------------
# numerical checks with a CPU reference
# --------------------------------------------------------------------------

def _is_tensor(x) -> bool:
    import torch
    return isinstance(x, torch.Tensor)


def freeze_cpu(out):
    """Recursive CPU copy of every tensor in an output structure (cloned before any restore)."""
    import torch
    if isinstance(out, torch.Tensor):
        return out.detach().to("cpu").clone()
    if isinstance(out, (tuple, list)):
        return type(out)(freeze_cpu(o) for o in out)
    if isinstance(out, dict):
        return {k: freeze_cpu(v) for k, v in out.items()}
    return out


def run_numerical_checks_neuron(run_candidate, run_reference, make_inputs, *, atol: float, rtol: float,
                                mutable_indices=frozenset(), sync, aliasing_allowed: bool = False):
    """evaluation.anticache.run_numerical_checks for device tensors and a CPU reference: the same three checks
    (fresh_storage, same_address_new_values = refill in place, repeat_same_inputs), the same mutation / aliasing
    rules and outcome records. Inputs are snapshotted, compared and restored through CPU copies; the reference runs
    on CPU copies of the inputs and is compared on the CPU. The repeat check reuses the reference of the identical
    (restored) inputs of the previous check (a deterministic CPU function of the same values)."""
    import torch
    from tilebench.core.verifier import verify
    from tilebench.llm.v2.evaluation.anticache import CheckOutcome, output_aliases, refill_in_place

    out: list = []
    mutable = set(mutable_indices)

    def once(name: str, inputs: tuple, ref=None):
        snap = [freeze_cpu(x) if _is_tensor(x) else x for x in inputs]
        if ref is None:
            ref = freeze_cpu(run_reference(*inputs))
        sync()
        cand_raw = run_candidate(*inputs)
        sync()
        aliases = output_aliases(cand_raw, inputs)
        cand = freeze_cpu(cand_raw)
        cand_raw = None
        mutated = [i for i, (x, s) in enumerate(zip(inputs, snap))
                   if _is_tensor(x) and not torch.equal(freeze_cpu(x), s)]
        ok, msg = verify(cand, ref, atol=atol, rtol=rtol)
        undeclared = [i for i in mutated if i not in mutable]
        if undeclared:
            ok, msg = False, f"candidate mutated input(s) {undeclared} which the contract declares immutable; {msg}"
        if aliases and not aliasing_allowed:
            ok, msg = False, f"candidate output shares storage with input(s) {aliases}; the contract requires a fresh output; {msg}"
        for i in mutated:
            inputs[i].copy_(snap[i])
        sync()
        out.append(CheckOutcome(name=name, ok=ok, message=msg, inputs_mutated=mutated, output_aliases_inputs=aliases))
        return ref

    a = make_inputs()
    once("fresh_storage", a)
    if not out[-1].ok:
        return out
    refill_in_place(a, make_inputs())
    sync()
    ref2 = once("same_address_new_values", a)
    if not out[-1].ok:
        return out
    once("repeat_same_inputs", a, ref=ref2)
    return out


# --------------------------------------------------------------------------
# timing: device busy sum per call (neuron_native)
# --------------------------------------------------------------------------

@dataclass
class NeuronTimingRecord:
    """Field-compatible with evaluation.timing.TimingRecord (run_case reads warmup_runs, timed_runs, samples_ms,
    mean_ms, graph_prep_runs, timing_execution_mode, capture_succeeded, capture_error, note, to_dict)."""
    warmup_runs: int
    timed_runs: int
    samples_ms: list
    mean_ms: float | None
    requested_use_cuda_graph: bool
    effective_use_cuda_graph: bool = False
    capture_succeeded: bool | None = None
    capture_error: str | None = None
    graph_prep_runs: int = 0
    timing_execution_mode: str = "eager"
    flush_before_each_launch: bool = False
    flush_requested: bool = False
    flush_buffer_mb: int | None = None
    scope_names: list = field(default_factory=list)
    note: str | None = None
    profile_path: str | None = None
    measured_quantity: str = "neuron_device_busy_sum_per_call_from_runtime_trace"
    timing_definition: str = ("busy_sum: sum over the device executions of one run() call of each execution's duration; "
                              "per-physical-core copies of one execution merged by (model_id, flow_id) first")
    calls: list = field(default_factory=list)          # per timed call: executions, busy/union/span, model ids
    executables_identical: bool | None = None
    lnc: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _compact_call(agg: dict) -> dict:
    rec = {k: agg.get(k) for k in ("n_executions", "busy_sum_ms", "union_ms", "span_ms", "overlap_between_executions")}
    rec["model_ids"] = agg.get("model_ids")
    return rec


def measure_neuron(f, *, rt, warmup: int = 1, repeat: int = 3, use_cuda_graph: bool = True, flush: bool = True,
                   before_launch=None, proton_output_dir: str | None = None, label: str | None = None,
                   keep_profile: bool = False) -> NeuronTimingRecord:
    """Signature-compatible with evaluation.timing.measure (bind `rt` with functools.partial). f() runs the
    whole candidate on fixed device inputs; before_launch() restores declared-mutable inputs outside the timed
    scope (before the warmup and before every timed call)."""
    from tilebench.core.neuron_native import _device_call

    for _ in range(max(0, warmup)):
        if before_launch is not None:
            before_launch()
        f()
    rt.sync()
    root = Path(proton_output_dir or tempfile.mkdtemp(prefix="neuron_timing_")) / "neuron_profiles"
    root.mkdir(parents=True, exist_ok=True)
    tag = f"{label or 'call'}_{uuid.uuid4().hex[:8]}"
    calls, samples, multisets, problems = [], [], set(), []
    for i in range(max(1, repeat)):
        if before_launch is not None:
            before_launch()
            rt.sync()
        pdir = root / f"{tag}_t{i}"
        agg = _device_call(rt, f, (), {}, pdir, f"{tag}-{i}")
        agg.pop("out", None)
        calls.append(_compact_call(agg))
        multisets.add(tuple(sorted(collections.Counter(agg.get("model_ids") or []).items())))
        if not agg.get("n_executions"):
            problems.append(f"timed call {i + 1} recorded no device execution")
        samples.append(agg.get("busy_sum_ms") or 0.0)
        if not keep_profile:
            shutil.rmtree(pdir, ignore_errors=True)
    identical = len(multisets) == 1
    if not identical:
        problems.append("executables differ between timed calls (compile or a different NEFF inside the timed window)")
    ok = not problems and all(isinstance(s, (int, float)) and s > 0 for s in samples)
    return NeuronTimingRecord(
        warmup_runs=warmup, timed_runs=repeat, samples_ms=samples, mean_ms=(sum(samples) / len(samples)) if ok else None,
        requested_use_cuda_graph=bool(use_cuda_graph), flush_requested=bool(flush),
        scope_names=[f"{tag}_t{i}" for i in range(len(calls))],
        note=None if ok else "device timing unavailable: " + "; ".join(problems or ["a non-positive busy sum"]),
        profile_path=str(root) if keep_profile else None, calls=calls, executables_identical=identical,
        lnc=os.environ.get("NEURON_LOGICAL_NC_CONFIG"))


# --------------------------------------------------------------------------
# isolation: bwrap (if usable) or docker with the same allowlist
# --------------------------------------------------------------------------

def docker_usable() -> tuple[bool, str | None]:
    if not shutil.which("docker"):
        return False, "docker not found"
    try:
        subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, timeout=30, check=True)
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def ensure_sandbox_image() -> str:
    """The empty base image (docker import of an empty tar); its id is part of the isolation record."""
    q = subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE, "--format", "{{.Id}}"], capture_output=True, text=True)
    if q.returncode == 0 and q.stdout.strip():
        return q.stdout.strip()
    with tempfile.TemporaryDirectory() as d:
        tar = Path(d) / "empty.tar"
        subprocess.run(["tar", "cf", str(tar), "--files-from", "/dev/null"], check=True)
        subprocess.run(["docker", "import", str(tar), DOCKER_IMAGE], check=True, capture_output=True)
    return subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE, "--format", "{{.Id}}"], capture_output=True,
                          text=True, check=True).stdout.strip()


def _docker_base(sandbox: Path | None, *, name: str, env_keys: list[str], operator_dir: Path | None = None,
                 repo_root: Path | None = None) -> list[str]:
    """docker run with the bwrap allowlist (launcher._bwrap_base): nothing else of the host file system is
    visible. Environment values are taken from the docker client's environment (`-e KEY`), never written to the
    command line."""
    repo = Path(repo_root or REPO_ROOT).resolve()
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--pids-limit", "4096",
           "--user", f"{os.getuid()}:{os.getgid()}", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
           "--read-only", "--memory", CONTAINER_MEMORY, "--memory-swap", CONTAINER_MEMORY, "--shm-size", "8g",
           "--ipc", "private"]                          # own pid namespace: docker's default
    for d in L.SYSTEM_RO:
        if os.path.isdir(d):
            cmd += ["-v", f"{d}:{d}:ro"]
    for p in L.device_nodes().get("neuron", {}).get("present", []):
        cmd += ["--device", p]
    for t in ("/tmp", "/var/tmp", "/home", "/root"):
        cmd += ["--tmpfs", f"{t}:rw,exec,size=64g,mode=1777" if t in ("/tmp", "/var/tmp") else f"{t}:rw,size=1g,mode=755"]
    prefix = L.runtime_prefix()
    cmd += ["-v", f"{prefix}:{prefix}:ro"]
    for r in editable_roots(prefix):
        cmd += ["-v", f"{r}:{r}:ro"]
    cmd += ["-v", f"{repo}:{repo}:ro"]
    for m in L.REPO_MASKS:
        target = repo / m
        if target.is_dir():
            cmd += ["--tmpfs", f"{target}:rw,size=1m,mode=755"]   # empty, private, discarded (as bwrap's --tmpfs)
        elif target.exists():
            cmd += ["-v", f"{_unreadable_file()}:{target}:ro"]     # e.g. a worktree's .git pointer
    if operator_dir is not None:
        ref = Path(operator_dir) / "impl_torch.py"
        if ref.exists():
            cmd += ["-v", f"{ref}:{ref}:ro"]
    if sandbox is not None:
        cmd += ["-v", f"{sandbox}:{sandbox}:rw", "--workdir", str(sandbox)]
    for k in env_keys:
        cmd += ["-e", k]
    cmd += [DOCKER_IMAGE]
    return cmd


def _unreadable_file() -> str:
    """An empty mode-000 file to mask a repository FILE (bwrap binds /dev/null, which reads as empty but not as
    unreadable). The container has no CAP_DAC_OVERRIDE and the bind is read-only, so it cannot be opened."""
    path = Path(tempfile.gettempdir()) / f"tilebench_neuron_mask_000_{os.getuid()}"
    if not path.exists():
        path.touch()
    os.chmod(path, 0)
    return str(path)


# The probe script of launcher._PROBE with one addition: a "skeleton" check. A directory may contain only the names
# of allowlisted mountpoint parents (the runtime prefix / editable roots / repository may live under the real home).
_NEURON_PROBE = r"""
import json, os, socket, sys
checks = json.loads(sys.argv[1])
out = {}
for name, path in checks["unreadable"].items():
    try:
        if os.path.isdir(path):
            out[name] = {"readable": bool(os.listdir(path)), "path": path}
        else:
            open(path, "rb").read(1); out[name] = {"readable": True, "path": path}
    except Exception as e:
        out[name] = {"readable": False, "path": path, "error": type(e).__name__}
for name, spec in checks["skeleton_only"].items():
    path, allowed = spec["path"], set(spec["allowed"])
    try:
        entries = sorted(os.listdir(path))
        extra = [e for e in entries if e not in allowed]
        out[name] = {"readable": bool(extra), "path": path, "entries": entries, "beyond_skeleton": extra}
    except Exception as e:
        out[name] = {"readable": False, "path": path, "error": type(e).__name__}
for name, path in checks["readable"].items():
    try:
        open(path, "rb").read(1); out[name] = {"readable": True, "path": path}
    except Exception as e:
        out[name] = {"readable": False, "path": path, "error": type(e).__name__}
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2); out["network"] = {"reachable": True}
except Exception as e:
    out["network"] = {"reachable": False, "error": type(e).__name__}
out["secret_env_names"] = [k for k in os.environ if any(m in k.upper() for m in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
print(json.dumps(out))
"""


def _skeleton(home: Path, bound: list[str]) -> list[str]:
    """Names directly under `home` that are parents of a bound path (mountpoint skeleton, no host content)."""
    names = set()
    for b in bound:
        try:
            rel = Path(b).resolve().relative_to(home.resolve())
        except ValueError:
            continue
        if rel.parts:
            names.add(rel.parts[0])
    return sorted(names)


def _docker_kill(name: str) -> None:
    subprocess.run(["docker", "kill", name], capture_output=True, timeout=60)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)


_NEURON_ISOLATION_CACHE: dict[str, dict] = {}


def detect_isolation_neuron(mode: str = "auto") -> dict:
    """mode: auto (bwrap when it works, else docker when usable, else none), bwrap, docker, none."""
    if mode in _NEURON_ISOLATION_CACHE:
        return dict(_NEURON_ISOLATION_CACHE[mode])
    if mode in ("auto", "bwrap"):
        try:
            rep = L.detect_isolation("bwrap" if mode == "bwrap" else "auto")
        except RuntimeError:
            if mode == "bwrap":
                raise
            rep = L.detect_isolation("none")
        if rep["backend"] == "bwrap" or mode == "bwrap":
            _NEURON_ISOLATION_CACHE[mode] = rep
            return dict(rep)
        bwrap_error = rep.get("probe_error")
    else:
        bwrap_error = None
    report = L.detect_isolation("none")
    report["bwrap_error"] = bwrap_error
    if mode in ("auto", "docker"):
        ok, err = docker_usable()
        if ok:
            image = ensure_sandbox_image()
            report.update({"backend": "docker", "home_hidden": True, "network": "none", "pid_namespace": True,
                           "filesystem": "allowlist: system dirs, runtime prefix, editable package roots and repository "
                                         "read-only with data masks; sandbox read-write; /home,/root,/tmp,/var/tmp private tmpfs",
                           "tmp_private": True, "capabilities": "all dropped; no-new-privileges; read-only root",
                           "image": DOCKER_IMAGE, "image_id": image, "memory_limit": CONTAINER_MEMORY,
                           "allowlist": {"system_ro": list(L.SYSTEM_RO), "runtime_prefix": L.runtime_prefix(),
                                         "editable_roots": editable_roots(), "repo_masks": list(L.REPO_MASKS)},
                           "probe_error": None})
        else:
            report["probe_error"] = f"docker unusable: {err}"
    if mode == "docker" and report["backend"] != "docker":
        raise RuntimeError(f"docker isolation required but unavailable: {report['probe_error']}")
    _NEURON_ISOLATION_CACHE[mode] = report
    return dict(report)


def isolation_probe_neuron(operator: str = "vector_add", *, backend: str = "docker", repo_root: Path | None = None,
                           sandbox_root: Path | None = None) -> dict:
    """launcher.isolation_probe for the Trn2 backends: the same restricted/readable paths (plus the operator's
    manual impl_nki.py and the device node), the same sentinel, the same pass rule."""
    from tilebench.paths import operator_dir
    if backend == "bwrap":
        return L.isolation_probe(operator, repo_root=repo_root, sandbox_root=sandbox_root)
    repo = Path(repo_root or REPO_ROOT).resolve()
    root = sandbox_root or (repo / "outputs" / "llm_v2" / "_isolation_probe")
    root.mkdir(parents=True, exist_ok=True)
    sandbox = Path(tempfile.mkdtemp(prefix="probe_", dir=root)).resolve()
    os.chmod(sandbox, 0o755)
    sentinel = root / f"sentinel_{os.getpid()}.txt"
    sentinel.write_text("isolation sentinel: non-sensitive marker\n")
    home = Path(os.path.expanduser("~"))
    opdir = Path(operator_dir(operator))
    checks = {"unreadable": {
                  "manual_dsl_impl": str(opdir / "impl_triton.py"),
                  "manual_nki_impl": str(opdir / "impl_nki.py"),
                  "another_operator_reference": str(opdir.parent / ("softmax" if operator != "softmax" else "vector_add") / "impl_torch.py"),
                  "results_dir": str(repo / "results"), "artifacts_dir": str(repo / "artifacts"), "skills_dir": str(repo / "skills"),
                  "contracts_dir": str(repo / "tilebench/llm/v2/contracts/data"), "sentinel_in_outputs": str(sentinel),
                  "home_bashrc": str(home / ".bashrc"), "home_ssh": str(home / ".ssh"), "home_aws": str(home / ".aws"),
                  "home_claude": str(home / ".claude"), "git_dir": str(repo / ".git"),
                  "sibling_checkout": next((str(d) for d in sorted(repo.parent.iterdir()) if d.is_dir() and d.resolve() != repo), str(repo.parent / "__none__"))},
              "skeleton_only": {"real_home_dir": {"path": str(home), "allowed": _skeleton(home, [L.runtime_prefix(), str(repo), str(sandbox)] + editable_roots())}},
              "readable": {"task_reference": str(opdir / "impl_torch.py"), "runtime_python": sys.executable}}
    probe = _NEURON_PROBE
    name = f"tb-probe-{uuid.uuid4().hex[:12]}"
    env = L.scrubbed_env(sandbox)
    argv = _docker_base(sandbox, name=name, env_keys=sorted(env), operator_dir=opdir, repo_root=repo) + \
        [sys.executable, "-c", probe, json.dumps(checks)]
    try:
        proc = subprocess.run(argv, cwd=str(sandbox), env=env, capture_output=True, text=True, timeout=180)
        rep = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {"error": proc.stderr[-2000:]}
    except Exception as e:  # noqa: BLE001
        rep = {"error": f"{type(e).__name__}: {e}"}
        _docker_kill(name)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
        try:
            sentinel.unlink()
        except OSError:
            pass
    # the device node must be visible (the evaluation needs it); checked with a second, trivial run
    dev = subprocess.run(_docker_base(None, name=f"{name}-dev", env_keys=[], repo_root=repo) +
                         ["/bin/sh", "-c", "ls /dev | grep -c '^neuron'"], capture_output=True, text=True, timeout=120)
    nodes_inside = int(dev.stdout.strip() or 0) if dev.returncode == 0 else 0
    ok = "error" not in rep and not rep.get("network", {}).get("reachable", True) and not rep.get("secret_env_names") \
        and all(not rep[k]["readable"] for k in checks["unreadable"]) and all(not rep[k]["readable"] for k in checks["skeleton_only"]) \
        and all(rep[k]["readable"] for k in checks["readable"]) and nodes_inside > 0
    return {"ok": ok, "backend": "docker", "operator": operator, "results": rep, "neuron_device_nodes_inside": nodes_inside,
            "allowlist": {"system_ro": list(L.SYSTEM_RO), "runtime_prefix": L.runtime_prefix(), "editable_roots": editable_roots(),
                          "repo_root": str(repo), "repo_masks": list(L.REPO_MASKS), "task_reference": str(opdir / "impl_torch.py")},
            "device_nodes": L.device_nodes()}


# --------------------------------------------------------------------------
# evaluator
# --------------------------------------------------------------------------

def neuron_worker_env(sandbox: Path) -> dict:
    """launcher.scrubbed_env plus the per-candidate Neuron / torch caches inside the sandbox (set before the worker
    starts: `import torch` auto-loads torch_neuronx through its backend entry point)."""
    env = L.scrubbed_env(sandbox)
    cache = sandbox / "neuron_cache"
    env["TILEBENCH_NEURON_CACHE"] = str(cache)
    for var, sub in (("TORCH_NEURONX_NEFF_CACHE_DIR", "neff"), ("TORCH_NEURONX_HLO_CACHE_DIR", "hlo"),
                     ("TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "local"), ("NKI_TRACE_CACHE_URL", "nki_trace"),
                     ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TORCH_NEURONX_DEBUG_DIR", "debug")):
        env[var] = str(cache / sub)
    env["NEURON_COMPILE_CACHE_URL"] = str(cache / "neuronx_cc")
    return env


@dataclass
class NeuronSubprocessEvaluator(L.SubprocessEvaluator):
    """launcher.SubprocessEvaluator for Trn2: the neuron worker module, bwrap or docker isolation, the same
    device lock, wall-clock limit, partial-result classification, cgroup record and evidence archive."""

    def __post_init__(self):
        self.report = detect_isolation_neuron(self.isolation)
        self.report["timing_adapter"] = ADAPTER

    def evaluate(self, source_path: Path, job, round_index: int, attempt: int, *, archive_dir: Path | None = None) -> dict:
        from tilebench.paths import operator_dir
        root = self.sandbox_root or Path(tempfile.gettempdir()) / "tilebench_llm_v2_sandbox"
        root.mkdir(parents=True, exist_ok=True)
        sandbox = Path(tempfile.mkdtemp(prefix=f"{job.operator}_{round_index}_{attempt}_", dir=root)).resolve()
        os.chmod(sandbox, 0o755)
        seed = int(time.time_ns() % (2**31))
        backend = self.report["backend"]
        name = f"tb-eval-{uuid.uuid4().hex[:12]}"
        try:
            copied = sandbox / source_path.name
            shutil.copyfile(source_path, copied)
            wjob = job.worker_job(source_path=str(copied), sandbox_dir=str(sandbox), seed=seed,
                                  round_index=round_index, attempt=attempt)
            (sandbox / "job.json").write_text(json.dumps(wjob, indent=1))
            out = sandbox / "result.json"
            env = neuron_worker_env(sandbox)
            argv = [sys.executable, "-m", WORKER_MODULE, "--job", str(sandbox / "job.json"), "--out", str(out)]
            if backend == "bwrap":
                extra = []
                for r in editable_roots():
                    extra += ["--ro-bind", r, r]
                base = L._bwrap_base(sandbox, operator_dir=Path(operator_dir(job.operator)))
                cut = base.index("--unshare-net")
                argv = base[:cut] + extra + base[cut:] + argv
            elif backend == "docker":
                argv = _docker_base(sandbox, name=name, env_keys=sorted(env), operator_dir=Path(operator_dir(job.operator))) + argv
            t_lock = time.time()
            with L.device_lock(self.device, root=self.lock_root):
                lock_wait = time.time() - t_lock
                mem_before = L.cgroup_memory_events()
                t0 = time.time()
                try:
                    proc = subprocess.run(argv, cwd=str(sandbox), env=env, timeout=self.timeout_s, capture_output=True, text=True)
                    rc, stdout, stderr, timed_out = proc.returncode, proc.stdout, proc.stderr, False
                except subprocess.TimeoutExpired as e:
                    rc, timed_out = None, True
                    stdout = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
                    stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
                    if backend == "docker":
                        _docker_kill(name)
                mem_after = L.cgroup_memory_events()
            wall = time.time() - t0
            (sandbox / "worker_stdout.txt").write_text(stdout or "")
            (sandbox / "worker_stderr.txt").write_text(stderr or "")
            if timed_out or not out.exists():
                res = L.classify_no_result(sandbox, timed_out=timed_out, timeout_s=self.timeout_s, rc=rc, stderr=stderr or "",
                                           job_cases=getattr(job, "cases", None))
            else:
                res = json.loads(out.read_text())
            res["resources"] = {"cgroup_before": mem_before, "cgroup_after": mem_after,
                                "cgroup_oom_kill_delta": (mem_after.get("events.oom_kill", 0) - mem_before.get("events.oom_kill", 0))
                                if mem_before and mem_after else None,
                                "container_memory_limit": CONTAINER_MEMORY if backend == "docker" else None,
                                "container_oom_suspected": bool(backend == "docker" and rc == 137)}
            res["seed"] = seed
            res["worker_rc"] = rc
            res["worker_wall_s"] = wall
            res["device_lock_wait_s"] = lock_wait
            res["isolation"] = dict(self.report)
            res["sandbox_dir"] = str(sandbox)
            if archive_dir is not None:
                (sandbox / "result.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
                idx = L.archive_sandbox(sandbox, archive_dir, extra={"round": round_index, "attempt": attempt, "seed": seed,
                                                                     "isolation": dict(self.report)})
                res["archive"] = {"dir": str(archive_dir), "files": sorted(idx["files"])}
            return res
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)


def make_evaluator(device: str, *, study: dict | None = None, **kw):
    """The device's evaluator: NeuronSubprocessEvaluator when its timing adapter is neuron_native_trace, the
    shared SubprocessEvaluator otherwise (unchanged for every GPU device)."""
    if study is None:
        from tilebench.llm.v2.manifests import schema as ms
        study = ms.load_study()
    if study["support_matrix"][device]["timing_adapter"] == ADAPTER:
        return NeuronSubprocessEvaluator(device=device, **kw)
    return L.SubprocessEvaluator(device=device, **kw)


def formal_isolation_backends(device: str, study: dict | None = None) -> tuple[str, ...]:
    """Backends a formal campaign accepts: bwrap everywhere; on the Neuron adapter also docker with the same
    allowlist and a passing probe (bwrap cannot create user namespaces on the Trn2 host without root)."""
    if study is None:
        from tilebench.llm.v2.manifests import schema as ms
        study = ms.load_study()
    if study["support_matrix"][device]["timing_adapter"] == ADAPTER:
        return ("bwrap", "docker")
    return ("bwrap",)
