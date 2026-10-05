"""Launches the isolated worker: bubblewrap sandbox when available, scrubbed
environment, private sandbox directory, redirected compile caches, the
per-device lock, a wall-clock timeout, and archiving of the run's evidence
before the sandbox is deleted. Also provides the mock evaluator used by
tests and dry runs.

Isolation layers, reported in every result under `isolation` (never
claimed when not applied):

bwrap (user namespaces; probed once per process)
  - the whole host filesystem is bound READ-ONLY; only the sandbox
    directory is writable
  - /home is replaced by an empty tmpfs: the user's dotfiles (shell rc files
    that export API keys) are not visible to generated code
  - /tmp and /var/tmp are private tmpfs
  - no network namespace membership (--unshare-net), new pid namespace
  - GPU device nodes (/dev/nvidia*) are bound in; the worker needs them
environment (always)
  - every variable whose name contains KEY / TOKEN / SECRET / PASSWORD /
    HUGGING_FACE / HF_ is dropped before the worker starts
  - HOME, TMPDIR, XDG_CACHE_HOME and the Triton / cuTile / TileLang cache
    directories point into the sandbox
  - PYTHONSTARTUP is dropped
fallback (bwrap unavailable or refused)
  - the plain subprocess with the scrubbed environment; `isolation.backend`
    is "none", `home_hidden` is false and the campaign record says so.

Not enforced in either mode: seccomp, memory cgroups, GPU partitioning."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from tilebench.paths import REPO_ROOT

from tilebench.llm.v2.orchestration.locks import device_lock

SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "HUGGING_FACE", "HF_")
_ISOLATION_CACHE: dict[str, dict] = {}


def scrubbed_env(sandbox: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in SECRET_MARKERS)}
    home = sandbox / "home"
    tmp = sandbox / "tmp"
    for d in (home, tmp, sandbox / "triton_cache", sandbox / "cutile_cache", sandbox / "tilelang_cache"):
        d.mkdir(parents=True, exist_ok=True)
    env["TILEBENCH_V2_SANDBOX"] = "1"
    env["HOME"] = str(home)
    env["TMPDIR"] = str(tmp)
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    env["TRITON_CACHE_DIR"] = str(sandbox / "triton_cache")
    env["CUDA_TILE_CACHE_DIR"] = str(sandbox / "cutile_cache")
    env["TILELANG_CACHE_DIR"] = str(sandbox / "tilelang_cache")
    env["PYTHONPATH"] = str(REPO_ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("PYTHONSTARTUP", None)
    return env


def _bwrap_base(sandbox: Path | None) -> list[str]:
    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev"]
    for p in sorted(glob.glob("/dev/nvidia*")):
        cmd += ["--dev-bind", p, p]
    cmd += ["--proc", "/proc", "--tmpfs", "/home", "--tmpfs", "/tmp", "--tmpfs", "/var/tmp"]
    if sandbox is not None:
        cmd += ["--bind", str(sandbox), str(sandbox), "--chdir", str(sandbox)]
    cmd += ["--unshare-net", "--unshare-pid", "--die-with-parent", "--new-session", "--"]
    return cmd


def detect_isolation(mode: str = "auto") -> dict:
    """mode: auto (bwrap when it works, else none), bwrap (required), none."""
    if mode in _ISOLATION_CACHE:
        return dict(_ISOLATION_CACHE[mode])
    report = {"backend": "none", "home_hidden": False, "network": "host", "pid_namespace": False,
              "filesystem": "host (read-write as the user)", "tmp_private": False, "probe_error": None,
              "env_secrets_scrubbed": True}
    if mode == "none":
        _ISOLATION_CACHE[mode] = report
        return dict(report)
    if not shutil.which("bwrap"):
        report["probe_error"] = "bwrap not found"
    else:
        try:
            subprocess.run(_bwrap_base(None) + ["/bin/true"], capture_output=True, timeout=60, check=True)
            report.update({"backend": "bwrap", "home_hidden": True, "network": "unshared", "pid_namespace": True,
                           "filesystem": "host read-only; sandbox read-write; /home,/tmp,/var/tmp private tmpfs",
                           "tmp_private": True})
        except Exception as e:  # noqa: BLE001
            err = getattr(e, "stderr", b"")
            report["probe_error"] = f"{type(e).__name__}: {e} {err.decode(errors='replace') if isinstance(err, bytes) else err}".strip()
    if mode == "bwrap" and report["backend"] != "bwrap":
        raise RuntimeError(f"bwrap isolation required but unavailable: {report['probe_error']}")
    _ISOLATION_CACHE[mode] = report
    return dict(report)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CANDIDATE_PHASES = ("candidate_loaded", "first_execution_done", "numerical_checks_done", "timing_started", "timing_done")


def classify_no_result(sandbox: Path, *, timed_out: bool, timeout_s: int, rc, stderr: str) -> dict:
    """The worker produced no result.json. The worker's progress marker says
    how far it got: a hang or crash after the candidate was loaded is the
    candidate's ordinary failure (`runtime_error`, round consumed, no
    repair); before that it is `infrastructure_incomplete`."""
    phase = None
    try:
        phase = json.loads((sandbox / "progress.json").read_text()).get("phase")
    except (OSError, ValueError):
        pass
    base = {"stages": {}, "config": None, "latency_ms_mean": None, "latency_ms_samples": None, "timing": None,
            "worker_phase": phase, "timed_out": timed_out}
    tail = stderr[-4000:]
    if phase in CANDIDATE_PHASES:
        what = (f"candidate exceeded the evaluation wall-clock limit of {timeout_s}s" if timed_out
                else f"evaluation process ended (exit {rc}) without a result")
        return {**base, "status": "runtime_error",
                "diagnostic": f"{what} during phase {phase} (import/compile, verification or timing of the generated file)\n{tail}".rstrip()}
    what = f"worker exceeded {timeout_s}s" if timed_out else f"worker exited {rc} without a result"
    return {**base, "status": "infrastructure_incomplete", "diagnostic": f"{what} before the candidate was loaded (phase {phase})\n{tail}".rstrip()}


def archive_sandbox(sandbox: Path, archive_dir: Path, extra: dict | None = None) -> dict:
    """Copy the evidence of one evaluation (job, result, stdout/stderr, the
    candidate file, the retained Proton profile) with sha256s; compile
    caches and other large files stay behind and are deleted with the
    sandbox."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"files": {}, "extra": extra or {}}
    names = ["job.json", "result.json", "worker_stdout.txt", "worker_stderr.txt", "progress.json"]
    for p in sorted(sandbox.glob("tilebench_proton_*")):
        names.append(p.name)
    for p in sorted(sandbox.glob("impl_*.py")):
        names.append(p.name)
    for name in names:
        src = sandbox / name
        if src.exists() and src.is_file():
            dst = archive_dir / name
            shutil.copyfile(src, dst)
            index["files"][name] = {"sha256": _sha256(dst), "bytes": dst.stat().st_size}
    (archive_dir / "ARCHIVE_INDEX.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    return index


@dataclass
class SubprocessEvaluator:
    device: str
    timeout_s: int = 1800
    sandbox_root: Path | None = None
    isolation: str = "auto"           # auto | bwrap | none
    lock_root: Path | None = None
    report: dict = field(default_factory=dict)

    def __post_init__(self):
        self.report = detect_isolation(self.isolation)

    def evaluate(self, source_path: Path, job, round_index: int, attempt: int, *, archive_dir: Path | None = None) -> dict:
        root = self.sandbox_root or Path(tempfile.gettempdir()) / "tilebench_llm_v2_sandbox"
        root.mkdir(parents=True, exist_ok=True)
        sandbox = Path(tempfile.mkdtemp(prefix=f"{job.operator}_{round_index}_{attempt}_", dir=root)).resolve()
        seed = int(time.time_ns() % (2**31))
        try:
            copied = sandbox / source_path.name
            shutil.copyfile(source_path, copied)
            wjob = job.worker_job(source_path=str(copied), sandbox_dir=str(sandbox), seed=seed,
                                  round_index=round_index, attempt=attempt)
            (sandbox / "job.json").write_text(json.dumps(wjob, indent=1))
            out = sandbox / "result.json"
            argv = [sys.executable, "-m", "tilebench.llm.v2.evaluation.worker", "--job", str(sandbox / "job.json"),
                    "--out", str(out)]
            if self.report["backend"] == "bwrap":
                argv = _bwrap_base(sandbox) + argv
            env = scrubbed_env(sandbox)
            t_lock = time.time()
            with device_lock(self.device, root=self.lock_root):
                lock_wait = time.time() - t_lock
                t0 = time.time()
                try:
                    proc = subprocess.run(argv, cwd=str(sandbox), env=env, timeout=self.timeout_s,
                                          capture_output=True, text=True)
                    rc, stdout, stderr, timed_out = proc.returncode, proc.stdout, proc.stderr, False
                except subprocess.TimeoutExpired as e:
                    rc, timed_out = None, True
                    stdout = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
                    stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            wall = time.time() - t0
            (sandbox / "worker_stdout.txt").write_text(stdout or "")
            (sandbox / "worker_stderr.txt").write_text(stderr or "")
            if timed_out or not out.exists():
                res = classify_no_result(sandbox, timed_out=timed_out, timeout_s=self.timeout_s, rc=rc, stderr=stderr or "")
            else:
                res = json.loads(out.read_text())
            res["seed"] = seed
            res["worker_rc"] = rc
            res["worker_wall_s"] = wall                 # worker process only (the timeout applies to this)
            res["device_lock_wait_s"] = lock_wait
            res["isolation"] = dict(self.report)
            res["sandbox_dir"] = str(sandbox)
            if archive_dir is not None:
                (sandbox / "result.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
                idx = archive_sandbox(sandbox, archive_dir, extra={"round": round_index, "attempt": attempt, "seed": seed,
                                                                   "isolation": dict(self.report)})
                res["archive"] = {"dir": str(archive_dir), "files": sorted(idx["files"])}
            return res
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)


@dataclass
class MockEvaluator:
    """Scripted outcomes keyed by the source text's first line marker, e.g.
    '# MOCK: valid 1.5' -> valid with samples around 1.5 ms,
    '# MOCK: numerical_error', '# MOCK: compile_error', '# MOCK: infrastructure',
    '# MOCK: contract_violation', '# MOCK: interface_error'."""
    calls: list = None

    def evaluate(self, source_path: Path, job, round_index: int, attempt: int, *, archive_dir: Path | None = None) -> dict:
        self.calls = self.calls if self.calls is not None else []
        op = job.operator if hasattr(job, "operator") else job["operator"]
        self.calls.append((op, round_index, attempt))
        text = source_path.read_text()
        first = text.splitlines()[0] if text else ""
        parts = first.replace("# MOCK:", "").split()
        kind = parts[0] if parts else "valid"
        if kind == "valid":
            base = float(parts[1]) if len(parts) > 1 else 1.0
            samples = [base * 0.99, base, base * 1.01]
            return {"status": "valid", "latency_ms_mean": sum(samples) / 3, "latency_ms_samples": samples,
                    "config": {"BLOCK": 128}, "timing": {"timing_execution_mode": "graph", "capture_succeeded": True},
                    "timing_mode_differs": False, "isolation": {"backend": "mock"}}
        if kind == "infrastructure":
            return {"status": "infrastructure_incomplete", "diagnostic": "mock infrastructure failure"}
        if kind == "contract_violation":
            return {"status": "contract_violation", "diagnostic": "contract_violation: mock execution evidence"}
        return {"status": kind, "diagnostic": f"mock {kind} diagnostic\nroofline 55% should be scrubbed"}
