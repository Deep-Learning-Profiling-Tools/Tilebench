"""Launches the isolated worker: bubblewrap sandbox when available, scrubbed
environment, private sandbox directory, redirected compile caches, the
per-device lock, a wall-clock timeout, and archiving of the run's evidence
before the sandbox is deleted. Also provides the mock evaluator used by
tests and dry runs.

Isolation layers, reported in every result under `isolation` (never
claimed when not applied):

bwrap (user namespaces; probed once per process)
  - bounded allowlist, nothing else is bound: system directories
    (/usr, /lib, /lib64, /bin, /sbin, /etc, /opt, /sys) read-only, the python
    runtime prefix read-only, the repository root read-only with empty tmpfs
    masks over tilebench/benchmarks (then only the task's impl_torch.py is
    re-bound), results, artifacts, outputs, skills, docs, tests, problems,
    contracts/data, .git; the sandbox directory is the only writable path
  - /home and /root are empty tmpfs; the user's real home is not bound
    wherever it lives (the probe checks the resolved home path too)
  - /tmp and /var/tmp are private tmpfs
  - no network namespace membership (--unshare-net), new pid namespace
  - GPU device nodes are bound in per vendor (NVIDIA /dev/nvidia* verified on
    B200; AMD /dev/kfd,/dev/dri and Neuron /dev/neuron* described, pending)
  - `isolation_probe()` verifies the restricted paths inside a fresh sandbox
    with a non-sensitive sentinel; formal preflight requires it to pass
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


# What the sandboxed worker may see. Nothing outside this list is bound: not
# the user's home (wherever it is), not other checkouts, results, artifacts,
# skills, docs, contracts or the manual DSL implementations of any operator.
SYSTEM_RO = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/opt", "/sys")
# Repository paths masked with an empty tmpfs after the read-only bind of the
# repository root; the task's own impl_torch.py is re-bound afterwards.
REPO_MASKS = ("tilebench/benchmarks", "results", "artifacts", "outputs", "skills", "docs", "tests",
              "tilebench/problems", "tilebench/llm/v2/contracts/data", ".git", ".claude", "archive")
GPU_DEVICE_GLOBS = {"nvidia": ("/dev/nvidia*",), "amd": ("/dev/kfd", "/dev/dri"), "neuron": ("/dev/neuron*",)}


def runtime_prefix() -> str:
    return sys.prefix


def device_nodes() -> dict:
    """GPU device nodes bound into the sandbox, per vendor, with what is
    actually present on this host. Only the NVIDIA pattern has been
    exercised (B200); AMD (/dev/kfd, /dev/dri) and Neuron (/dev/neuron*)
    are described but remain `pending` until a run on such a host."""
    out = {}
    for vendor, globs in GPU_DEVICE_GLOBS.items():
        found = []
        for g in globs:
            found += sorted(glob.glob(g))
        out[vendor] = {"globs": list(globs), "present": found,
                       "verified": "B200 validation campaign 2026-10-05" if vendor == "nvidia" and found else "pending"}
    return out


def _bwrap_base(sandbox: Path | None, *, operator_dir: Path | None = None, repo_root: Path | None = None) -> list[str]:
    repo = Path(repo_root or REPO_ROOT).resolve()
    cmd = ["bwrap"]
    for d in SYSTEM_RO:
        if os.path.isdir(d):
            cmd += ["--ro-bind", d, d]
    cmd += ["--dev", "/dev"]
    for vendor in device_nodes().values():
        for p in vendor["present"]:
            cmd += ["--dev-bind", p, p]
    cmd += ["--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/var/tmp", "--tmpfs", "/home", "--tmpfs", "/root"]
    prefix = runtime_prefix()
    cmd += ["--ro-bind", prefix, prefix]
    cmd += ["--ro-bind", str(repo), str(repo)]
    for m in REPO_MASKS:
        target = repo / m
        if target.is_dir():
            cmd += ["--tmpfs", str(target)]             # empty directory in place of the data
        elif target.exists():
            cmd += ["--ro-bind", "/dev/null", str(target)]   # a file (e.g. a worktree's .git pointer) masked by /dev/null
    if operator_dir is not None:
        ref = Path(operator_dir) / "impl_torch.py"
        if ref.exists():
            cmd += ["--ro-bind", str(ref), str(ref)]
    if sandbox is not None:
        cmd += ["--bind", str(sandbox), str(sandbox), "--chdir", str(sandbox)]
    cmd += ["--unshare-net", "--unshare-pid", "--die-with-parent", "--new-session", "--"]
    return cmd


_PROBE = r"""
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


def isolation_probe(operator: str = "vector_add", *, repo_root: Path | None = None, sandbox_root: Path | None = None) -> dict:
    """Runs a small python inside a fresh sandbox (no GPU use) and reports
    what the candidate could read. Uses a sentinel file with non-sensitive
    text; never reads or prints a real key. `ok` is True only when every
    restricted path is unreadable, the task reference is readable and the
    network is unreachable."""
    from tilebench.paths import operator_dir
    repo = Path(repo_root or REPO_ROOT).resolve()
    root = sandbox_root or (repo / "outputs" / "llm_v2" / "_isolation_probe")
    root.mkdir(parents=True, exist_ok=True)
    sandbox = Path(tempfile.mkdtemp(prefix="probe_", dir=root)).resolve()
    sentinel = root / f"sentinel_{os.getpid()}.txt"
    sentinel.write_text("isolation sentinel: non-sensitive marker\n")
    home = Path(os.path.expanduser("~"))
    opdir = Path(operator_dir(operator))
    checks = {"unreadable": {
                  "manual_dsl_impl": str(opdir / "impl_triton.py"),
                  "another_operator_reference": str(opdir.parent / ("softmax" if operator != "softmax" else "vector_add") / "impl_torch.py"),
                  "results_dir": str(repo / "results"), "artifacts_dir": str(repo / "artifacts"), "skills_dir": str(repo / "skills"),
                  "contracts_dir": str(repo / "tilebench/llm/v2/contracts/data"), "sentinel_in_outputs": str(sentinel),
                  "real_home_dir": str(home), "home_bashrc": str(home / ".bashrc"), "git_dir": str(repo / ".git"),
                  "sibling_checkout": next((str(d) for d in sorted(repo.parent.iterdir()) if d.is_dir() and d.resolve() != repo), str(repo.parent / "__none__"))},
              "readable": {"task_reference": str(opdir / "impl_torch.py"), "runtime_python": sys.executable}}
    argv = _bwrap_base(sandbox, operator_dir=opdir, repo_root=repo) + [sys.executable, "-c", _PROBE, json.dumps(checks)]
    try:
        proc = subprocess.run(argv, cwd=str(sandbox), env=scrubbed_env(sandbox), capture_output=True, text=True, timeout=120)
        rep = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {"error": proc.stderr[-2000:]}
    except Exception as e:  # noqa: BLE001
        rep = {"error": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
        try:
            sentinel.unlink()
        except OSError:
            pass
    ok = "error" not in rep and not rep.get("network", {}).get("reachable", True) and not rep.get("secret_env_names") \
        and all(not rep[k]["readable"] for k in checks["unreadable"]) and all(rep[k]["readable"] for k in checks["readable"])
    return {"ok": ok, "backend": "bwrap", "operator": operator, "results": rep,
            "allowlist": {"system_ro": list(SYSTEM_RO), "runtime_prefix": runtime_prefix(), "repo_root": str(repo),
                          "repo_masks": list(REPO_MASKS), "task_reference": str(opdir / "impl_torch.py")},
            "device_nodes": device_nodes()}


def detect_isolation(mode: str = "auto") -> dict:
    """mode: auto (bwrap when it works, else none), bwrap (required), none."""
    if mode in _ISOLATION_CACHE:
        return dict(_ISOLATION_CACHE[mode])
    report = {"backend": "none", "home_hidden": False, "network": "host", "pid_namespace": False,
              "filesystem": "host (read-write as the user)", "tmp_private": False, "probe_error": None,
              "env_secrets_scrubbed": True, "allowlist": None, "device_nodes": device_nodes()}
    if mode == "none":
        _ISOLATION_CACHE[mode] = report
        return dict(report)
    if not shutil.which("bwrap"):
        report["probe_error"] = "bwrap not found"
    else:
        try:
            subprocess.run(_bwrap_base(None) + ["/bin/true"], capture_output=True, timeout=60, check=True)
            report.update({"backend": "bwrap", "home_hidden": True, "network": "unshared", "pid_namespace": True,
                           "filesystem": "allowlist: system dirs, runtime prefix and repository read-only with data masks; "
                                         "sandbox read-write; /home,/root,/tmp,/var/tmp private tmpfs",
                           "tmp_private": True, "allowlist": {"system_ro": list(SYSTEM_RO), "runtime_prefix": runtime_prefix(),
                                                              "repo_masks": list(REPO_MASKS)}})
        except Exception as e:  # noqa: BLE001
            err = getattr(e, "stderr", b"")
            report["probe_error"] = f"{type(e).__name__}: {e} {err.decode(errors='replace') if isinstance(err, bytes) else err}".strip()
    if mode == "bwrap" and report["backend"] != "bwrap":
        raise RuntimeError(f"bwrap isolation required but unavailable: {report['probe_error']}")
    _ISOLATION_CACHE[mode] = report
    return dict(report)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CANDIDATE_PHASES = ("candidate_loaded", "first_execution", "first_execution_done", "numerical_checks_done", "timing_started",
                    "timing_done", "suite_done")


def _partial_cases(sandbox: Path) -> list[dict]:
    out = []
    try:
        for line in (sandbox / "cases.jsonl").read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return out


def classify_no_result(sandbox: Path, *, timed_out: bool, timeout_s: int, rc, stderr: str, job_cases: list | None = None) -> dict:
    """The worker produced no result.json. The worker's progress marker says
    how far it got: a hang or crash after the candidate was loaded is the
    candidate's ordinary failure (`runtime_error`, round consumed, no
    repair); before that it is `infrastructure_incomplete`. Cases the worker
    finished before it died (cases.jsonl) are kept; the case it was working
    on is recorded as the failing one; later cases are `not_evaluated`."""
    from tilebench.llm.v2.evaluation.worker import RESULT_SCHEMA, summarize_suite
    prog = {}
    try:
        prog = json.loads((sandbox / "progress.json").read_text())
    except (OSError, ValueError):
        pass
    phase, at_case = prog.get("phase"), prog.get("case")
    tail = stderr[-4000:]
    base = {"schema": RESULT_SCHEMA, "stages": {}, "worker_phase": phase, "worker_case_position": at_case, "timed_out": timed_out,
            "worker_exit": rc, "worker_killed_by_signal": (-rc if isinstance(rc, int) and rc < 0 else None)}
    if phase in CANDIDATE_PHASES:
        what = (f"candidate exceeded the evaluation wall-clock limit of {timeout_s}s" if timed_out
                else f"evaluation process ended (exit {rc}) without a result")
        diag = f"{what} during phase {phase} (case position {at_case}) of the generated file\n{tail}".rstrip()
        res = {**base, "status": "runtime_error", "diagnostic": diag}
        if job_cases:
            done = _partial_cases(sandbox)
            seen = {c.get("case_id") for c in done}
            records = list(done)
            dying = True
            for c in job_cases:
                if c["case_id"] in seen:
                    continue
                records.append({"case_id": c["case_id"], "case_index": c.get("case_index"), "params": c["params"],
                                "status": "runtime_error" if dying else "not_evaluated",
                                "diagnostic": diag if dying else "worker ended before this case"})
                dying = False
            summarize_suite(res, records, len(job_cases))
            res["status"], res["diagnostic"] = "runtime_error", diag
            res["suite_stopped"] = {"reason": "worker_death", "at_case": next((r["case_id"] for r in records if r["status"] == "runtime_error"), None)}
        return res
    what = f"worker exceeded {timeout_s}s" if timed_out else f"worker exited {rc} without a result"
    return {**base, "status": "infrastructure_incomplete", "diagnostic": f"{what} before the candidate was loaded (phase {phase})\n{tail}".rstrip()}


def cgroup_memory_events() -> dict:
    """oom / oom_kill counters and peak of this process's cgroup (v2), read-only; {} when unavailable."""
    try:
        rel = next(line.split("::", 1)[1].strip() for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    except (OSError, StopIteration):
        return {}
    out: dict = {"cgroup": rel}
    base = Path("/sys/fs/cgroup") / rel.lstrip("/")
    for d in [base] + list(base.parents):
        ev = d / "memory.events"
        if ev.exists() and (d / "memory.max").exists() and (d / "memory.max").read_text().strip() != "max":
            try:
                out.update({f"events.{k}": int(v) for k, v in (ln.split() for ln in ev.read_text().splitlines())})
                out["memory.max"] = (d / "memory.max").read_text().strip()
                pk = d / "memory.peak"
                out["memory.peak"] = pk.read_text().strip() if pk.exists() else None
                out["limited_cgroup"] = str(d)
            except (OSError, ValueError):
                pass
            break
    return out


def archive_sandbox(sandbox: Path, archive_dir: Path, extra: dict | None = None) -> dict:
    """Copy the evidence of one evaluation (job, result, stdout/stderr, the
    candidate file, the retained Proton profile) with sha256s; compile
    caches and other large files stay behind and are deleted with the
    sandbox."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    index: dict = {"files": {}, "extra": extra or {}}
    names = ["job.json", "result.json", "worker_stdout.txt", "worker_stderr.txt", "progress.json", "cases.jsonl"]
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
    timeout_s: int = 3600
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
                from tilebench.paths import operator_dir
                argv = _bwrap_base(sandbox, operator_dir=Path(operator_dir(job.operator))) + argv
            env = scrubbed_env(sandbox)
            t_lock = time.time()
            with device_lock(self.device, root=self.lock_root):
                lock_wait = time.time() - t_lock
                mem_before = cgroup_memory_events()
                t0 = time.time()
                try:
                    proc = subprocess.run(argv, cwd=str(sandbox), env=env, timeout=self.timeout_s,
                                          capture_output=True, text=True)
                    rc, stdout, stderr, timed_out = proc.returncode, proc.stdout, proc.stderr, False
                except subprocess.TimeoutExpired as e:
                    rc, timed_out = None, True
                    stdout = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
                    stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
                mem_after = cgroup_memory_events()
            wall = time.time() - t0
            (sandbox / "worker_stdout.txt").write_text(stdout or "")
            (sandbox / "worker_stderr.txt").write_text(stderr or "")
            if timed_out or not out.exists():
                res = classify_no_result(sandbox, timed_out=timed_out, timeout_s=self.timeout_s, rc=rc, stderr=stderr or "",
                                         job_cases=getattr(job, "cases", None))
            else:
                res = json.loads(out.read_text())
            res["resources"] = {"cgroup_before": mem_before, "cgroup_after": mem_after,
                                "cgroup_oom_kill_delta": (mem_after.get("events.oom_kill", 0) - mem_before.get("events.oom_kill", 0))
                                if mem_before and mem_after else None}
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


def mock_suite(job, kind: str, base: float = 1.0, fail_at: int | None = None) -> dict:
    """A suite result shaped like the worker's (schema eval/3), for tests and dry runs."""
    from tilebench.llm.v2.evaluation.worker import RESULT_SCHEMA, summarize_suite
    cases = list(getattr(job, "cases", None) or (job.get("cases") if isinstance(job, dict) else None) or
                 [{"case_id": "c0", "case_index": 0, "params": {}}])
    recs = []
    res: dict = {"schema": RESULT_SCHEMA, "status": None, "isolation": {"backend": "mock"}, "stages": {}}
    for i, c in enumerate(cases):
        lat = base * (1.0 + 0.1 * i)
        samples = [lat * 0.99, lat, lat * 1.01]
        if kind == "valid" or (fail_at is not None and i < fail_at):
            recs.append({"case_id": c["case_id"], "case_index": c.get("case_index"), "params": c["params"], "status": "valid",
                         "latency_ms_mean": sum(samples) / 3, "latency_ms_samples": samples, "config": {"BLOCK": 128},
                         "timing": {"timing_execution_mode": "graph", "capture_succeeded": True}, "timing_mode_differs": False})
        else:
            recs.append({"case_id": c["case_id"], "case_index": c.get("case_index"), "params": c["params"], "status": kind,
                         "diagnostic": f"mock {kind} diagnostic\nroofline 55% should be scrubbed"})
    if kind == "contract_violation":
        res["status"] = "contract_violation"
    summarize_suite(res, recs, len(cases))
    if kind == "contract_violation":
        res.update(status="contract_violation", diagnostic="contract_violation: mock execution evidence")
    return res


@dataclass
class MockEvaluator:
    """Scripted outcomes keyed by the source text's first line marker, e.g.
    '# MOCK: valid 1.5' -> every case valid with samples around 1.5 ms (x 1.1 per case),
    '# MOCK: numerical_error' (every case), '# MOCK: numerical_error_at 3' (cases 0..2 valid, case 3 on fails),
    '# MOCK: compile_error', '# MOCK: infrastructure', '# MOCK: contract_violation', '# MOCK: interface_error'."""
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
            return mock_suite(job, "valid", float(parts[1]) if len(parts) > 1 else 1.0)
        if kind == "infrastructure":
            return {"status": "infrastructure_incomplete", "diagnostic": "mock infrastructure failure"}
        if kind.endswith("_at"):
            return mock_suite(job, kind[:-3], 1.0, fail_at=int(parts[1]) if len(parts) > 1 else 0)
        return mock_suite(job, kind)
