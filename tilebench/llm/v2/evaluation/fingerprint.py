"""Evaluator fingerprint: what actually decides a candidate's verdict and
its timing, hashed so that a resume can tell whether the evaluator changed.

Covered (and only this; unrelated files of the repository do not matter):
- the evaluation job: operator/dtype/params, effective tolerance, the
  evaluator rules (sha256), normalized timing settings, capture-failure
  policy, expected timing mode, timing adapter;
- the checker: `CHECKER_VERSION` and the sources of the static / contract
  checks and the strict parser;
- the evaluation implementation: worker, timing, anticache, launcher, job,
  plus `tilebench.core.timer` and `tilebench.core.verifier`;
- the launcher settings that bound the measurement: worker timeout,
  isolation backend;
- the environment: package versions (torch, triton, cuda-tile, tilelang,
  apache-tvm-ffi), CUDA/HIP runtime version, the first visible device name.

`fingerprint_sha256` is the hash of the canonical JSON of everything above;
`diff()` lists the differing keys so a refusal names the change."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
from pathlib import Path

from tilebench.llm.v2.manifests.schema import canonical_json, sha256_text

V2_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = V2_ROOT.parents[1] / "core"

CHECKER_SOURCES = ("validation/static_checks.py", "validation/contract_checks.py", "validation/parser.py")
EVALUATION_SOURCES = ("evaluation/worker.py", "evaluation/timing.py", "evaluation/anticache.py",
                      "evaluation/launcher.py", "evaluation/job.py")
CORE_SOURCES = ("timer.py", "verifier.py")
PACKAGES = ("torch", "triton", "cuda-tile", "tilelang", "apache-tvm-ffi")


def _sources_sha256(root: Path, names: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for n in names:
        p = root / n
        h.update(n.encode()); h.update(b"\0")
        h.update(p.read_bytes() if p.exists() else b"<missing>"); h.update(b"\0")
    return h.hexdigest()


def checker_fingerprint() -> dict:
    from tilebench.llm.v2.validation.contract_checks import CHECKER_VERSION
    return {"checker_version": CHECKER_VERSION, "checker_sources_sha256": _sources_sha256(V2_ROOT, CHECKER_SOURCES)}


def environment_identity() -> dict:
    env: dict = {"packages": {}}
    for pkg in PACKAGES:
        try:
            env["packages"][pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            env["packages"][pkg] = None
    try:
        import torch
        env["torch_cuda"] = torch.version.cuda
        env["torch_hip"] = torch.version.hip
        env["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        env["torch_cuda"] = env["torch_hip"] = env["device_name"] = None
    return env


def evaluator_fingerprint(job, *, worker_timeout_s: int, isolation_backend: str) -> dict:
    """job: evaluation.job.EvaluationJob. Returns a JSON-able record whose
    `fingerprint_sha256` changes whenever anything that decides verdicts or
    timing changes."""
    jr = job.record()
    jr.pop("identity", None)
    body = {
        "job": jr,
        "rules_sha256": job.rules_sha256(),
        "checker": checker_fingerprint(),
        "evaluation_sources_sha256": _sources_sha256(V2_ROOT, EVALUATION_SOURCES),
        "core_sources_sha256": _sources_sha256(CORE_ROOT, CORE_SOURCES),
        "worker_timeout_s": int(worker_timeout_s),
        "isolation_backend": isolation_backend,
        "environment": environment_identity(),
    }
    body["fingerprint_sha256"] = sha256_text(canonical_json(body))
    return body


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def diff(a: dict | None, b: dict | None) -> list[str]:
    """Keys whose values differ between two fingerprints (empty = identical)."""
    fa, fb = _flatten(a or {}), _flatten(b or {})
    return sorted(k for k in set(fa) | set(fb) if k != "fingerprint_sha256" and fa.get(k) != fb.get(k))
