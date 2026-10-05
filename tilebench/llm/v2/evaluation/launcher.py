"""Launches the isolated worker with a scrubbed environment, a sandbox
directory, redirected compile caches, the per-device lock and a wall-clock
timeout. Also provides the mock evaluator used by tests and dry runs."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tilebench.paths import REPO_ROOT

from tilebench.llm.v2.orchestration.locks import device_lock

SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "HUGGING_FACE", "HF_")


def scrubbed_env(sandbox: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in SECRET_MARKERS)}
    env["TILEBENCH_V2_SANDBOX"] = "1"
    env["TRITON_CACHE_DIR"] = str(sandbox / "triton_cache")
    env["CUDA_TILE_CACHE_DIR"] = str(sandbox / "cutile_cache")
    env["TILELANG_CACHE_DIR"] = str(sandbox / "tilelang_cache")
    env["PYTHONPATH"] = str(REPO_ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("PYTHONSTARTUP", None)
    return env


@dataclass
class SubprocessEvaluator:
    device: str
    timing: dict
    timeout_s: int = 1800
    sandbox_root: Path | None = None

    def evaluate(self, source_path: Path, task: dict, round_index: int, attempt: int) -> dict:
        root = self.sandbox_root or Path(tempfile.gettempdir()) / "tilebench_llm_v2_sandbox"
        root.mkdir(parents=True, exist_ok=True)
        sandbox = Path(tempfile.mkdtemp(prefix=f"{task['operator']}_{round_index}_{attempt}_", dir=root))
        try:
            copied = sandbox / source_path.name
            shutil.copyfile(source_path, copied)
            job = {"operator": task["operator"], "dtype": task["dtype"], "params": task["params"], "dsl": task["dsl"],
                   "source_path": str(copied), "sandbox_dir": str(sandbox), "atol": task["atol"], "rtol": task["rtol"],
                   "rules": task.get("rules", {}), "timing": self.timing,
                   "seed": int(time.time_ns() % (2**31))}
            (sandbox / "job.json").write_text(json.dumps(job, indent=1))
            out = sandbox / "result.json"
            with device_lock(self.device):
                try:
                    proc = subprocess.run([sys.executable, "-m", "tilebench.llm.v2.evaluation.worker",
                                           "--job", str(sandbox / "job.json"), "--out", str(out)],
                                          cwd=str(sandbox), env=scrubbed_env(sandbox), timeout=self.timeout_s,
                                          capture_output=True, text=True)
                except subprocess.TimeoutExpired:
                    return {"status": "infrastructure_incomplete", "diagnostic": f"worker exceeded {self.timeout_s}s",
                            "seed": job["seed"]}
            if not out.exists():
                return {"status": "infrastructure_incomplete", "diagnostic": f"worker exited {proc.returncode} without a result\n"
                        + (proc.stderr or "")[-4000:], "seed": job["seed"]}
            res = json.loads(out.read_text())
            res["seed"] = job["seed"]
            res["worker_rc"] = proc.returncode
            return res
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)


@dataclass
class MockEvaluator:
    """Scripted outcomes keyed by the source text's first line marker, e.g.
    '# MOCK: valid 1.5' -> valid with samples around 1.5 ms,
    '# MOCK: numerical_error', '# MOCK: compile_error', '# MOCK: infrastructure'."""
    calls: list = None

    def evaluate(self, source_path: Path, task: dict, round_index: int, attempt: int) -> dict:
        self.calls = self.calls if self.calls is not None else []
        self.calls.append((task["operator"], round_index, attempt))
        first = source_path.read_text().splitlines()[0] if source_path.read_text() else ""
        parts = first.replace("# MOCK:", "").split()
        kind = parts[0] if parts else "valid"
        if kind == "valid":
            base = float(parts[1]) if len(parts) > 1 else 1.0
            samples = [base * 0.99, base, base * 1.01]
            return {"status": "valid", "latency_ms_mean": sum(samples) / 3, "latency_ms_samples": samples,
                    "config": {"BLOCK": 128}, "timing": {"timing_execution_mode": "graph", "capture_succeeded": True}}
        if kind == "infrastructure":
            return {"status": "infrastructure_incomplete", "diagnostic": "mock infrastructure failure"}
        return {"status": kind, "diagnostic": f"mock {kind} diagnostic\nroofline 55% should be scrubbed"}
