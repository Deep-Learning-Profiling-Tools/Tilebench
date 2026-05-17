"""NCU sweep driver — iterate ncu_catalogue.json, run NCU on each
(op, dtype, backend) triple at sweep-max input.

For each op/dtype:
  - Use autotune-winner cfg from catalogue if present; else use default
    (no _DEFAULT_CONFIG override).
  - Run NCU on both backends.
  - Report under tilebench_run/ncu/<op>/{backend}_{dtype}.ncu-rep.
  - On failure: log and continue.

Progress is appended to tilebench_run/ncu/sweep_log.json after each pair.
Run with: PYTHONPATH=. python tilebench_run/ncu_driver.py
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/projects/kzhou6/bcui2/research/tilebench/Tilebench")
NCU = "/usr/local/cuda/bin/ncu"
HARNESS = ROOT / "tilebench_run" / "ncu_generic_harness.py"
CATALOGUE = ROOT / "tilebench_run" / "ncu_catalogue.json"
NCU_DIR = ROOT / "tilebench_run" / "ncu"
LOG_PATH = NCU_DIR / "sweep_log.json"
FAIL_PATH = NCU_DIR / "sweep_failures.md"

KERNEL_REGEX_BY_BACKEND_DEFAULT = ".*"


def out_path(op: str, backend: str, dtype: str) -> Path:
    return NCU_DIR / op / f"{backend}_{dtype}.ncu-rep"


def run_one(op: str, dtype: str, backend: str, params: dict, cfg: dict | None,
            timeout_s: int = 600) -> dict:
    out = out_path(op, backend, dtype)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["NCU_OP"] = op
    env["NCU_BACKEND"] = backend
    env["NCU_DTYPE"] = dtype
    env["NCU_PARAMS_JSON"] = json.dumps(params)
    if cfg is not None:
        env["NCU_CFG_JSON"] = json.dumps(cfg)
    env["PYTHONPATH"] = str(ROOT)

    cmd = [
        NCU, "--set", "full", "--import-source", "on",
        "--launch-skip", "3", "--launch-count", "1",
        "--force-overwrite",
        "-o", str(out).removesuffix(".ncu-rep"),
        sys.executable, str(HARNESS),
    ]
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, env=env, cwd=str(ROOT),
            capture_output=True, text=True, timeout=timeout_s,
        )
        elapsed = time.time() - t0
        ok = (r.returncode == 0) and out.exists() and out.stat().st_size > 0
        return {
            "op": op, "dtype": dtype, "backend": backend,
            "ok": ok, "rc": r.returncode, "elapsed_s": round(elapsed, 2),
            "stderr_tail": r.stderr[-600:] if not ok else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "op": op, "dtype": dtype, "backend": backend,
            "ok": False, "rc": -1, "elapsed_s": timeout_s,
            "stderr_tail": "TIMEOUT",
        }


def main() -> None:
    catalogue = json.loads(CATALOGUE.read_text())
    NCU_DIR.mkdir(parents=True, exist_ok=True)
    pairs = []
    for c in catalogue:
        op = c["op"]
        for dt in c["dtypes"]:
            params = c["default_params_per_dtype"][dt]
            winner = c["autotune_winner_per_dtype"].get(dt)
            for backend in ("triton", "cutile"):
                cfg = None
                if winner is not None:
                    cfg = winner.get(backend)
                pairs.append((op, dt, backend, params, cfg))

    total = len(pairs)
    log = []
    if LOG_PATH.exists():
        try:
            log = json.loads(LOG_PATH.read_text())
        except Exception:
            log = []
    done_keys = {(e["op"], e["dtype"], e["backend"]) for e in log if e.get("ok")}

    fails: list[dict] = [e for e in log if not e.get("ok")]
    t_start = time.time()

    for i, (op, dt, backend, params, cfg) in enumerate(pairs):
        key = (op, dt, backend)
        if key in done_keys:
            continue
        out = out_path(op, backend, dt)
        if out.exists() and out.stat().st_size > 0:
            log.append({"op": op, "dtype": dt, "backend": backend, "ok": True,
                        "rc": 0, "elapsed_s": 0, "stderr_tail": "(pre-existing)"})
            done_keys.add(key)
            continue
        res = run_one(op, dt, backend, params, cfg)
        log.append(res)
        if not res["ok"]:
            fails.append(res)
        LOG_PATH.write_text(json.dumps(log, indent=2))

        done = len(log)
        if done % 20 == 0 or done == total:
            ok_n = sum(1 for e in log if e["ok"])
            elapsed = time.time() - t_start
            print(f"[progress] {done}/{total} done, ok={ok_n}, "
                  f"fail={done-ok_n}, elapsed={elapsed/60:.1f} min",
                  flush=True)

    ok_n = sum(1 for e in log if e["ok"])
    print(f"\n=== DONE === {ok_n}/{len(log)} ok, {len(log)-ok_n} failed",
          flush=True)
    if fails:
        lines = ["# NCU sweep failures\n"]
        for f in fails:
            lines.append(
                f"- **{f['op']}/{f['dtype']}/{f['backend']}** "
                f"(rc={f['rc']}, {f['elapsed_s']}s)\n"
                f"  ```\n  {f['stderr_tail'].strip()}\n  ```\n"
            )
        FAIL_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
