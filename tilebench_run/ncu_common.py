"""Shared helpers for the TileBench NCU scripts."""

from __future__ import annotations

import os
import re
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace


BACKENDS = ("triton", "cutile", "tilelang")
LEGACY_BACKENDS = ("triton", "cutile")


def repo_root() -> Path:
    """Resolve the TileBench repository root.

    Defaults to the parent of this file's directory, with TILEBENCH_ROOT as an
    escape hatch for remote boxes or copied scripts.
    """
    return Path(os.environ.get("TILEBENCH_ROOT", Path(__file__).resolve().parents[1])).resolve()


def ncu_bin() -> str:
    """Resolve the Nsight Compute CLI binary."""
    return os.environ.get("NCU_BIN") or shutil.which("ncu") or "/usr/local/cuda/bin/ncu"


def ncu_run_dir() -> Path:
    """Directory containing NCU reports/logs for one profiling run."""
    return Path(os.environ.get("NCU_RUN_DIR", repo_root() / "tilebench_run" / "ncu")).resolve()


def ncu_source_dir() -> Path:
    """Persistent generated-source dump directory used during NCU captures."""
    return Path(os.environ.get("NCU_SOURCE_DIR", ncu_run_dir() / "ncu_source")).resolve()


def ncu_json_dir() -> Path:
    """Structured NCU export directory consumed by downstream tools."""
    return Path(os.environ.get("NCU_JSON_DIR", repo_root() / "tilebench_run" / "ncu_json")).resolve()


def ncu_source_folders() -> str:
    """Comma-separated recursive source lookup paths for `ncu --source-folders`."""
    root = repo_root()
    folders = [
        root,
        root / "benchmarks",
        root / "data",
        ncu_source_dir(),
    ]
    extra = os.environ.get("NCU_SOURCE_FOLDERS")
    if extra:
        folders.extend(Path(p).expanduser() for p in extra.split(",") if p.strip())

    seen: set[str] = set()
    out: list[str] = []
    for folder in folders:
        resolved = str(Path(folder).resolve())
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return ",".join(out)


def ncu_profile_source_args() -> list[str]:
    return ["--import-source", "yes", "--source-folders", ncu_source_folders()]


def prepare_profile_env(env: dict[str, str], *, op: str, backend: str, dtype: str) -> dict[str, str]:
    """Set profiling-time compiler/source knobs without changing benchmark code."""
    env["PYTHONPATH"] = str(repo_root())
    # Triton 3.x passes -lineinfo to ptxas unless this is set.
    env.pop("TRITON_DISABLE_LINE_INFO", None)
    # TileLang/cuTile generated CUDA goes through nvcc in common installs.
    # -lineinfo is for profiling; unlike -G it does not intentionally disable
    # device optimizations.
    flags = env.get("NVCC_PREPEND_FLAGS", "")
    for flag in ("-lineinfo", "-allow-unsupported-compiler"):
        if flag not in flags:
            flags = (flags + " " + flag).strip()
    env["NVCC_PREPEND_FLAGS"] = flags
    env["TILEBENCH_NCU_OP"] = op
    env["TILEBENCH_NCU_BACKEND"] = backend
    env["TILEBENCH_NCU_DTYPE"] = dtype
    env["TILEBENCH_NCU_SOURCE_DUMP_DIR"] = str(ncu_source_dir())
    env.setdefault("TILEBENCH_NCU_CC_BIN", shutil.which("g++-13") or shutil.which("g++") or "")
    return env


def parse_backends(raw: str | None, *, default=BACKENDS) -> tuple[str, ...]:
    if not raw:
        return tuple(default)
    names = tuple(x.strip() for x in raw.split(",") if x.strip())
    bad = sorted(set(names) - set(BACKENDS))
    if bad:
        raise SystemExit(f"unknown NCU backend(s): {bad}; expected one of {BACKENDS}")
    return names


_PREFIXED_CONFIG_RE = re.compile(r"^_DEFAULT_([A-Z0-9]+)_CONFIG$")


def _inject_prefixed(impl, cfg: dict) -> dict:
    """Route prefixed winner keys into per-kernel config dicts."""
    targets = {}
    for attr in dir(impl):
        m = _PREFIXED_CONFIG_RE.match(attr)
        if m:
            targets[m.group(1).lower()] = (attr, getattr(impl, attr))
    if not targets:
        return cfg

    leftover = {}
    for key, val in cfg.items():
        placed = False
        for pref, (attr, target) in targets.items():
            if not key.lower().startswith(pref + "_"):
                continue
            base = key[len(pref) + 1:]
            if isinstance(target, dict) and base in target:
                target[base] = val
                placed = True
            elif isinstance(target, SimpleNamespace) and hasattr(target, base):
                setattr(target, base, val)
                placed = True
            if placed:
                print(f"  cfg: {attr}[{base}] = {val}")
                break
        if not placed:
            leftover[key] = val
    return leftover


def apply_config_override(impl, cfg: dict | None, dtype_key=None) -> None:
    """Apply an autotune-winner config to an implementation module."""
    if not cfg:
        return
    cfg = _inject_prefixed(impl, dict(cfg))
    if not cfg:
        return

    existing = getattr(impl, "_DEFAULT_CONFIG", None)
    configs = getattr(impl, "_DEFAULT_CONFIGS", None)

    if existing is None and configs is not None:
        if dtype_key in configs:
            cur = configs[dtype_key]
            if isinstance(cur, dict):
                merged = dict(cur)
                merged.update(cfg)
                configs[dtype_key] = merged
            else:
                merged = vars(cur).copy()
                merged.update(cfg)
                configs[dtype_key] = SimpleNamespace(**merged)
        else:
            configs[dtype_key] = SimpleNamespace(**cfg)
    elif isinstance(existing, dict):
        merged = dict(existing)
        merged.update(cfg)
        impl._DEFAULT_CONFIG = merged
    elif isinstance(existing, SimpleNamespace):
        merged = vars(existing).copy()
        merged.update(cfg)
        impl._DEFAULT_CONFIG = SimpleNamespace(**merged)
    else:
        impl._DEFAULT_CONFIG = SimpleNamespace(**cfg)


def _write_tilelang_source(op: str, backend: str, dtype: str, code: str, counter: dict[str, int]) -> str:
    dump_root = os.environ.get("TILEBENCH_NCU_SOURCE_DUMP_DIR")
    if not dump_root:
        return code
    if code.lstrip().startswith("#line "):
        return code
    out_dir = Path(dump_root) / op
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = counter["i"]
    counter["i"] += 1
    path = out_dir / f"{backend}_{dtype}_{idx:02d}.cu"
    path.write_text(code)
    return f'#line 1 "{path}"\n{code}'


def register_tilelang_source_capture(op: str, backend: str, dtype: str) -> None:
    """Persist TileLang generated CUDA under a stable path before NVCC compiles it.

    NCU can import source only when line-info references a file it can find. The
    callback writes the generated CUDA to a stable path and injects a #line
    directive so NVCC's line table points at that stable copy rather than a
    transient internal string/temp path.
    """
    if backend != "tilelang":
        return
    dump_root = os.environ.get("TILEBENCH_NCU_SOURCE_DUMP_DIR")
    if not dump_root:
        return
    try:
        from tilelang.engine.callback import register_cuda_postproc_callback
    except Exception as exc:
        print(f"  WARNING: TileLang source capture unavailable: {exc}")
        return

    counter = {"i": 0}

    @register_cuda_postproc_callback
    def tilelang_callback_cuda_postproc(code, _target):
        return _write_tilelang_source(op, backend, dtype, code, counter)


def register_tilelang_compile_hook(op: str, backend: str, dtype: str) -> None:
    """Override TileLang CUDA compilation for profiling-only flags/source paths."""
    if backend != "tilelang":
        return
    try:
        import tvm_ffi
        from tilelang.contrib import nvcc
        from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
        from tilelang.transform import PassConfigKey
    except Exception as exc:
        print(f"  WARNING: TileLang compile hook unavailable: {exc}")
        return

    counter = {"i": 0}

    @tvm_ffi.register_global_func("tilelang_callback_cuda_compile", override=True)
    def tilebench_callback_cuda_compile(code, target, pass_config=None):
        code = _write_tilelang_source(op, backend, dtype, code, counter)
        target_arch = nvcc.get_target_arch(nvcc.get_target_compute_version(target))
        arch = [f"-arch=sm_{target_arch}"]
        cfg = pass_config or {}
        options = [
            "-std=c++20",
            "-I" + TILELANG_TEMPLATE_PATH,
            "-I" + CUTLASS_INCLUDE_DIR,
            "-allow-unsupported-compiler",
        ]
        ccbin = os.environ.get("TILEBENCH_NCU_CC_BIN") or shutil.which("g++-13") or shutil.which("g++")
        if ccbin:
            options += ["-ccbin", ccbin]

        extra_flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
        if extra_flags:
            tokens = shlex.split(extra_flags) if isinstance(extra_flags, str) else []
            if not isinstance(extra_flags, str):
                for flag in extra_flags:
                    tokens.extend(shlex.split(flag) if isinstance(flag, str) else [str(flag)])
            options += tokens

        verbose = False
        if bool(cfg.get(PassConfigKey.TL_ENABLE_FAST_MATH, False)):
            options.append("--use_fast_math")
        ptxas_usage_level = cfg.get(PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL, None)
        if ptxas_usage_level is not None:
            options.append(f"--ptxas-options=--register-usage-level={int(ptxas_usage_level)}")
        if bool(cfg.get(PassConfigKey.TL_ENABLE_PTXAS_VERBOSE_OUTPUT, False)):
            options.append("--ptxas-options=--verbose")
            options.append("-w")
            verbose = True

        return nvcc.compile_cuda(code, "cubin", arch, options=options, verbose=verbose)
