"""Dump B200 (sm_100a) PTX/SASS for an operator's kernels, generated offline.

Unlike lower_to_ptx.py (which compiles for the *present* GPU by running the
kernels), this forces the compile target to Blackwell and extracts ISA without
needing a B200 -- codegen is host-side. Kernels are NOT executed for real; any
launch that TileLang/Triton/cuTile attempts on the local GPU is swallowed.

Per backend, for the first benchmark case of the operator:
  - TileLang: forces every JITKernel target to sm_100a, emits PTX via nvcc
    (needs a Blackwell-capable CUDA toolkit -- see CUDA_HOME below), then
    ptxas+cuobjdump for SASS.
  - Triton: forces the compile target to sm_100 and harvests PTX from the
    Triton cache, then ptxas+cuobjdump for SASS.
  - cuTile: forces cuda.tile's sm_arch to sm_100a; tileiras emits a cubin
    directly (no PTX), disassembled to SASS via cuobjdump.

Usage:
    PYTHONPATH=. python lower_to_ptx_sm100.py --mul2
    PYTHONPATH=. python lower_to_ptx_sm100.py --all

Artifacts are written to results/ptx_sm100a/<operator>/.
"""

import argparse
import importlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OPERATORS_ROOT = REPO_ROOT / "benchmarks" / "operators"

ARCH = "sm_100a"          # Blackwell, architecture-specific (TileLang/Triton/ptxas)
PTXAS_ARCH = "sm_100a"
CUTILE_ARCH = "sm_100"    # cuTile parses sm_arch as int(...), rejects the 'a' suffix

# A Blackwell-capable CUDA toolkit. The conda env root works as CUDA_HOME
# (bin/nvcc + targets/x86_64-linux/include). Overridable via env.
CUDA_HOME = os.environ.get("CUDA_HOME", sys.prefix)
os.environ["CUDA_HOME"] = CUDA_HOME
os.environ["PATH"] = f"{CUDA_HOME}/bin:" + os.environ.get("PATH", "")

_TRITON_CACHE = tempfile.mkdtemp(prefix="lower_sm100_triton_")
os.environ["TRITON_CACHE_DIR"] = _TRITON_CACHE

import torch  # noqa: E402
import yaml  # noqa: E402

from core.dtypes import resolve_dtype  # noqa: E402
from data.tensors import expand_cases, get_generator  # noqa: E402


def _tool(name):
    """Prefer a Blackwell-capable tool: conda bin, else Triton's bundled 12.8."""
    sp = Path(sys.prefix) / "lib/python3.10/site-packages"
    for cand in (
        sp / "nvidia/cuda_nvcc/bin" / name,   # pip 12.9 ptxas (handles PTX .version 8.8)
        sp / "triton/backends/nvidia/bin" / name,
        Path(CUDA_HOME) / "bin" / name,
        Path(sys.prefix) / "bin" / name,
    ):
        if cand.is_file():
            return str(cand)
    from shutil import which

    return which(name)


def ptx_to_sass(ptx_text):
    """ptxas (sm_100a) -> cubin -> cuobjdump --dump-sass. Returns SASS or None."""
    ptxas, cuobjdump = _tool("ptxas"), _tool("cuobjdump")
    if not ptxas or not cuobjdump:
        return None
    with tempfile.TemporaryDirectory() as d:
        ptx_path = Path(d) / "k.ptx"
        cubin_path = Path(d) / "k.cubin"
        ptx_path.write_text(ptx_text)
        r = subprocess.run(
            [ptxas, f"-arch={PTXAS_ARCH}", "-O3", str(ptx_path), "-o", str(cubin_path)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return f"// ptxas failed:\n// {r.stderr.strip()}\n"
        r = subprocess.run(
            [cuobjdump, "--dump-sass", str(cubin_path)],
            capture_output=True, text=True,
        )
        return r.stdout if r.returncode == 0 else f"// cuobjdump failed:\n// {r.stderr}\n"


def cubin_to_sass(cubin_bytes):
    cuobjdump = _tool("cuobjdump")
    if not cuobjdump:
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin_bytes)
        path = f.name
    try:
        r = subprocess.run([cuobjdump, "--dump-sass", path], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 and r.stdout.strip() else None
    finally:
        os.unlink(path)


def config_dtypes(operator):
    with (OPERATORS_ROOT / operator / "config.yaml").open() as f:
        config = yaml.safe_load(f)
    d = config.get("case_grid", {}).get("dtype", [])
    return list(d) if isinstance(d, list) else [d]


def pick_case(operator, dtype=None):
    """First benchmark case, optionally the first matching a requested dtype."""
    with (OPERATORS_ROOT / operator / "config.yaml").open() as f:
        config = yaml.safe_load(f)
    cases = expand_cases(operator, config)
    if dtype is not None:
        cases = [c for c in cases if c.get("dtype") == dtype] or cases
    case = cases[0]
    params = {k: v for k, v in case.items() if k not in ("dtype", "block_size")}
    resolved = resolve_dtype(case.get("dtype", "fp32"))
    return case, get_generator(operator)(**params, dtype=resolved)


def _try_import(operator, impl):
    try:
        return importlib.import_module(f"benchmarks.operators.{operator}.{impl}")
    except ImportError as e:
        print(f"// {impl}: import skipped ({e})", file=sys.stderr)
        return None


def _run_kwargs(fn):
    import inspect

    sig = inspect.signature(fn).parameters
    kw = {}
    if "autotune" in sig:
        kw["autotune"] = False
    return kw


def _entry_name(ptx, fallback):
    m = re.search(r"\.(?:visible \.)?entry\s+(\w+)", ptx)
    return m.group(1) if m else fallback


# ---------------- TileLang ----------------
def collect_tilelang(operator, inputs):
    module = _try_import(operator, "impl_tilelang")
    if module is None:
        return []
    import tilelang  # noqa: F401
    import tvm
    from tilelang.jit.kernel import JITKernel

    forced = tvm.target.Target({"kind": "cuda", "arch": ARCH}, host="llvm")
    kmod = sys.modules["tilelang.jit.kernel"]
    orig_dt = kmod.determine_target
    kmod.determine_target = lambda *a, **k: forced

    captured = []
    orig_init = JITKernel.__init__
    orig_call = JITKernel.__call__

    def spy(self, *a, **k):
        orig_init(self, *a, **k)
        captured.append(self)

    # Skip the real GPU launch: the kernel is already compiled at __init__ (that's
    # what we capture), and Blackwell code can't run on the local GPU anyway.
    # No-op lets multi-kernel run()s construct every kernel instead of aborting
    # at the first launch. Intermediate kernels write in-place, so a None return
    # is harmless for those flows.
    def noop_call(self, *a, **k):
        return None

    JITKernel.__init__ = spy
    JITKernel.__call__ = noop_call
    try:
        try:
            module.run(*inputs, **_run_kwargs(module.run))
        except Exception as e:
            print(f"// tilelang: run stopped early ({type(e).__name__}: {str(e)[:80]})", file=sys.stderr)
    finally:
        JITKernel.__init__ = orig_init
        JITKernel.__call__ = orig_call
        kmod.determine_target = orig_dt

    out = []
    for kern in captured:
        try:
            ptx = kern._get_ptx()
        except Exception as e:
            print(f"// tilelang: _get_ptx failed ({str(e)[:120]})", file=sys.stderr)
            continue
        out.append((_entry_name(ptx, f"kernel{len(out)}"), ptx))
    return out


# ---------------- Triton ----------------
def collect_triton(operator, inputs):
    module = _try_import(operator, "impl_triton")
    if module is None:
        return []
    import triton

    drv = triton.runtime.driver.active
    orig_target = drv.get_current_target
    cur = orig_target()
    forced = type(cur)(cur.backend, 100, cur.warp_size)
    drv.get_current_target = lambda: forced

    # Force warmup mode: Triton compiles (and caches PTX) but does not launch,
    # so a multi-kernel run() constructs every kernel instead of aborting.
    from triton.runtime.jit import JITFunction

    orig_run = JITFunction.run

    def warmup_run(self, *args, **kwargs):
        kwargs["warmup"] = True
        try:
            return orig_run(self, *args, **kwargs)
        except Exception:
            return None

    JITFunction.run = warmup_run

    before = set(Path(_TRITON_CACHE).rglob("*.ptx"))
    try:
        try:
            module.run(*inputs, **_run_kwargs(module.run))
        except Exception as e:
            print(f"// triton: run stopped early ({type(e).__name__}: {str(e)[:80]})", file=sys.stderr)
    finally:
        drv.get_current_target = orig_target
        JITFunction.run = orig_run

    out = {}
    for p in sorted(Path(_TRITON_CACHE).rglob("*.ptx")):
        if p in before:
            continue
        txt = p.read_text()
        out.setdefault(_entry_name(txt, p.stem), txt)
    return list(out.items())


# ---------------- cuTile ----------------
def collect_cutile(operator, inputs):
    module = _try_import(operator, "impl_cutile")
    if module is None:
        return []
    try:
        import cuda.tile._compile as ctc
        from cuda.tile._execution import kernel as CutileKernel
    except ImportError as e:
        print(f"// cutile: unavailable ({e})", file=sys.stderr)
        return []

    import cuda.tile as ct

    orig_arch = ctc.get_sm_arch
    ctc.get_sm_arch = lambda: CUTILE_ARCH

    captured = []
    orig_compile = CutileKernel._compile

    def spy(self, signature, context):
        result = orig_compile(self, signature, context)
        cubin, symbol = result[0], result[1]
        if cubin is not None:
            captured.append((symbol or f"kernel{len(captured)}", cubin))
        return result

    # Swallow the actual GPU launch per-call so every kernel still hits _compile
    # (the cubin is built before the launch that fails on the local GPU).
    orig_launch = ct.launch

    def safe_launch(*a, **k):
        try:
            return orig_launch(*a, **k)
        except Exception:
            return None

    CutileKernel._compile = spy
    ct.launch = safe_launch
    try:
        try:
            module.run(*inputs, **_run_kwargs(module.run))
        except Exception as e:
            print(f"// cutile: run stopped early ({type(e).__name__}: {str(e)[:80]})", file=sys.stderr)
    finally:
        CutileKernel._compile = orig_compile
        ct.launch = orig_launch
        ctc.get_sm_arch = orig_arch

    out = []
    for symbol, cubin in captured:
        out.append((symbol, cubin, cubin_to_sass(cubin)))
    return out


def _safe(name):
    return re.sub(r"[^\w.\-]+", "_", name).strip("_") or "kernel"


def process(operator, dtype=None):
    case, inputs = pick_case(operator, dtype)
    print(f"// operator: {operator}  case: {case}", file=sys.stderr)
    out_dir = REPO_ROOT / "results" / "ptx_sm100a" / operator
    if dtype is not None:
        out_dir = out_dir / _safe(dtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for name, ptx in collect_tilelang(operator, inputs):
        p = out_dir / f"tilelang_{_safe(name)}.ptx"
        p.write_text(ptx)
        written.append(p)
        sass = ptx_to_sass(ptx)
        if sass:
            s = out_dir / f"tilelang_{_safe(name)}.sass"
            s.write_text(sass)
            written.append(s)

    for name, ptx in collect_triton(operator, inputs):
        p = out_dir / f"triton_{_safe(name)}.ptx"
        p.write_text(ptx)
        written.append(p)
        sass = ptx_to_sass(ptx)
        if sass:
            s = out_dir / f"triton_{_safe(name)}.sass"
            s.write_text(sass)
            written.append(s)

    for name, cubin, sass in collect_cutile(operator, inputs):
        c = out_dir / f"cutile_{_safe(name)}.cubin"
        c.write_bytes(cubin)
        written.append(c)
        if sass:
            s = out_dir / f"cutile_{_safe(name)}.sass"
            s.write_text(sass)
            written.append(s)

    for p in written:
        print(f"//   wrote {p}", file=sys.stderr)
    return written


def all_operators():
    return sorted(
        p.name
        for p in OPERATORS_ROOT.iterdir()
        if p.is_dir() and (p / "config.yaml").is_file()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dtype", help="only this dtype (e.g. fp16, fp8_e4m3fn); writes to <op>/<dtype>/")
    parser.add_argument("--all-dtypes", action="store_true", help="every dtype in the op's config")
    args, extra = parser.parse_known_args()

    run_all = args.all
    operator = args.operator
    for token in extra:
        cand = token.lstrip("-")
        if cand == "all":
            run_all = True
        elif (OPERATORS_ROOT / cand).is_dir():
            operator = cand

    if run_all:
        operators = all_operators()
    elif operator:
        operators = [operator]
    else:
        parser.error("give --<operator>, --operator <name>, or --all")

    failures = []
    for op in operators:
        if args.all_dtypes:
            dtypes = config_dtypes(op) or [None]
        elif args.dtype:
            dtypes = [args.dtype]
        else:
            dtypes = [None]
        for dt in dtypes:
            label = f"{op}:{dt}" if dt else op
            print(f"\n// ===== {label} =====", file=sys.stderr)
            try:
                process(op, dt)
            except Exception as e:
                failures.append((label, e))
                print(f"// {label}: FAILED ({type(e).__name__}: {e})", file=sys.stderr)

    if failures:
        print(f"\n// {len(failures)} failed:", file=sys.stderr)
        for op, e in failures:
            print(f"//   {op}: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
