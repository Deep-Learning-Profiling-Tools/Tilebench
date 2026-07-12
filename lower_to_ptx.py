"""Dump the PTX (or best available ISA) of an operator's kernels.

Usage:
    PYTHONPATH=. python lower_to_ptx.py --<operator_name>          # e.g. --mul2
    PYTHONPATH=. python lower_to_ptx.py --operator <operator_name>
    PYTHONPATH=. python lower_to_ptx.py --all                      # every operator

For the first benchmark case of the operator (same case the benchmark runs):
  - TileLang: runs impl_tilelang.run() with its default config and exports the
    PTX of every TileLang kernel it compiles (some operators launch several).
  - Triton: runs impl_triton.run() with autotune off and harvests PTX of the
    Triton kernels written to the Triton cache during that run.
  - cuTile: runs impl_cutile.run() with autotune off, captures every cubin
    compiled at launch, and dumps SASS via cuobjdump. cuTile's tileiras path
    emits SASS-only cubins (no embedded PTX), so files are written as
    cutile_<name>.sass (and the cubin is kept alongside as .cubin).
  - Torch: torch.compile()s impl_torch.run() and collects the PTX of the
    Triton kernels Inductor generates. Eager torch kernels ship inside
    libtorch as SASS-only fatbinaries (no PTX to extract), so Inductor is the
    only way to get real PTX for the torch implementation; ops that lower to
    library calls (cuBLAS, SDPA, ...) have no PTX — for those the launched
    kernel names are reported instead.

Artifacts are printed to stdout and also written to results/ptx/<operator>/.
"""

import argparse
import importlib
import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OPERATORS_ROOT = REPO_ROOT / "benchmarks" / "operators"

# Inductor / hand-written Triton write build artifacts (including .ptx) to the
# Triton cache; pointing it at a fresh dir isolates this run's kernels. Must
# happen before torch/triton are imported.
_TRITON_CACHE = tempfile.mkdtemp(prefix="lower_to_ptx_triton_")
os.environ["TRITON_CACHE_DIR"] = _TRITON_CACHE

import torch  # noqa: E402
import yaml  # noqa: E402

from core.dtypes import resolve_dtype  # noqa: E402
from data.tensors import expand_cases, get_generator  # noqa: E402


def first_benchmark_case(operator):
    with (OPERATORS_ROOT / operator / "config.yaml").open("r") as f:
        config = yaml.safe_load(f)
    case = expand_cases(operator, config)[0]
    params = {k: v for k, v in case.items() if k not in ("dtype", "block_size")}
    dtype = resolve_dtype(case.get("dtype", "fp32"))
    inputs = get_generator(operator)(**params, dtype=dtype)
    return case, inputs


def run_kwargs(fn, block_size=1024):
    sig = inspect.signature(fn).parameters
    kw = {}
    if "block_size" in sig:
        kw["block_size"] = block_size
    if "autotune" in sig:
        kw["autotune"] = False
    return kw


def _try_import(operator, impl_name):
    """Import benchmarks.operators.<op>.<impl_name>; return module or None."""
    try:
        return importlib.import_module(f"benchmarks.operators.{operator}.{impl_name}")
    except ImportError as e:
        print(f"// {impl_name}: import skipped ({e})", file=sys.stderr)
        return None


def _snapshot_ptx_paths():
    return set(Path(_TRITON_CACHE).rglob("*.ptx"))


def _harvest_new_ptx(before):
    """New *.ptx files under TRITON_CACHE since `before` (deduped by stem)."""
    ptx = {}
    for path in sorted(Path(_TRITON_CACHE).rglob("*.ptx")):
        if path in before:
            continue
        ptx.setdefault(path.stem, path.read_text())
    return list(ptx.items())


def _ptx_entry_name(ptx, fallback):
    match = re.search(r"\.entry\s+(\w+)", ptx)
    return match.group(1) if match else fallback


def collect_tilelang_ptx(operator, inputs):
    """Run impl_tilelang once, capturing every JITKernel it compiles."""
    module = _try_import(operator, "impl_tilelang")
    if module is None:
        return []

    from tilelang.jit.kernel import JITKernel

    captured = []
    orig_init = JITKernel.__init__

    def spy_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        captured.append(self)

    JITKernel.__init__ = spy_init
    try:
        module.run(*inputs, **run_kwargs(module.run))
        torch.cuda.synchronize()
    finally:
        JITKernel.__init__ = orig_init

    out = []
    for kernel in captured:
        ptx = kernel._get_ptx()
        name = _ptx_entry_name(ptx, f"kernel{len(out)}")
        out.append((name, ptx))
    return out


def collect_triton_ptx(operator, inputs):
    """Run impl_triton once and harvest PTX written to the Triton cache."""
    module = _try_import(operator, "impl_triton")
    if module is None:
        return []

    before = _snapshot_ptx_paths()
    module.run(*inputs, **run_kwargs(module.run))
    torch.cuda.synchronize()
    return _harvest_new_ptx(before)


def _find_tool(name):
    path = shutil.which(name)
    if path:
        return path
    for base in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-13.0",
    ):
        if not base:
            continue
        candidate = Path(base) / "bin" / name
        if candidate.is_file():
            return str(candidate)
    return None


def _cubin_to_sass(cubin_bytes, symbol=None):
    """Disassemble a cubin with cuobjdump --dump-sass. Returns SASS text or None."""
    cuobjdump = _find_tool("cuobjdump")
    if cuobjdump is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin_bytes)
        cubin_path = f.name
    try:
        result = subprocess.run(
            [cuobjdump, "--dump-sass", cubin_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        text = result.stdout
        if symbol and symbol not in text:
            # Still return full dump; multi-symbol cubins are rare for cutile.
            pass
        return text
    finally:
        try:
            os.unlink(cubin_path)
        except OSError:
            pass


def _cubin_to_ptx(cubin_bytes):
    """Try to extract embedded PTX from a cubin (usually empty for cuTile)."""
    cuobjdump = _find_tool("cuobjdump")
    if cuobjdump is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(cubin_bytes)
        cubin_path = f.name
    try:
        result = subprocess.run(
            [cuobjdump, "--dump-ptx", cubin_path],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        # cuobjdump prints a short header even when there is no PTX section.
        body = result.stdout
        if ".version" not in body and ".entry" not in body:
            return None
        return body
    finally:
        try:
            os.unlink(cubin_path)
        except OSError:
            pass


def collect_cutile_isa(operator, inputs):
    """Run impl_cutile once, capturing cubins; return (name, kind, text, cubin).

    kind is "ptx" if embedded PTX exists, else "sass". cubin is raw bytes
    (always available when capture succeeds).
    """
    module = _try_import(operator, "impl_cutile")
    if module is None:
        return []

    try:
        from cuda.tile._execution import kernel as CutileKernel
    except ImportError as e:
        print(f"// cutile: cuda.tile not available ({e})", file=sys.stderr)
        return []

    captured = []
    orig_compile = CutileKernel._compile

    def spy_compile(self, signature, context):
        result = orig_compile(self, signature, context)
        # result is (cubin, symbol, ..., ...)
        cubin, symbol = result[0], result[1]
        if cubin is not None:
            captured.append((symbol or f"kernel{len(captured)}", cubin))
        return result

    CutileKernel._compile = spy_compile
    try:
        module.run(*inputs, **run_kwargs(module.run))
        torch.cuda.synchronize()
    finally:
        CutileKernel._compile = orig_compile

    out = []
    for symbol, cubin in captured:
        ptx = _cubin_to_ptx(cubin)
        if ptx:
            out.append((symbol, "ptx", ptx, cubin))
            continue
        sass = _cubin_to_sass(cubin, symbol=symbol)
        if sass is None:
            sass = (
                f"// cuTile cubin for {symbol}: no PTX embedded and "
                f"cuobjdump --dump-sass failed (is CUDA toolkit on PATH?)\n"
            )
        out.append((symbol, "sass", sass, cubin))
    return out


def collect_torch_ptx(operator, inputs):
    """torch.compile impl_torch and harvest Inductor's Triton PTX."""
    module = _try_import(operator, "impl_torch")
    if module is None:
        return []

    before = _snapshot_ptx_paths()
    compiled = torch.compile(module.run)
    compiled(*inputs)
    torch.cuda.synchronize()
    return _harvest_new_ptx(before)


def eager_kernel_names(operator, inputs):
    """Names of the kernels eager torch actually launches (for extern ops)."""
    from torch.profiler import ProfilerActivity, profile

    module = importlib.import_module(f"benchmarks.operators.{operator}.impl_torch")
    module.run(*inputs)  # warm up so profiling sees steady-state kernels
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        module.run(*inputs)
        torch.cuda.synchronize()
    return [
        e.key for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA
    ]


def _safe_name(name):
    safe = re.sub(r"[^\w.\-]+", "_", name).strip("_")
    return safe or "kernel"


def emit(out_dir, backend, name, text, ext="ptx"):
    path = out_dir / f"{backend}_{_safe_name(name)}.{ext}"
    path.write_text(text)
    print(f"// ==================== {backend}: {name} ({path}) ====================")
    print(text)
    return path


def emit_bytes(out_dir, backend, name, data, ext):
    path = out_dir / f"{backend}_{_safe_name(name)}.{ext}"
    path.write_bytes(data)
    return path


def all_operators():
    """Every operator with a config.yaml under the operators root, sorted."""
    return sorted(
        p.name
        for p in OPERATORS_ROOT.iterdir()
        if p.is_dir() and (p / "config.yaml").is_file()
    )


def process_operator(operator):
    case, inputs = first_benchmark_case(operator)
    print(f"// operator: {operator}", file=sys.stderr)
    print(f"// benchmark case: {case}", file=sys.stderr)

    out_dir = REPO_ROOT / "results" / "ptx" / operator
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # --- TileLang ---
    for name, ptx in collect_tilelang_ptx(operator, inputs):
        written.append(emit(out_dir, "tilelang", name, ptx))

    # --- Triton (hand-written) ---
    triton_kernels = collect_triton_ptx(operator, inputs)
    if triton_kernels:
        for name, ptx in triton_kernels:
            # Prefer .entry name when present; stem is a stable fallback.
            entry = _ptx_entry_name(ptx, name)
            written.append(emit(out_dir, "triton", entry, ptx))
    else:
        print("// triton: no PTX produced (import failed or no kernels compiled)")

    # --- cuTile ---
    cutile_kernels = collect_cutile_isa(operator, inputs)
    if cutile_kernels:
        for name, kind, text, cubin in cutile_kernels:
            written.append(emit_bytes(out_dir, "cutile", name, cubin, "cubin"))
            if kind == "ptx":
                written.append(emit(out_dir, "cutile", name, text, ext="ptx"))
            else:
                print(
                    f"// cutile: {name} is SASS-only (tileiras → cubin, no PTX section)"
                )
                written.append(emit(out_dir, "cutile", name, text, ext="sass"))
    else:
        print("// cutile: no kernels captured (import failed or no compile)")

    # --- Torch (Inductor / Triton) ---
    torch_kernels = collect_torch_ptx(operator, inputs)
    if torch_kernels:
        for name, ptx in torch_kernels:
            entry = _ptx_entry_name(ptx, name)
            written.append(emit(out_dir, "torch", entry, ptx))
    else:
        try:
            names = eager_kernel_names(operator, inputs)
        except Exception as e:
            names = []
            print(f"// torch: Inductor emitted no Triton PTX; profiler failed ({e})")
        else:
            print(
                "// torch: Inductor emitted no Triton kernels for this operator "
                "(it lowers to precompiled library calls, which ship as SASS-only "
                "binaries with no PTX). Kernels actually launched by eager torch:"
            )
            for n in names:
                print(f"//   {n}")
        path = out_dir / "torch_kernels.txt"
        path.write_text("\n".join(names) + ("\n" if names else ""))
        written.append(path)

    print("// written:", file=sys.stderr)
    for path in written:
        print(f"//   {path}", file=sys.stderr)
    return written


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--operator", help="operator name (or pass --<operator_name>)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="lower every operator (equivalent to `--all`/`-all`)",
    )
    args, extra = parser.parse_known_args()

    run_all = args.all
    operator = args.operator
    for token in extra:  # support the `--<operator_name>` / `-all` spelling
        candidate = token.lstrip("-")
        if candidate == "all":
            run_all = True
        elif (OPERATORS_ROOT / candidate).is_dir():
            operator = candidate

    if run_all:
        operators = all_operators()
    elif operator:
        if not (OPERATORS_ROOT / operator).is_dir():
            parser.error(f"unknown operator '{operator}'; see {OPERATORS_ROOT}")
        operators = [operator]
    else:
        parser.error("no operator given; use --operator <name>, --<name>, or --all")

    failures = []
    for op in operators:
        print(f"\n// #################### operator: {op} ####################",
              file=sys.stderr)
        try:
            process_operator(op)
        except Exception as e:  # keep going so one bad op doesn't abort --all
            failures.append((op, e))
            print(f"// {op}: FAILED ({type(e).__name__}: {e})", file=sys.stderr)

    if failures:
        print(f"\n// {len(failures)} operator(s) failed:", file=sys.stderr)
        for op, e in failures:
            print(f"//   {op}: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
