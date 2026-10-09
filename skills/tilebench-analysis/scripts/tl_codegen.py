"""Emit the CUDA source TileLang generates for one kernel configuration.

Lowers the kernel for the given shapes and config to CUDA source for an explicit
target architecture (default sm_100) and writes it. The source is taken before
it would be handed to nvcc, so neither a GPU nor nvcc is needed and nothing is
built or executed. TileBench operators that choose a kernel by device capability
see a device of that architecture. Requires the same TileLang version as the
experiment being analysed, since lowering changes between versions.

--tir also writes the kernel as TIR before TileLang's passes (the program as
written, with shapes and config bound). --lowered-tir writes the device TIR
after all of TileLang's passes, the form the CUDA is generated from. Diffing
the two shows what the passes did; the CUDA shows the final result.
"""

import argparse
import importlib
import inspect
import importlib.util
import json
from pathlib import Path
import sys


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("impl", type=Path, help="impl_tilelang.py of the operator")
    parser.add_argument("kernel", help="name of the @tilelang.jit function in that file")
    parser.add_argument("--tensor", action="append", default=[], metavar="SHAPE:DTYPE",
                        help="positional tensor argument, e.g. 10240x10240:float16; repeat in call order")
    parser.add_argument("--kw", action="append", default=[], metavar="NAME=VALUE",
                        help="keyword argument (shape scalars, dtype strings, winner config); repeat")
    parser.add_argument("--hardware", default="B200", help="TileBench hardware the reports were captured on: B200 (sm_100) or GH200 (sm_90). Any other GPU: its architecture, e.g. sm_89")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tir", type=Path, help="also write the TIR before TileLang's passes")
    parser.add_argument("--lowered-tir", type=Path, help="also write the device TIR after TileLang's passes")
    parser.add_argument("--ptx", type=Path, help="also write the PTX nvcc produces from the generated CUDA (needs nvcc)")
    args = parser.parse_args()
    args.arch = {"B200": "sm_100", "GH200": "sm_90"}.get(args.hardware, args.hardware)
    if not args.arch.startswith("sm_"):
        parser.error(f"--hardware {args.hardware}: this backend targets NVIDIA only; use B200, GH200 or an sm_NN architecture")
    try:
        import torch
        import tilelang
        import tilelang.contrib.nvcc as nvcc
        from tvm.target import Target
    except ImportError as error:
        parser.error(f"TileLang toolchain not importable in this interpreter: {error}")
    impl = args.impl.resolve()
    root = next((p for p in impl.parents if (p / "tilebench").is_dir()), None)
    if root:
        sys.path.insert(0, str(root))
    digits = args.arch.split("_")[1].rstrip("af")
    try:
        from tilebench import hardware
        forced = hardware.DeviceInfo("nvidia", f"forced {args.arch}", (int(digits[:-1]), int(digits[-1])), None)
        hardware.device_info = lambda: forced
    except ImportError:
        pass
    if root and impl.is_relative_to(root):
        module = importlib.import_module(".".join(impl.relative_to(root).with_suffix("").parts))
    else:
        spec = importlib.util.spec_from_file_location(impl.stem, impl)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    tensors = []
    for item in args.tensor:
        shape, dtype = item.split(":")
        tensors.append(torch.zeros([int(d) for d in shape.split("x")], dtype=getattr(torch, dtype)))
    kwargs = dict(item.split("=", 1) for item in args.kw)
    kwargs = {k: parse_value(v) for k, v in kwargs.items()}
    captured, calls = [], []
    compile_cuda = nvcc.compile_cuda

    def capture(code, *unused, **ignored):
        captured.append(code)
        calls.append((unused, ignored))
        raise RuntimeError("source captured")

    nvcc.compile_cuda = capture
    lowering = importlib.import_module("tilelang.engine.lower")
    lowered = []
    for name in ("device_codegen", "device_codegen_without_compile"):
        original = getattr(lowering, name)

        def record(device_mod, target, _original=original):
            lowered.append(str(device_mod))
            return _original(device_mod, target)

        setattr(lowering, name, record)
    entry = getattr(module, args.kernel)
    pending, seen = [entry], set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if hasattr(item, "target"):
            item.target = Target({"kind": "cuda", "arch": args.arch})
        pending += [getattr(item, name) for name in ("jit_impl", "fn", "func", "_jit", "__wrapped__") if hasattr(item, name)]
    try:
        source = entry.compile(*tensors, **kwargs).get_kernel_source()
    except Exception:
        if not captured:
            raise
        source = captured[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source)
    written = {}
    if args.ptx:
        if not calls:
            parser.error("the nvcc call was not captured by this TileLang version's compile path")
        bound = inspect.signature(compile_cuda).bind(source, *calls[0][0], **calls[0][1])
        bound.arguments.update(target_format="ptx", path_target=str(args.ptx.resolve()))
        if not bound.arguments.get("arch"):
            bound.arguments["arch"] = args.arch
        args.ptx.parent.mkdir(parents=True, exist_ok=True)
        compile_cuda(*bound.args, **bound.kwargs)
        written["ptx"] = str(args.ptx)
    if args.lowered_tir:
        if not lowered:
            parser.error("the lowered TIR was not produced by this TileLang version's compile path")
        args.lowered_tir.parent.mkdir(parents=True, exist_ok=True)
        args.lowered_tir.write_text(lowered[0])
        written["lowered_tir"] = str(args.lowered_tir)
    if args.tir:
        traced = next((item for item in [entry] + [getattr(entry, name) for name in ("jit_impl", "fn", "func", "_jit", "__wrapped__") if hasattr(entry, name)]
                       if hasattr(item, "get_tir")), None)
        if traced is None:
            parser.error("this kernel object does not expose get_tir")
        args.tir.parent.mkdir(parents=True, exist_ok=True)
        args.tir.write_text(str(traced.get_tir(*tensors, **kwargs)))
        written["tir"] = str(args.tir)
    start = source.rfind("extern", 0, source.find("__launch_bounds__"))
    print(json.dumps({"out": str(args.out), **written, "tilelang": tilelang.__version__, "arch": args.arch,
                      "kernel_body_lines": source[start:].count("\n")}))


if __name__ == "__main__":
    main()
