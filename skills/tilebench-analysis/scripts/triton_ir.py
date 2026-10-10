"""Get the IR Triton produces for one kernel specialisation.

Compiles the kernel from its source for an explicit target (default cuda:100)
with the argument types given by --arg, and writes
<kernel>.{ttir,ttgir,llir,ptx,json} to --out (amdgcn in place of ptx for an AMD
target). No GPU is needed and nothing is launched.
"""

import argparse
import atexit
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


def describe(directory, kernel, meta):
    text = (directory / f"{kernel}.ttgir").read_text()
    target = re.search(r'ttg\.target = "([^"]+)"', text)
    return {"num_warps": meta.get("num_warps"), "num_stages": meta.get("num_stages"), "shared": meta.get("shared"),
            "target": target.group(1) if target else None, "ttgir_lines": text.count("\n"),
            "convert_layout": text.count("ttg.convert_layout")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("impl", type=Path, help="impl_triton.py of the operator")
    parser.add_argument("kernel", help="name of the @triton.jit function in that file")
    parser.add_argument("--arg", action="append", default=[], metavar="NAME=KIND:VALUE",
                        help="one per kernel parameter: ptr:i8 | desc:fp16:256x64 | int:20480 | float:1e-5 | const:2048 | type:fp32 "
                             "(pass runtime integers with their real value: it decides the divisibility and equal-to-1 specialisation)")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int)
    parser.add_argument("--hardware", default="B200",
                        help="TileBench hardware the reports were captured on: B200 (cuda:100), GH200 (cuda:90) or MI300X (hip:gfx942). "
                             "Any other GPU: its architecture, e.g. sm_89 or hip:gfx950")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    args.target = {"B200": "cuda:100", "GH200": "cuda:90", "MI300X": "hip:gfx942"}.get(args.hardware, args.hardware)
    if args.target.startswith("sm_"):
        args.target = "cuda:" + args.target[3:].rstrip("af")
    if ":" not in args.target:
        parser.error(f"--hardware {args.hardware}: use B200, GH200, MI300X, an sm_NN architecture or backend:arch")
    cache = tempfile.mkdtemp(prefix="triton_ir_")
    atexit.register(shutil.rmtree, cache, ignore_errors=True)
    os.environ["TRITON_CACHE_DIR"] = cache
    try:
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
    except ImportError as error:
        parser.error(f"Triton not importable in this interpreter: {error}")
    impl = args.impl.resolve()
    root = next((p for p in impl.parents if (p / "tilebench").is_dir()), None)
    if root:
        sys.path.insert(0, str(root))
    if root and impl.is_relative_to(root):
        module = importlib.import_module(".".join(impl.relative_to(root).with_suffix("").parts))
    else:
        spec = importlib.util.spec_from_file_location(impl.stem, impl)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    kernel = getattr(module, args.kernel)
    while not hasattr(kernel, "params") and hasattr(kernel, "fn"):
        kernel = kernel.fn
    given = dict(item.split("=", 1) for item in args.arg)
    missing = [name for name in kernel.arg_names if name not in given]
    if missing:
        parser.error(f"--arg needed for every kernel parameter; missing {missing} (parameters: {kernel.arg_names})")
    signature, constexprs, attrs = {}, {}, {}
    for index, name in enumerate(kernel.arg_names):
        kind, _, value = given[name].partition(":")
        aliases = {"int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64", "float16": "fp16", "bfloat16": "bf16",
                   "float32": "fp32", "float64": "fp64"}
        if kind == "ptr":
            signature[name] = "*" + aliases.get(value, value)
            attrs[(index,)] = [["tt.divisibility", 16]]
        elif kind == "desc":
            dtype, _, shape = value.partition(":")
            signature[name] = f"tensordesc<{aliases.get(dtype, dtype)}{[int(d) for d in shape.split('x')]}>"
        elif kind == "type":
            signature[name] = "constexpr"
            constexprs[name] = triton.language.core.dtype(aliases.get(value, value))
        elif kind == "const" or (kind == "int" and int(value) == 1):
            signature[name] = "constexpr"
            constexprs[name] = json.loads(value)
        elif kind == "int":
            number = int(value)
            signature[name] = "i32" if -2**31 <= number < 2**31 else "i64"
            if number % 16 == 0:
                attrs[(index,)] = [["tt.divisibility", 16]]
        elif kind == "float":
            signature[name] = "fp32"
        else:
            parser.error(f"--arg {name}: use ptr:<dtype>, desc:<dtype>:<AxB>, int:<value>, float:<value>, const:<json> or type:<dtype>")
    backend, _, arch = args.target.partition(":")
    options = {"num_warps": args.num_warps}
    if args.num_stages is not None:
        options["num_stages"] = args.num_stages
    target = GPUTarget(backend, int(arch), 32) if arch.isdigit() else GPUTarget(backend, arch, 64)
    compiled = triton.compile(ASTSource(kernel, signature=signature, constexprs=constexprs, attrs=attrs), target=target, options=options)
    args.out.mkdir(parents=True, exist_ok=True)
    for extension in ("ttir", "ttgir", "llir", "ptx", "amdgcn"):
        if isinstance(compiled.asm.get(extension), str):
            (args.out / f"{args.kernel}.{extension}").write_text(compiled.asm[extension])
    fields = compiled.metadata._asdict() if hasattr(compiled.metadata, "_asdict") else vars(compiled.metadata)
    meta = {key: value for key, value in fields.items() if isinstance(value, (int, float, str, bool, type(None)))}
    (args.out / f"{args.kernel}.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({"kernel": args.kernel, "out": str(args.out), "triton": triton.__version__,
                      "signature": signature, "constexprs": constexprs, **describe(args.out, args.kernel, meta)}, indent=1, default=str))


if __name__ == "__main__":
    main()
