"""Print the front-end IR cuTile builds for one kernel specialisation.

Runs cuTile's Python front end and passes for the given argument shapes and
constants and writes the resulting Tile IR as text. The kernel signature is
built explicitly from the shapes, so no GPU, NVIDIA driver or tileiras binary is
needed, no cubin is produced and nothing is launched. Array arguments are
described the way cuTile describes a contiguous, 16-byte-aligned array passed
at launch: unit inner stride, and shape, outer strides and base address assumed
divisible by what the given shape allows (up to 16 bytes).
"""

import argparse
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("impl", type=Path, help="impl_cutile.py of the operator")
    parser.add_argument("kernel", help="name of the @ct.kernel function in that file")
    parser.add_argument("--arg", action="append", default=[], metavar="SHAPE:DTYPE|VALUE",
                        help="kernel argument in call order: 10240x10240:float16 for an array, "
                             "or a JSON scalar such as 8192; repeat")
    parser.add_argument("--hardware", default="B200", help="TileBench hardware the reports were captured on: B200 (sm_100) or GH200 (sm_90). Any other GPU: its architecture, e.g. sm_89")
    parser.add_argument("--bytecode-version", help="e.g. 13.2; default is the newest this cuda-tile supports")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.arch = {"B200": "sm_100", "GH200": "sm_90"}.get(args.hardware, args.hardware)
    if not args.arch.startswith("sm_"):
        parser.error(f"--hardware {args.hardware}: this backend targets NVIDIA only; use B200, GH200 or an sm_NN architecture")
    try:
        import cuda.tile as ct
        from cuda.tile import _compile, compilation
        from cuda.tile.compilation import _signature
    except ImportError as error:
        parser.error(f"cuTile toolchain not importable in this interpreter: {error}")
    root = next((p for p in args.impl.resolve().parents if (p / "tilebench").is_dir()), None)
    if root:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("ct_impl", args.impl)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = getattr(module, args.kernel)
    function = kernel._annotated_function.pyfunc
    names = list(inspect.signature(function).parameters)
    if len(names) != len(args.arg):
        parser.error(f"{args.kernel} takes {len(names)} arguments {names}; got {len(args.arg)} --arg")
    constraints = []
    for name, item in zip(names, args.arg):
        if ":" in item:
            shape, dtype = item.split(":")
            shape = [int(d) for d in shape.split("x")]
            element = getattr(ct, dtype)
            size = max(1, int("".join(c for c in dtype if c.isdigit()) or 8) // 8)
            strides = [math.prod(shape[i + 1:]) for i in range(len(shape))]
            constraints.append(_signature.ArrayConstraint(
                element, len(shape), index_dtype=ct.int32, stride_lower_bound_incl=[0] * len(shape),
                alias_groups=(), may_alias_internally=False,
                stride_constant=[None] * (len(shape) - 1) + [1],
                stride_divisible_by=[max(1, math.gcd(s * size, 16) // size) for s in strides],
                shape_divisible_by=[math.gcd(d, 16) for d in shape], base_addr_divisible_by=16))
            continue
        value = json.loads(item)
        if name in function.__annotations__:
            constraints.append(_signature.ConstantConstraint(value))
        elif isinstance(value, bool):
            constraints.append(_signature.ScalarConstraint(ct.bool_))
        elif isinstance(value, float):
            constraints.append(_signature.ScalarConstraint(ct.float32))
        else:
            constraints.append(_signature.ScalarConstraint(ct.int32 if -2**31 <= value < 2**31 else ct.int64))
    signature = compilation.KernelSignature(
        tuple(constraints), compilation.CallingConvention.cutile_python_v1(), None).with_mangled_symbol(function.__name__)
    versions = _compile._all_bytecode_versions(False)
    version = _compile.parse_bytecode_version(args.bytecode_version) if args.bytecode_version else versions[-1]
    result = _compile.compile_tile(kernel._annotated_function, [signature], sm_arch=args.arch,
                                   compiler_options=kernel._compiler_options, bytecode_version=version,
                                   return_final_ir=True, return_cubin=False)
    text = result.final_ir[0].to_string()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    ops = {}
    for line in text.splitlines():
        if "= " in line and "(" in line:
            name = line.split("= ", 1)[1].split("(", 1)[0].strip()
            ops[name] = ops.get(name, 0) + 1
    print(json.dumps({"out": str(args.out), "cuda_tile": ct.__version__, "arch": args.arch, "symbol": signature.symbol,
                      "lines": text.count("\n"), "ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])[:16])}, indent=1))


if __name__ == "__main__":
    main()
