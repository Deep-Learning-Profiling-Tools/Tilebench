"""Record the arguments each kernel is launched with, for scripts/export_ir.py.

For one operator this builds the benchmark's own inputs for the largest
autotuned shape of every dtype, installs each backend's logged autotune winner
as the implementation's default configuration, and calls the implementation's
run() with the kernel launch replaced by a recorder. Nothing is compiled or
launched: the recorder notes the kernel and its arguments and returns. The
recorded arguments are written to scripts/ir_cases.json in the form the IR
generators take.

    python scripts/capture_ir_cases.py --checkout <tilebench checkout> --winners <dir> --only relu

<dir> holds <op>_autotune.json and <op>_tilelang_autotune.json for the
platform. Needs a CUDA device, because the benchmark's input generators and
run() functions create device tensors.
"""

import argparse
import importlib
import inspect
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TRITON_DTYPES = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32", "float64": "fp64", "int8": "i8",
                 "uint8": "u8", "int16": "i16", "int32": "i32", "int64": "i64", "bool": "i1",
                 "float8_e4m3fn": "fp8e4nv", "float8_e5m2": "fp8e5"}


def dtype_name(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def shape_text(tensor):
    return "x".join(str(d) for d in tensor.shape) + ":" + dtype_name(tensor)


def scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    return value


def winners(directory, op):
    records = []
    for name in (f"{op}_autotune.json", f"{op}_tilelang_autotune.json"):
        path = directory / name
        if path.exists():
            records += json.loads(path.read_text())
    by_dtype = {}
    for record in records:
        entry = by_dtype.setdefault((record["dtype"], json.dumps(record["params"], sort_keys=True)),
                                    {"dtype": record["dtype"], "params": record["params"],
                                     "problem_size": record.get("problem_size") or 0, "configs": {}})
        for backend in ("triton", "cutile", "tilelang"):
            if record.get(f"{backend}_autotune_cfg"):
                entry["configs"][backend] = record[f"{backend}_autotune_cfg"]
    largest = {}
    for entry in by_dtype.values():
        current = largest.get(entry["dtype"])
        if current is None or entry["problem_size"] >= current["problem_size"]:
            largest[entry["dtype"]] = entry
    return largest


def merged(default, config):
    if isinstance(default, dict):
        return {**default, **config}
    return SimpleNamespace(**{**vars(default), **config})


def install_config(module, config):
    saved = {}
    if getattr(module, "_DEFAULT_CONFIG", None) is not None:
        saved["_DEFAULT_CONFIG"] = module._DEFAULT_CONFIG
        module._DEFAULT_CONFIG = merged(module._DEFAULT_CONFIG, config)
    if isinstance(getattr(module, "_DEFAULT_CONFIGS", None), dict):
        saved["_DEFAULT_CONFIGS"] = module._DEFAULT_CONFIGS
        module._DEFAULT_CONFIGS = {key: merged(value, config) for key, value in module._DEFAULT_CONFIGS.items()}
    for name, value in list(vars(module).items()):
        part = re.fullmatch(r"_DEFAULT_([A-Z0-9]+)_CONFIG", name)
        if part and value is not None:
            prefix = part.group(1).lower() + "_"
            own = {key[len(prefix):]: item for key, item in config.items() if key.lower().startswith(prefix)}
            if own:
                saved[name] = value
                setattr(module, name, merged(value, own))
    return saved


def apply_winner(case, values, config):
    """An autotuned kernel takes the logged winner even where run()'s default path passes something else."""
    changed = []
    for key, value in config.items():
        if key in values and values[key] != value:
            changed.append(f"{key}: run() passed {values[key]}, winner is {value}")
            values[key] = value
    if changed:
        case["config_note"] = "taken from the winner log, not from run()'s default path: " + "; ".join(changed)


def capture_triton(module, torch, call, config):
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction
    records, saved = [], {}
    def unwrap(value):
        heuristics = {}
        while not isinstance(value, JITFunction) and hasattr(value, "fn"):
            heuristics.update(getattr(value, "values", None) or {})
            value = value.fn
        return (value, heuristics) if isinstance(value, JITFunction) else (None, {})

    tuned = {id(unwrap(value)[0]) for value in vars(module).values() if isinstance(value, Autotuner)}

    class Recorder:
        def __init__(self, name, kernel, heuristics):
            self.name, self.kernel, self.heuristics = name, kernel, heuristics

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                bound = dict(zip(self.kernel.arg_names, args))
                bound.update(kwargs)
                for key, rule in self.heuristics.items():
                    bound[key] = rule(bound)
                records.append((self.name, self.kernel, bound))
            return launch

        def __getattr__(self, item):
            return getattr(self.kernel, item)

    for name, value in list(vars(module).items()):
        if isinstance(value, Autotuner):
            continue
        kernel, heuristics = unwrap(value)
        if kernel is not None:
            saved[name] = value
            setattr(module, name, Recorder(name, kernel, heuristics))
    try:
        call()
    except Exception as error:
        if not records:
            raise
        print(f"   triton run stopped after {len(records)} launch(es): {type(error).__name__}")
    finally:
        for name, value in saved.items():
            setattr(module, name, value)
    cases = []
    for name, kernel, bound in records:
        args, note = [], {}
        if id(kernel) in tuned:
            apply_winner(note, bound, config)
        for parameter in kernel.params:
            if parameter.name not in bound:
                raise ValueError(f"{name}: no value recorded for parameter {parameter.name}")
            value = bound[parameter.name]
            if isinstance(value, torch.Tensor):
                args.append(f"{parameter.name}=ptr:{TRITON_DTYPES[dtype_name(value)]}")
            elif type(value).__name__ == "TensorDescriptor":
                block = "x".join(str(d) for d in value.block_shape)
                args.append(f"{parameter.name}=desc:{TRITON_DTYPES[dtype_name(value.base)]}:{block}")
            else:
                value = scalar(value)
                if type(value).__name__ == "dtype" and hasattr(value, "name"):
                    args.append(f"{parameter.name}=type:{value.name}")
                elif parameter.is_constexpr:
                    args.append(f"{parameter.name}=const:{json.dumps(value)}")
                elif isinstance(value, bool) or isinstance(value, int):
                    args.append(f"{parameter.name}=int:{int(value)}")
                else:
                    args.append(f"{parameter.name}=float:{value!r}")
        case = {"kernel": name, "args": args, "num_warps": int(bound.get("num_warps", 4)), **note}
        if bound.get("num_stages") is not None:
            case["num_stages"] = int(bound["num_stages"])
        cases.append(case)
    return cases


def capture_tilelang(module, torch, call, config):
    records, saved = [], {}

    class Recorder:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __call__(self, *args, **kwargs):
            records.append((self.name, self.kernel, args, kwargs))
            return lambda *a, **k: None

        def compile(self, *args, **kwargs):
            return self(*args, **kwargs)

        def __getattr__(self, item):
            return getattr(self.kernel, item)

    for name, value in list(vars(module).items()):
        if type(value).__name__ in ("AutoTuneImpl", "JITImpl"):
            saved[name] = value
            setattr(module, name, Recorder(name, value))
    try:
        call()
    except Exception as error:
        if not records:
            raise
        print(f"   tilelang run stopped after {len(records)} launch(es): {type(error).__name__}")
    finally:
        for name, value in saved.items():
            setattr(module, name, value)
    cases = []
    for name, kernel, args, kwargs in records:
        function = getattr(getattr(kernel, "jit_impl", kernel), "func")
        names = list(inspect.signature(function).parameters)
        tensors, values, seen_scalar = [], {}, False
        for index, value in enumerate(args):
            if isinstance(value, torch.Tensor):
                if seen_scalar:
                    raise ValueError(f"{name}: a tensor follows a scalar positional argument")
                tensors.append(shape_text(value))
            else:
                seen_scalar = True
                values[names[index]] = scalar(value)
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                raise ValueError(f"{name}: tensor passed by keyword ({key})")
            values[key] = scalar(value)
        case = {"kernel": name}
        if type(kernel).__name__ == "AutoTuneImpl":
            apply_winner(case, values, config)
        kw = [f"{key}={value if isinstance(value, str) else json.dumps(value)}" for key, value in values.items()]
        cases.append({**case, "tensors": tensors, "kw": kw})
    return cases


def capture_cutile(module, torch, call, config):
    import cuda.tile as ct
    records, original = [], ct.launch

    def launch(stream, grid, kernel, args, *rest, **kwargs):
        records.append((kernel, tuple(args)))

    ct.launch = launch
    try:
        call()
    except Exception as error:
        if not records:
            raise
        print(f"   cutile run stopped after {len(records)} launch(es): {type(error).__name__}")
    finally:
        ct.launch = original
    cases = []
    for kernel, args in records:
        name = kernel._annotated_function.pyfunc.__name__
        if not hasattr(module, name):
            raise ValueError(f"cuTile kernel {name} is not a module-level name")
        out = []
        for value in args:
            if isinstance(value, torch.Tensor):
                if not value.is_contiguous() or value.data_ptr() % 16:
                    raise ValueError(f"{name}: array argument is not contiguous and 16-byte aligned")
                out.append(shape_text(value))
            else:
                out.append(json.dumps(scalar(value)))
        cases.append({"kernel": name, "args": out})
    return cases


CAPTURE = {"triton": capture_triton, "tilelang": capture_tilelang, "cutile": capture_cutile}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--winners", type=Path, required=True)
    parser.add_argument("--platform", default="B200")
    parser.add_argument("--only", required=True, help="operator to capture")
    parser.add_argument("--cases", type=Path, default=ROOT / "scripts" / "ir_cases.json")
    parser.add_argument("--max-variants", type=int, default=4,
                        help="distinct argument sets to keep for a kernel that is launched several times")
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout))
    import torch
    from tilebench.core.dtypes import resolve_dtype
    from tilebench.data.tensors import get_generator
    op = args.only
    document = json.loads(args.cases.read_text()) if args.cases.exists() else {"cases": []}
    existing = {(c["platform"], c["op"], c["backend"], c["dtype"], c["kernel"], c.get("variant", 1)): c for c in document["cases"]}
    captured = failed = 0
    for dtype, entry in sorted(winners(args.winners, op).items()):
        try:
            inputs = get_generator(op)(**entry["params"], dtype=resolve_dtype(dtype))
        except Exception as error:
            print(f"SKIP {op}/{dtype}: inputs not generated ({type(error).__name__}: {str(error)[:80]})")
            continue
        for backend, config in sorted(entry["configs"].items()):
            label = f"{args.platform}/{op}/{backend}/{dtype}"
            try:
                module = importlib.import_module(f"tilebench.benchmarks.operators.{op}.impl_{backend}")
                saved = install_config(module, config)
                if not saved:
                    raise ValueError("implementation has no _DEFAULT_CONFIG to carry the winner")
                run_parameters = inspect.signature(module.run).parameters
                extra = {"autotune": False} if "autotune" in run_parameters else {}
                try:
                    found = CAPTURE[backend](module, torch, lambda: module.run(*inputs, **extra), config)
                finally:
                    for name, value in saved.items():
                        setattr(module, name, value)
            except Exception as error:
                failed += 1
                print(f"FAILED {label}: {type(error).__name__}: {str(error)[:140]}")
                continue
            unique, seen, fresh = [], {}, set()
            for case in found:
                if any(case == {k: v for k, v in other.items() if k != "variant"} for other in unique):
                    continue
                seen[case["kernel"]] = seen.get(case["kernel"], 0) + 1
                if seen[case["kernel"]] <= args.max_variants:
                    unique.append({**case, "variant": seen[case["kernel"]]})
            for case in unique:
                case = {"platform": args.platform, "op": op, "dtype": dtype, "backend": backend, **case,
                        "shape": entry["params"], "config": config}
                if seen[case["kernel"]] > 1:
                    case["launch_variants"] = seen[case["kernel"]]
                key = (args.platform, op, backend, dtype, case["kernel"], case["variant"])
                previous = existing.get(key)
                if previous is not None:
                    same = all(previous.get(k) == case.get(k) for k in ("args", "tensors", "kw", "num_warps", "num_stages"))
                    same = same and not case.get("config_note")
                    if same:
                        for keep in ("verified", "note"):
                            if previous.get(keep):
                                case[keep] = previous[keep]
                    else:
                        print(f"   differs from the existing case for {label}/{case['kernel']}")
                existing[key] = case
                fresh.add(key)
                captured += 1
            for key in [k for k in existing if k[:4] == (args.platform, op, backend, dtype) and k not in fresh]:
                del existing[key]
            print(f"ok {label}: {', '.join(sorted(seen)) or 'no launch recorded'}")
        del inputs
        torch.cuda.empty_cache()
    document["cases"] = [existing[k] for k in sorted(existing)]
    args.cases.write_text(json.dumps(document, indent=1) + "\n")
    print(f"{op}: {captured} cases recorded, {failed} backend captures failed")


if __name__ == "__main__":
    main()
