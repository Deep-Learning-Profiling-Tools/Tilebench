"""Introspect the installed NKI package (no device needed).

Records, for nki, nki.language, nki.isa, nki.compiler and nki.collectives:
  - the public symbol list (``__all__`` and ``dir()``),
  - for every public symbol: kind, ``inspect.signature``, full docstring, defining
    module, source file and first line,
plus: enum members (nki.isa enums), dtype objects, ``nl.tile_size`` constants
(static attributes and, under a trn2 target, the nc-version dependent properties),
names that older public material uses and whether they exist in this install, and
the package versions.

Run:  /home/ubuntu/workspace/native_venv/bin/python introspect.py [out.json]
Writes introspection.json next to this file by default.
"""

import enum
import importlib
import importlib.metadata as md
import inspect
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "introspection.json")

# Same target selection as the evaluator; nothing here touches the device.
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")

import nki  # noqa: E402
import nki.collectives  # noqa: E402
import nki.compiler  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

MODULES = ["nki", "nki.language", "nki.isa", "nki.compiler", "nki.collectives"]
SITE = os.path.dirname(os.path.dirname(nki.__file__))


def _rel(path):
    if not path:
        return None
    try:
        return os.path.relpath(path, SITE)
    except ValueError:
        return path


def describe(obj):
    d = {"kind": type(obj).__name__}
    if isinstance(obj, enum.EnumMeta):
        d["kind"] = "enum"
        d["members"] = {m.name: repr(m.value) for m in obj}
    try:
        d["signature"] = str(inspect.signature(obj))
    except (TypeError, ValueError):
        d["signature"] = None
    doc = inspect.getdoc(obj)
    # Opcode sentinels / dtype objects share class docstrings; keep them short.
    if doc is not None and type(obj).__module__.startswith("nki") and not (
        inspect.isfunction(obj) or inspect.isclass(obj) or inspect.ismethod(obj)
    ):
        own = obj.__dict__.get("__doc__") if hasattr(obj, "__dict__") else None
        doc = own if own else None
    d["doc"] = doc
    target = inspect.unwrap(obj) if callable(obj) else obj
    try:
        d["module"] = getattr(target, "__module__", None)
        src = inspect.getsourcefile(target)
        d["source"] = _rel(src)
        d["line"] = inspect.getsourcelines(target)[1]
    except (TypeError, OSError):
        pass
    if not callable(obj) and not inspect.ismodule(obj):
        try:
            d["repr"] = repr(obj)[:200]
        except Exception as e:  # pragma: no cover
            d["repr"] = f"<repr failed: {e!r}>"
    return d


def module_record(name):
    mod = importlib.import_module(name)
    all_ = list(getattr(mod, "__all__", []))
    public = sorted(a for a in dir(mod) if not a.startswith("_"))
    symbols = {}
    for a in public:
        obj = getattr(mod, a)
        if inspect.ismodule(obj):
            symbols[a] = {"kind": "module", "module": obj.__name__}
            continue
        symbols[a] = describe(obj)
    return {
        "file": _rel(mod.__file__),
        "__all__": all_,
        "dir_public": public,
        "not_in___all__": [a for a in public if a not in all_],
        "symbols": symbols,
    }


def tile_size_record():
    ts = nl.tile_size
    static = {}
    for a in ["pmax", "psum_fmax", "psum_bank_fmax", "psum_bank_fmax_bytes",
              "gemm_stationary_fmax", "gemm_moving_fmax", "bn_stats_fmax",
              "psum_min_align", "sbuf_min_align"]:
        static[a] = getattr(type(ts), a)
    from nki.language import _tile_size as _ts
    tables = {
        "_SBUF_SIZE_BYTES": _ts._SBUF_SIZE_BYTES,
        "_SBUF_FMAX_BYTES": _ts._SBUF_FMAX_BYTES,
        "_PSUM_PARTITION_SIZE": _ts._PSUM_PARTITION_SIZE,
        "_PSUM_BANK_SIZE": _ts._PSUM_BANK_SIZE,
        "_PSUM_NUM_BANKS": _ts._PSUM_NUM_BANKS,
        "_MAX_PSUM_FREE_DIM": _ts._MAX_PSUM_FREE_DIM,
    }
    # The nc-version dependent properties need an active backend; evaluate them
    # outside a kernel to record what happens (they are documented from the tables).
    props = {}
    for a in ["sbuf_size_bytes", "sbuf_fmax", "sbuf_fmax_bytes", "psum_num_banks",
              "total_available_sbuf_size"]:
        try:
            props[a] = getattr(ts, a)
        except Exception as e:
            props[a] = f"<error outside kernel: {type(e).__name__}: {e}>"
    return {"static": static, "nc_version_tables": tables, "properties_outside_kernel": props}


# Names used by older public NKI material (Beta 1/2 docs, agentic skills, samples).
LEGACY_NAMES = {
    "nki.language": ["arange", "mgrid", "par_dim", "atomic_rmw", "load", "store",
                     "matmul", "transpose", "affine_range", "sequential_range",
                     "static_range", "dynamic_range", "fori_loop", "while_loop",
                     "ds", "tile_size", "shared_hbm", "private_hbm", "hbm", "sbuf",
                     "psum", "program_id", "num_programs", "spmd_dim", "nc",
                     "loop_reduce", "invert", "empty_like", "zeros", "ndarray",
                     "zeros_like", "full", "ones", "copy", "where", "softmax",
                     "load_transpose2d", "exp", "add", "range", "no_reorder",
                     "device_print", "rand", "random_seed", "gather_flattened",
                     "shared_identity_matrix", "mask", "multiply", "static_cast",
                     "abs", "bool_", "tfloat32", "float8_e4m3", "float8_e4m3fn",
                     "float8_e5m2", "int8", "int32"],
    "nki.isa": ["nc_matmul", "nc_transpose", "dma_copy", "dma_transpose",
                "dma_compute", "tensor_copy", "tensor_tensor", "tensor_scalar",
                "tensor_reduce", "activation", "activation_reduce", "memset",
                "iota", "affine_select", "range_select", "select_reduce",
                "tensor_tensor_scan", "topk", "reciprocal", "core_barrier",
                "sendrecv", "sendrecv_compute", "nc_matmul_mx", "quantize_mx",
                "exponential", "activate2", "tensor_copy_dynamic_src",
                "tensor_copy_dynamic_dst", "builtin_custom_op", "nc_find_index8",
                "max8", "bn_stats", "bn_aggr", "local_gather", "nc_n_gather",
                "nc_stream_shuffle", "scalar_tensor_tensor", "tensor_scalar_reduce",
                "tensor_partition_reduce", "register_alloc", "register_load",
                "register_store", "dropout", "rng", "rand2", "get_nc_version",
                "matmul_perf_mode", "engine", "nc_version", "dge_mode", "oob_mode"],
    "nki": ["jit", "simulate", "benchmark", "baremetal", "profile", "simulate_kernel",
            "trace", "language", "isa", "tensor", "Kernel", "debug"],
}


def legacy_record():
    out = {}
    for modname, names in LEGACY_NAMES.items():
        mod = importlib.import_module(modname)
        out[modname] = {n: hasattr(mod, n) for n in names}
    return out


def enums_record():
    out = {}
    for n in ["engine", "matmul_perf_mode", "nc_version", "dge_mode", "oob_mode", "reduce_cmd"]:
        e = getattr(nisa, n)
        try:
            out[n] = {m.name: repr(m.value) for m in e}
        except TypeError:
            out[n] = repr(e)
    return out


def dtypes_record():
    names = ["bool_", "int8", "int16", "int32", "uint8", "uint16", "uint32", "float16",
             "float32", "bfloat16", "tfloat32", "float8_e4m3", "float8_e4m3fn",
             "float8_e5m2", "float8_e5m2_x4", "float8_e4m3fn_x4", "float8_e8m0fnu",
             "float4_e2m1fn_x4"]
    out = {}
    for n in names:
        o = getattr(nl, n)
        out[n] = {"repr": repr(o), "type": type(o).__name__}
    # torch <-> nki mapping used by the torch-native integration.
    try:
        import torch
        from nki.framework import _torch_convert as tc
        tc._get_torch()
        out["_torch_to_nki"] = {str(k): repr(v) for k, v in tc._TORCH_TO_NKI.items()}
    except Exception as e:
        out["_torch_to_nki"] = f"<unavailable: {e!r}>"
    return out


def versions():
    v = {"python": platform.python_version(), "executable": sys.executable}
    for p in ["nki", "neuronx-cc", "torch", "torch-neuronx", "numpy", "islpy", "ml_dtypes"]:
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    v["nki.__version__"] = nki.__version__
    for p in ["torch_xla"]:
        try:
            importlib.import_module(p)
            v[p + "_importable"] = True
        except Exception:
            v[p + "_importable"] = False
    return v


def extra_record():
    from nki import _config
    from nki.framework import _lnc
    from nki.framework.kernel import CompileConfig, Kernel, SpecializationConfig
    import dataclasses
    return {
        "jit_signature": str(inspect.signature(nki.jit)),
        "Kernel_fields": {f.name: repr(f.default) for f in dataclasses.fields(Kernel)},
        "CompileConfig_fields": {f.name: repr(f.default) for f in dataclasses.fields(CompileConfig)},
        "SpecializationConfig_fields": {f.name: repr(f.default) for f in dataclasses.fields(SpecializationConfig)},
        "resolve_lnc_with_env": _lnc.resolve_lnc(),
        "NEURON_LOGICAL_NC_CONFIG": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
        "nl.range_is_affine_range": nl.range is nl.affine_range,
        "nl.hbm_is_private_hbm": nl.hbm is nl.private_hbm,
        "config_options": {
            name: {"default": repr(st._default), "doc": st.__doc__}
            for name, st in _config.config.states.items()
        },
    }


def main():
    rec = {
        "generated_by": "evidence/introspect.py",
        "versions": versions(),
        "env": {k: os.environ.get(k) for k in ["NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LOGICAL_NC_CONFIG"]},
        "modules": {m: module_record(m) for m in MODULES},
        "tile_size": tile_size_record(),
        "enums": enums_record(),
        "dtypes": dtypes_record(),
        "legacy_names_present": legacy_record(),
        "extra": extra_record(),
    }
    with open(OUT, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=False, default=repr)
    counts = {m: len(r["dir_public"]) for m, r in rec["modules"].items()}
    print(json.dumps({"out": OUT, "public_symbols": counts, "nki": rec["versions"]["nki"]}))


if __name__ == "__main__":
    main()
