"""Sweep every autotune config of an operator and compare its B200 codegen.

Answers capability questions that timings cannot -- e.g. "does TileLang ever
emit tcgen05/TMA on Blackwell, under ANY config, or never?" -- without needing
a B200, because compilation is host-side. Nothing is executed.

The impls are the single source of truth for the search space: this reads the
same config list that `autotune=True` would search
  - TileLang: the module's `*_configs()` function
  - Triton:   the autotuned kernel's `.configs`
  - cuTile:   the module's `_SEARCH_SPACE`
and forces each one by overwriting the module-level `_DEFAULT_CONFIG` that
every `autotune=False` path reads at call time. So a config that appears here
is exactly a config autotune could pick -- no hand-maintained duplicate list.

Each config is compiled at sm_100a and its PTX/SASS scanned for the features
that matter on Blackwell (5th-gen tensor cores, TMA, async copy, spills).

Usage:
    PYTHONPATH=. python sweep_configs_sm100.py --operator matmul_fp32_fp16_fp8
    PYTHONPATH=. python sweep_configs_sm100.py --operator flash_attention --backend tilelang
    PYTHONPATH=. python sweep_configs_sm100.py --operator matmul_fp32_fp16_fp8 --dtype fp8_e4m3fn
    PYTHONPATH=. python sweep_configs_sm100.py --all --backend tilelang --limit 8

Writes results/sweeps_sm100a/<operator>[/<dtype>]/sweep.json plus a table on
stdout. Pass --dump to also keep every config's PTX/SASS.
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import lower_to_ptx_sm100 as L

REPO_ROOT = L.REPO_ROOT
OPERATORS_ROOT = L.OPERATORS_ROOT

BACKENDS = ("tilelang", "triton", "cutile")

# Codegen features worth a yes/no on Blackwell. Matched against SASS first,
# falling back to PTX for the backends whose SASS may be unavailable.
SASS_FEATURES = {
    # 5th-gen tensor core: the tcgen05 family is spelled UTC* in SASS
    # (UTCQMMA/UTCHMMA = the MMA itself, UTCBAR = barrier, UTCATOMSWS =
    # tensor-memory alloc), with LDTM/STTM moving tensor memory.
    "tcgen05": r"\bUTC[A-Z]*MMA",
    "tmem": r"\bLDTM|\bSTTM",                    # tensor-memory traffic
    "hmma": r"\bHMMA",                           # per-warp fp16 tensor core (Ampere-era)
    "imma": r"\bIMMA|\bQMMA(?<!UTCQMMA)",        # per-warp int/fp8 tensor core
    "ldsm": r"\bLDSM",                           # shared-memory matrix load
    "tma": r"\bUBLKCP|\bUTMA",                   # bulk tensor async copy
    "ldgsts": r"\bLDGSTS",                       # Ampere-style async global->shared
    "cvt_fp8": r"\bF2FP\.[A-Z0-9.]*E[45]M[23]",  # fp8 pack/unpack conversion
    "spill": r"\bLDL\b|\bSTL\b",                 # register spill traffic
}
PTX_FEATURES = {
    "tcgen05": r"tcgen05|wgmma",
    "tma": r"cp\.async\.bulk",
    "mma": r"\bmma\.sync",
}


def _count(text, pattern):
    return len(re.findall(pattern, text)) if text else 0


def analyze(ptx, sass):
    """Reduce one config's codegen to a comparable feature row."""
    row = {}
    for name, pat in SASS_FEATURES.items():
        row[name] = _count(sass, pat)
    for name, pat in PTX_FEATURES.items():
        row[f"ptx_{name}"] = _count(ptx, pat)
    # Instruction count: SASS lines carrying an opcode (skip headers/operands).
    row["insns"] = len(re.findall(r"^\s+/\*[0-9a-f]{4}\*/\s+\S", sass or "", re.M))
    row["ptx_lines"] = len((ptx or "").splitlines())
    return row


# ---------------- search-space discovery (impls are the source of truth) -----
def space_tilelang(module):
    for attr in dir(module):
        if attr.endswith("_configs") and callable(getattr(module, attr)):
            try:
                cfgs = getattr(module, attr)()
            except Exception:
                continue
            if isinstance(cfgs, (list, tuple)) and cfgs:
                return [dict(c) for c in cfgs]
    return []


def space_triton(module):
    for attr in dir(module):
        cfgs = getattr(getattr(module, attr), "configs", None)
        if isinstance(cfgs, (list, tuple)) and cfgs:
            out = []
            for c in cfgs:
                d = dict(getattr(c, "kwargs", {}) or {})
                for k in ("num_warps", "num_stages"):
                    v = getattr(c, k, None)
                    if v is not None:
                        d[k] = v
                out.append(d)
            return out
    return []


def space_cutile(module):
    space = getattr(module, "_SEARCH_SPACE", None)
    if not space:
        return []
    return [dict(vars(c)) if hasattr(c, "__dict__") else dict(c) for c in space]


SPACE_FN = {"tilelang": space_tilelang, "triton": space_triton, "cutile": space_cutile}
COLLECT_FN = {
    "tilelang": L.collect_tilelang,
    "triton": L.collect_triton,
    "cutile": L.collect_cutile,
}


# ---------------- config injection ------------------------------------------
def _default_holder(module, dtype_key):
    """Return (container, key) for the module's default config.

    Handles both the common module-level `_DEFAULT_CONFIG` and the dtype-keyed
    `_DEFAULT_CONFIGS` that matmul_fp32_fp16_fp8 uses.
    """
    if hasattr(module, "_DEFAULT_CONFIG"):
        return module, "_DEFAULT_CONFIG"
    cfgs = getattr(module, "_DEFAULT_CONFIGS", None)
    if isinstance(cfgs, dict) and cfgs:
        key = dtype_key if dtype_key in cfgs else next(iter(cfgs))
        return cfgs, key
    return None, None


def _merge(original, override):
    """Overlay swept keys onto the impl's own default, preserving its shape.

    Keeping the original as the base means keys the run() reads but the search
    space does not vary (and namespace-vs-dict typing) still work.
    """
    if isinstance(original, dict):
        merged = dict(original)
        merged.update(override)
        return merged
    merged = copy.copy(original)
    for k, v in override.items():
        setattr(merged, k, v)
    return merged


def sweep_backend(operator, backend, inputs, dtype, torch_dtype, limit=None, dump_dir=None):
    module = L._try_import(operator, f"impl_{backend}")
    if module is None:
        return []

    space = SPACE_FN[backend](module)
    if not space:
        print(f"//   {backend}: no search space found, skipping", file=sys.stderr)
        return []

    holder, key = _default_holder(module, torch_dtype)
    if holder is None:
        print(f"//   {backend}: no _DEFAULT_CONFIG to inject, skipping", file=sys.stderr)
        return []

    original = holder[key] if isinstance(holder, dict) else getattr(holder, key)
    if limit:
        space = space[:limit]

    rows = []
    try:
        for i, cfg in enumerate(space):
            merged = _merge(original, cfg)
            if isinstance(holder, dict):
                holder[key] = merged
            else:
                setattr(holder, key, merged)

            try:
                artifacts = COLLECT_FN[backend](operator, inputs)
            except Exception as e:
                print(f"//   {backend} cfg{i}: FAILED ({type(e).__name__}: {str(e)[:70]})",
                      file=sys.stderr)
                rows.append({"backend": backend, "config": cfg, "error": str(e)[:200]})
                continue

            for item in artifacts:
                if backend == "cutile":
                    name, blob, sass = item
                    ptx = None
                else:
                    name, ptx = item
                    sass = L.ptx_to_sass(ptx)
                row = {"backend": backend, "config": cfg, "kernel": name}
                row.update(analyze(ptx, sass))
                rows.append(row)

                if dump_dir:
                    stem = f"{backend}_cfg{i}_{L._safe(name)}"
                    if ptx:
                        (dump_dir / f"{stem}.ptx").write_text(ptx)
                    if sass:
                        (dump_dir / f"{stem}.sass").write_text(sass)
            print(f"//   {backend} cfg{i+1}/{len(space)}: {cfg}", file=sys.stderr)
    finally:
        if isinstance(holder, dict):
            holder[key] = original
        else:
            setattr(holder, key, original)
    return rows


def summarize(rows):
    """Which features vary across configs, and which never appear at all."""
    per_backend = {}
    for r in rows:
        if "error" in r:
            continue
        b = per_backend.setdefault(r["backend"], {})
        for k in list(SASS_FEATURES) + [f"ptx_{k}" for k in PTX_FEATURES]:
            b.setdefault(k, set()).add(r.get(k, 0))
    out = {}
    for b, feats in per_backend.items():
        out[b] = {
            "never": sorted(k for k, v in feats.items() if v == {0}),
            "always": sorted(k for k, v in feats.items() if 0 not in v),
            "config_dependent": sorted(k for k, v in feats.items() if len(v) > 1 and 0 in v),
        }
    return out


def print_table(rows):
    cols = ["tcgen05", "tmem", "hmma", "ldsm", "tma", "ldgsts", "cvt_fp8", "spill", "insns"]
    print(f"\n{'backend':9} {'kernel':28} " + " ".join(f"{c:>8}" for c in cols) + "  config")
    for r in rows:
        if "error" in r:
            print(f"{r['backend']:9} {'ERROR':28} {r['error'][:60]}")
            continue
        vals = " ".join(f"{r.get(c, 0):>8}" for c in cols)
        print(f"{r['backend']:9} {r['kernel'][:28]:28} {vals}  {r['config']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--backend", choices=BACKENDS, action="append",
                    help="restrict to a backend (repeatable); default all three")
    ap.add_argument("--dtype", help="e.g. fp16, fp8_e4m3fn")
    ap.add_argument("--limit", type=int, help="only the first N configs per backend")
    ap.add_argument("--dump", action="store_true", help="also write every config's PTX/SASS")
    args = ap.parse_args()

    if args.all:
        operators = L.all_operators()
    elif args.operator:
        operators = [args.operator]
    else:
        ap.error("give --operator <name> or --all")

    backends = args.backend or list(BACKENDS)

    for op in operators:
        print(f"\n// ===== {op}{':' + args.dtype if args.dtype else ''} =====", file=sys.stderr)
        try:
            case, inputs = L.pick_case(op, args.dtype)
        except Exception as e:
            print(f"// {op}: cannot build inputs ({e})", file=sys.stderr)
            continue
        torch_dtype = inputs[0].dtype if inputs and hasattr(inputs[0], "dtype") else None
        print(f"// case: {case}", file=sys.stderr)

        out_dir = REPO_ROOT / "results" / "sweeps_sm100a" / op
        if args.dtype:
            out_dir = out_dir / L._safe(args.dtype)
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_dir = out_dir if args.dump else None

        rows = []
        for be in backends:
            rows += sweep_backend(op, be, inputs, args.dtype, torch_dtype,
                                  limit=args.limit, dump_dir=dump_dir)

        report = {"operator": op, "dtype": args.dtype, "case": case,
                  "rows": rows, "summary": summarize(rows)}
        (out_dir / "sweep.json").write_text(json.dumps(report, indent=2, default=str))
        print_table(rows)
        print(f"\nsummary: {json.dumps(report['summary'], indent=2)}")
        print(f"wrote {out_dir / 'sweep.json'}")


if __name__ == "__main__":
    main()
