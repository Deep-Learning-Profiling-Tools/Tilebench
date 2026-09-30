"""Build the NCU sweep catalogue of one GPU: outputs/profiling/<gpu>/ncu_catalogue.json.

Thin command line over tilebench.profiling.ncu_catalogue. Sweep-max cases come
from the operator configs; autotune winners are read from exactly one log per
operator, results/<gpu>/logs/autotune_logs/<op>_autotune_<backends>.json.

Usage:  python scripts/profiling/ncu_catalogue.py --gpu B200 [op ...]
        ... --tile-language triton,cutile,tilelang   # winners from that run instead
"""
import argparse

# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without setting PYTHONPATH
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilebench.backends import parse_backends  # noqa: E402
from tilebench.paths import hardware_label  # noqa: E402
from tilebench.profiling.ncu_catalogue import write_catalogue  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the NCU sweep catalogue.")
    parser.add_argument("--gpu", type=hardware_label, required=True, metavar="LABEL",
                        help="Hardware label (e.g. B200): selects both the autotune logs read "
                             "(results/<gpu>/logs/autotune_logs/) and the catalogue written "
                             "(outputs/profiling/<gpu>/ncu_catalogue.json)")
    parser.add_argument("--tile-language", type=str, default="triton,cutile",
                        help="Backend selection of the autotune run whose winners to read, as "
                             "passed to run_bench.py --autotune. It must include triton and "
                             "cutile, the two backends NCU profiles (default: triton,cutile)")
    parser.add_argument("ops", nargs="*", help="Operators to refresh (default: all)")
    args = parser.parse_args(argv)
    try:
        backends = parse_backends(args.tile_language)
    except ValueError as e:
        parser.error(f"--tile-language: {e}")
    if not {"triton", "cutile"} <= set(backends):
        parser.error("--tile-language must include triton and cutile: the catalogue records "
                     "the autotune winners of exactly these two backends")

    try:
        out, built = write_catalogue(args.gpu, backends, args.ops)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"wrote {out}  ({len(built)} ops" + (f" refreshed: {args.ops})" if args.ops else ")"))
    no_log = [c["op"] for c in built if not c.get("has_autotune_log")]
    if no_log:
        print(f"ops without autotune log: {no_log}")
    for c in built:
        for dt in c["dtypes"]:
            has_tune = dt in c["autotune_winner_per_dtype"]
            print(f"  {c['op']:30s}  dtype={dt:8s}  "
                  f"autotune={'yes' if has_tune else 'NO'}  "
                  f"params={c['default_params_per_dtype'][dt]}")


if __name__ == "__main__":
    main()
