"""Input bundles shared by every stack and mode.

A bundle is created once on the CPU with a fixed seed, from the operator's
registered generator and its config case, together with the CPU reference
output of the pinned ``impl_torch.run``. The same bundle file is loaded by the
XLA and the native workers, so both stacks see byte-identical inputs. Each case
gets two bundles (seed and seed+1): the second checks that a changed input
produces a correspondingly changed, still correct output.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import torch


def tensor_digest(t: torch.Tensor) -> str:
    c = t.detach().contiguous().cpu()
    if c.dtype in (torch.bfloat16,) or str(c.dtype).startswith("torch.float8"):
        raw = c.view(torch.uint8 if c.element_size() == 1 else torch.int16).numpy().tobytes()
    else:
        raw = c.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _describe(x) -> object:
    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", ""),
                "stride": list(x.stride()), "sha256": tensor_digest(x)}
    if isinstance(x, (tuple, list)):
        return [_describe(v) for v in x]
    return {"value": x if isinstance(x, (int, float, bool, str, type(None))) else repr(x)}


def make_bundle(generate, impl_torch_run, params: dict, dtype, seed: int) -> dict:
    """Inputs + CPU reference for one case. Raises what the generator/reference raise."""
    torch.manual_seed(seed)
    inputs = generate(**params, dtype=dtype)
    inputs = tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,)
    clones = tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in inputs)
    ref = impl_torch_run(*clones)  # the reference may mutate its inputs (clones only)
    return {"inputs": inputs, "reference": ref, "seed": seed}


def save_bundle(bundle: dict, path: Path) -> dict:
    """Write with exclusive creation; returns metadata (file + per-tensor sha256)."""
    buf = io.BytesIO()
    torch.save({"inputs": bundle["inputs"], "reference": bundle["reference"],
                "seed": bundle["seed"]}, buf)
    data = buf.getvalue()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(data)
    meta = {"path": str(path), "file_sha256": hashlib.sha256(data).hexdigest(),
            "seed": bundle["seed"], "inputs": _describe(bundle["inputs"]),
            "reference": _describe(bundle["reference"])}
    with open(str(path) + ".json", "x") as f:
        json.dump(meta, f, indent=1)
    return meta


def load_bundle(path: Path, expected_sha256: str | None = None) -> dict:
    data = Path(path).read_bytes()
    if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"bundle {path} does not match its recorded checksum")
    return torch.load(io.BytesIO(data), weights_only=False)


def int64_tensors(obj) -> list[str]:
    """Paths of int64 tensors (the native stack documents an int64 -> int32 downcast)."""
    out = []

    def walk(x, p):
        if isinstance(x, torch.Tensor) and x.dtype == torch.int64:
            out.append(p)
        elif isinstance(x, (tuple, list)):
            for i, v in enumerate(x):
                walk(v, f"{p}[{i}]")
    walk(obj, "")
    return out


def _bundle_main(argv=None) -> int:
    """``python -m tilebench.neuron_diag.bundles --spec spec.json``: create the main and
    the changed-input bundle of one case inside the operator's overlay. Writes a JSON
    result (metadata or the reference error) to ``spec["out"]``."""
    import argparse
    import importlib
    import traceback

    from tilebench.core.dtypes import resolve_dtype
    from tilebench.data.tensors import get_generator

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    args = ap.parse_args(argv)
    with open(args.spec) as f:
        spec = json.load(f)
    res: dict = {}
    try:
        gen = get_generator(spec["operator"])
        impl_torch = importlib.import_module(
            f"tilebench.benchmarks.operators.{spec['operator']}.impl_torch")
        dtype = resolve_dtype(spec["dtype"])
        for tag, seed in (("main", spec["seed"]), ("alt", spec["seed"] + 1)):
            try:
                b = make_bundle(gen, impl_torch.run, spec["params"], dtype, seed)
            except (RuntimeError, TypeError, NotImplementedError, ValueError) as e:
                res = {"error": f"{type(e).__name__}: {e}"[:2000], "stage": f"{tag} bundle"}
                break
            res[tag] = save_bundle(b, Path(spec["dir"]) / f"{tag}_s{seed}.pt")
    except Exception as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {e}"[:2000], "traceback": traceback.format_exc()[-3000:]}
    with open(spec["out"], "x") as f:
        json.dump(res, f, default=str)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_bundle_main())
