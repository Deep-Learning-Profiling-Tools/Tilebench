"""The measurement protocol of the current source, for future formal runs.

- Every operator's config.yaml measures with warmup=1, repeat=3, and the
  framework falls back to the same values when a config omits them.
- Triton and TileLang time every autotune candidate with warmup=1, rep=3;
  cuTile's exhaustive search keeps its native policy.
- The autotune candidate lists (contents and order) match a snapshot.

These tests guard the source, not the committed results: parts of the committed
results were measured under an older protocol (docs/developer_guide.md,
Multi-Architecture Status).

The snapshot, tests/data/autotune_candidates.json, records every candidate list
as a count and a sha256. After an intentional change to a search space,
regenerate it on a host with all three GPU backends installed:

    TILEBENCH_UPDATE_CANDIDATE_SNAPSHOT=1 python -m pytest tests/test_measurement_protocol.py
"""
import ast
import hashlib
import importlib
import inspect
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tilebench.paths import list_operators, operator_config, operator_dir

SNAPSHOT = Path(__file__).parent / "data" / "autotune_candidates.json"
UPDATE = os.environ.get("TILEBENCH_UPDATE_CANDIDATE_SNAPSHOT") == "1"
BACKEND_PACKAGE = {"triton": "triton", "cutile": "cuda.tile", "tilelang": "tilelang"}


def test_suite_has_45_operators():
    assert len(list_operators()) == 45


@pytest.mark.parametrize("op", list_operators())
def test_config_measures_with_warmup_1_repeat_3(op):
    bench = yaml.safe_load(operator_config(op).read_text())["benchmark"]
    assert (bench["warmup"], bench["repeat"]) == (1, 3)


def test_framework_fallback_is_warmup_1_repeat_3():
    from tilebench.core import engine, timer

    assert (timer.DEFAULT_WARMUP, timer.DEFAULT_REPEAT) == (1, 3)
    assert (engine.DEFAULT_WARMUP, engine.DEFAULT_REPEAT) == (1, 3)
    params = inspect.signature(timer.report_benchmark).parameters
    assert (params["warmup"].default, params["repeat"].default) == (1, 3)


def _calls(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            yield name, {k.arg: k.value for k in node.keywords if k.arg}


def _impl_files(backend):
    return [p for op in list_operators() if (p := operator_dir(op) / f"impl_{backend}.py").exists()]


@pytest.mark.parametrize("backend", ["triton", "tilelang", "cutile"])
def test_candidate_timing_budget(backend):
    """Every autotune/do_bench site passes warmup=1, rep=3 explicitly, and no
    other warmup/rep value appears; exhaustive_search gets no budget at all."""
    sites = 0
    for path in _impl_files(backend):
        for name, kw in _calls(path):
            if name in ("autotune", "do_bench"):
                sites += 1
                assert "warmup" in kw and "rep" in kw, f"{path}: {name} without warmup/rep"
            if name == "exhaustive_search":
                assert not {"warmup", "rep"} & kw.keys(), f"{path}: cuTile search must stay native"
            for key, want in (("warmup", 1), ("rep", 3)):
                if key in kw:
                    assert isinstance(kw[key], ast.Constant) and kw[key].value == want, \
                        f"{path}: {name}({key}=...) is not {want}"
    if backend != "cutile":
        assert sites > 0


def _plain(x):
    if hasattr(x, "all_kwargs"):        # triton.Config
        return {"kwargs": _plain(x.kwargs),
                **{f: getattr(x, f, None) for f in ("num_warps", "num_stages", "num_ctas", "maxnreg")},
                "pre_hook": getattr(x.pre_hook, "__qualname__", None)}
    if isinstance(x, SimpleNamespace):
        x = vars(x)
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if callable(x):
        return getattr(x, "__qualname__", repr(x))
    return repr(x)


def _is_config_factory(module, name, obj):
    return (inspect.isfunction(obj) and obj.__module__ == module.__name__
            and re.fullmatch(r"[a-z0-9_]*_configs?", name)
            and "default" not in name and not name.startswith("get_last")
            and all(p.default is not inspect.Parameter.empty
                    for p in inspect.signature(obj).parameters.values()))


def _candidate_lists(module):
    """Every candidate list a module declares: triton.autotune / tilelang.autotune
    objects, *SPACE / *SPACE_BASE lists, and zero-argument *_config(s) factories."""
    found = {}
    for name, obj in vars(module).items():
        if type(obj).__name__ in ("Autotuner", "AutoTuneImpl"):
            configs = obj.configs
        elif re.search(r"SPACE(_BASE)?$", name):
            configs = obj
        elif _is_config_factory(module, name, obj):
            configs = obj()
        else:
            continue
        if isinstance(configs, (list, tuple)):
            found[name] = configs
    return found


def _snapshot_for(backend):
    entries = {}
    for path in _impl_files(backend):
        op = path.parent.name
        module = importlib.import_module(f"tilebench.benchmarks.operators.{op}.impl_{backend}")
        for name, configs in _candidate_lists(module).items():
            blob = json.dumps(_plain(configs), sort_keys=True).encode()
            entries[f"{op}/impl_{backend}.py::{name}"] = {
                "count": len(configs), "sha256": hashlib.sha256(blob).hexdigest()}
    return entries


@pytest.mark.parametrize("backend", ["triton", "cutile", "tilelang"])
def test_autotune_candidates_match_snapshot(backend):
    try:
        importlib.import_module(BACKEND_PACKAGE[backend])
    except Exception as e:
        pytest.skip(f"{BACKEND_PACKAGE[backend]} is not usable here: {e}")
    current = _snapshot_for(backend)
    snapshot = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
    if UPDATE:
        snapshot = {k: v for k, v in snapshot.items() if f"/impl_{backend}.py::" not in k}
        snapshot.update(current)
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(dict(sorted(snapshot.items())), indent=1) + "\n")
        pytest.skip(f"snapshot updated for {backend}")
    expected = {k: v for k, v in snapshot.items() if f"/impl_{backend}.py::" in k}
    assert current, f"no {backend} candidate lists found"
    changed = sorted(k for k in expected.keys() | current.keys() if expected.get(k) != current.get(k))
    assert not changed, f"autotune candidates differ from the snapshot: {changed}"
