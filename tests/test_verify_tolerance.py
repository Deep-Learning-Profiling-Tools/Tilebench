"""Architecture-specific correctness tolerances: a `verify.arch_overrides`
entry replaces atol/rtol on that architecture only (2d_conv: atol 0.2 on CDNA3,
0.1 elsewhere), and the engine verifies with the values of the GPU present.

The engine runs a fake operator with the timer and verifier replaced, so these
run on any host.
"""
import sys
import types

import pytest
import yaml

from tilebench.core import engine
from tilebench.core.verifier import config_tolerance
from tilebench.paths import operator_config

CONV2D_VERIFY = yaml.safe_load(operator_config("2d_conv").read_text())["verify"]


@pytest.mark.parametrize("arch, atol", [("blackwell", 0.1), ("hopper", 0.1), ("cdna3", 0.2), (None, 0.1)])
def test_2d_conv_atol_by_architecture(arch, atol):
    assert config_tolerance(CONV2D_VERIFY, arch) == (atol, 0.01)              # rtol unchanged


def test_an_override_replaces_only_the_values_it_names():
    cfg = {"atol": 1.0, "rtol": 1e-2, "arch_overrides": {"cdna3": {"rtol": 5e-2}}}
    assert config_tolerance(cfg, "cdna3") == (1.0, 0.05)
    assert config_tolerance(cfg, "hopper") == (1.0, 0.01)
    assert config_tolerance({}, "cdna3") == (None, None)                       # per-dtype defaults
    assert config_tolerance({"arch_overrides": {"cdna3": {"atol": 0.2}}}, "blackwell") == (None, None)


@pytest.fixture
def fake_operator(tmp_path, monkeypatch):
    """One case of an operator with a torch and a Triton backend and the 2d_conv
    verify section; records the tolerances the engine verifies with."""
    cfg = {"benchmark": {"warmup": 1, "repeat": 1, "use_cuda_graph": False, "flush_l2": False},
           "verify": CONV2D_VERIFY, "test_cases": [{"n": 1, "dtype": "fp32"}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    used = []
    for backend in ("torch", "triton"):
        monkeypatch.setitem(sys.modules, f"tilebench.benchmarks.operators.fakeop.impl_{backend}",
                            types.SimpleNamespace(run=lambda n, **kw: n, get_last_config=lambda: None))
    monkeypatch.setattr(engine, "operator_config", lambda op: path)
    monkeypatch.setattr(engine, "get_generator", lambda op: (lambda n, dtype: (n,)))
    monkeypatch.setattr(engine, "infer_problem_size", lambda op, params: params["n"])
    monkeypatch.setattr(engine, "HAS_CUDA", True)
    monkeypatch.setattr(engine, "_sync", lambda: None)
    monkeypatch.setattr(engine, "timing_mode", lambda requested: None)
    monkeypatch.setattr(engine, "report_benchmark", lambda fn, args, **kw: {"mean": 1.0})
    monkeypatch.setattr(engine, "verify", lambda out, ref, atol, rtol: used.append((atol, rtol)) or (True, ""))
    return used


@pytest.mark.parametrize("arch, atol", [("blackwell", 0.1), ("hopper", 0.1), ("cdna3", 0.2)])
def test_the_engine_verifies_with_the_present_architecture(fake_operator, monkeypatch, arch, atol):
    monkeypatch.setattr(engine, "detect_arch", lambda: arch)
    engine.run_benchmark_suite("fakeop", enabled_backends={"triton"})
    assert fake_operator == [(atol, 0.01)]
