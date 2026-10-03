"""TileLang in the GH200 profiling pipeline: strict winner replay, the catalogue
keeps TileLang winners, and selecting tilelang never touches the Triton/cuTile
metadata or reports.

Operator modules and GPU work are replaced by fakes, so these run on any host.
"""
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import tilebench.paths as paths
from tilebench.profiling import ncu_kernel_select as ks
from tilebench.profiling.replay import ReplayError, apply_winner

TOOLS = Path(__file__).resolve().parents[1] / "scripts" / "profiling"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_impl(**attrs):
    mod = ModuleType("tilebench.benchmarks.operators.fake.impl_tilelang")
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# --------------------------------------------------------------------------
# strict replay
# --------------------------------------------------------------------------

def test_strict_replay_installs_a_plain_winner():
    impl = fake_impl(_DEFAULT_CONFIG={"BLOCK_SIZE": 1024, "threads": 128})
    apply_winner(impl, {"BLOCK_SIZE": 256, "threads": 64}, strict=True)
    assert impl._DEFAULT_CONFIG == {"BLOCK_SIZE": 256, "threads": 64}


def test_strict_replay_rejects_a_key_run_would_ignore():
    impl = fake_impl(_DEFAULT_CONFIG={"BLOCK_SIZE": 1024, "threads": 128})
    with pytest.raises(ReplayError, match="no place"):
        apply_winner(impl, {"BLOCK_SIZE": 256, "num_warps": 4}, strict=True)
    assert impl._DEFAULT_CONFIG == {"BLOCK_SIZE": 1024, "threads": 128}      # untouched


def test_without_strict_the_existing_merge_is_unchanged():
    impl = fake_impl(_DEFAULT_CONFIG=SimpleNamespace(block=1024, occupancy=4))
    apply_winner(impl, {"block": 2048, "occupancy": 8, "K2": 16})              # top_k cuTile style
    assert vars(impl._DEFAULT_CONFIG) == {"block": 2048, "occupancy": 8, "K2": 16}


def test_strict_replay_collapses_identical_per_launch_winners():
    impl = fake_impl(_DEFAULT_CONFIG={"BLOCK_SIZE": 1024, "threads": 128})   # destindex: one kernel, two launches
    apply_winner(impl, {"nope_BLOCK_SIZE": 512, "nope_threads": 64,
                        "rope_BLOCK_SIZE": 512, "rope_threads": 64}, strict=True)
    assert impl._DEFAULT_CONFIG == {"BLOCK_SIZE": 512, "threads": 64}


def test_strict_replay_refuses_differing_per_launch_winners():
    impl = fake_impl(_DEFAULT_CONFIG={"BLOCK_SIZE": 1024, "threads": 128})
    with pytest.raises(ReplayError, match="cannot be replayed exactly"):
        apply_winner(impl, {"nope_BLOCK_SIZE": 512, "nope_threads": 64,
                            "rope_BLOCK_SIZE": 1024, "rope_threads": 64}, strict=True)


def test_strict_replay_routes_prefixed_and_per_dtype_configs():
    impl = fake_impl(_DEFAULT_KV_CONFIG={"BLOCK_M": 64, "threads": 128},
                     _DEFAULT_OUT_CONFIG={"BLOCK_M": 64, "threads": 128})
    apply_winner(impl, {"kv_BLOCK_M": 32, "kv_threads": 256, "out_BLOCK_M": 128, "out_threads": 64}, strict=True)
    assert impl._DEFAULT_KV_CONFIG == {"BLOCK_M": 32, "threads": 256}
    assert impl._DEFAULT_OUT_CONFIG == {"BLOCK_M": 128, "threads": 64}

    impl = fake_impl(_DEFAULT_CONFIGS={torch.float16: {"BLOCK_SIZE_M": 256, "threads": 128},
                                       torch.float32: {"BLOCK_SIZE_M": 128, "threads": 128}})
    apply_winner(impl, {"BLOCK_SIZE_M": 64, "threads": 256}, torch.float16, strict=True)
    assert impl._DEFAULT_CONFIGS[torch.float16] == {"BLOCK_SIZE_M": 64, "threads": 256}
    assert impl._DEFAULT_CONFIGS[torch.float32] == {"BLOCK_SIZE_M": 128, "threads": 128}
    with pytest.raises(ReplayError):
        apply_winner(impl, {"num_warps": 4}, torch.float16, strict=True)


# --------------------------------------------------------------------------
# metadata: catalogue keeps the TileLang winner; a tilelang probe keeps the rest
# --------------------------------------------------------------------------

@pytest.fixture
def gh200(tmp_path, monkeypatch):
    from tilebench.profiling import ncu_catalogue
    root = tmp_path / "outputs" / "profiling"
    monkeypatch.setattr(paths, "PROFILING_METADATA_ROOT", root)
    monkeypatch.setattr(paths, "NCU_OUTPUT_ROOT", tmp_path / "outputs" / "ncu")
    monkeypatch.setattr(paths, "RESULTS_ROOT", tmp_path / "results")
    ncu_catalogue.write_catalogue("GH200", ["triton", "cutile", "tilelang"])
    return root / "GH200"


def test_catalogue_keeps_the_tilelang_winner(gh200):
    entry = next(c for c in ks.load_catalogue("GH200") if c["op"] == "mul2")
    rows = [{"params": entry["default_params_per_dtype"][dt], "problem_size": 1, "dtype": dt,
             "triton_autotune_cfg": {"BLOCK_SIZE": 512}, "cutile_autotune_cfg": {"occupancy": 4},
             "tilelang_autotune_cfg": {"BLOCK_SIZE": 256, "threads": 64}} for dt in entry["dtypes"]]
    log = paths.autotune_log_path("GH200", "mul2", "autotune", ["triton", "cutile", "tilelang"])
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(rows))
    load_tool("ncu_catalogue").main(["--gpu", "GH200", "--tile-language", "triton,cutile,tilelang", "mul2"])
    entry = next(c for c in ks.load_catalogue("GH200") if c["op"] == "mul2")
    for w in entry["autotune_winner_per_dtype"].values():
        assert w["triton"] == {"BLOCK_SIZE": 512} and w["cutile"] == {"occupancy": 4}
        assert w["tilelang"] == {"BLOCK_SIZE": 256, "threads": 64}


def test_a_tilelang_probe_keeps_the_triton_and_cutile_rows(gh200, monkeypatch):
    pairs = [(c["op"], dt) for c in ks.load_catalogue("GH200") for dt in c["dtypes"]]
    existing = [{"op": op, "dtype": dt, "backend": be, "count": 3, "names": ["k"] * 3}
                for op, dt in pairs for be in ("triton", "cutile")]
    (gh200 / "kernel_counts.json").write_text(json.dumps(existing))
    probe = load_tool("probe_kernel_count")
    seen = []
    monkeypatch.setattr(probe, "count_one", lambda op, be, *a: seen.append(be) or {"count": 1, "names": ["main_kernel"]})
    monkeypatch.delenv("ONLY_OP", raising=False)
    monkeypatch.setattr(sys, "argv", ["probe", "--gpu", "GH200", "--tile-language", "tilelang"])
    with pytest.raises(SystemExit, match="no TileLang winners"):      # catalogue of a run without TileLang
        probe.main()
    assert not seen and json.loads((gh200 / "kernel_counts.json").read_text()) == existing

    cat = json.loads((gh200 / "ncu_catalogue.json").read_text())
    for c in cat:
        c["autotune_winner_per_dtype"] = {dt: {"params": c["default_params_per_dtype"][dt], "triton": {"A": 1},
                                               "cutile": {"B": 2}, "tilelang": {"threads": 128}}
                                          for dt in c["dtypes"]}
    (gh200 / "ncu_catalogue.json").write_text(json.dumps(cat))
    probe.main()
    rows = json.loads((gh200 / "kernel_counts.json").read_text())
    assert set(seen) == {"tilelang"}
    assert sorted((r["op"], r["dtype"], r["backend"], r["count"]) for r in rows if r["backend"] != "tilelang") == \
        sorted((r["op"], r["dtype"], r["backend"], r["count"]) for r in existing)
    assert len([r for r in rows if r["backend"] == "tilelang"]) == len(pairs)


@pytest.mark.parametrize("argv, backends", [([], {"triton", "cutile"}), (["--tile-language", "tilelang"], {"tilelang"})])
def test_the_driver_profiles_only_the_selected_backends(gh200, monkeypatch, argv, backends):
    rows = [{"op": c["op"], "dtype": dt, "backend": be, "count": 1, "names": [f"{c['op']}_kernel"]}
            for c in ks.load_catalogue("GH200") for dt in c["dtypes"] for be in ("triton", "cutile", "tilelang")]
    (gh200 / "kernel_counts.json").write_text(json.dumps(rows))
    driver = load_tool("ncu_driver")
    seen = set()

    def fake_run_one(op, dt, be, params, cfg, out, **kw):
        seen.add(be)
        return {"op": op, "dtype": dt, "backend": be, "ok": True, "rc": 0, "elapsed_s": 0, "stderr_tail": ""}
    monkeypatch.setattr(driver, "run_one", fake_run_one)
    monkeypatch.setattr(sys, "argv", ["ncu_driver", "--gpu", "GH200", *argv])
    driver.main()
    assert seen == backends


@pytest.mark.parametrize("backend, captured, ok", [
    ("tilelang", ["pad_kernel_kernel", "main_kernel", "main_kernel"], True),
    ("tilelang", ["main_kernel", "pad_kernel_kernel", "main_kernel"], False),     # same names and count, wrong order
    ("triton", ["main_kernel", "pad_kernel_kernel", "main_kernel"], True),        # Triton/cuTile rule unchanged
])
def test_a_tilelang_capture_must_follow_the_probed_launch_order(tmp_path, monkeypatch, backend, captured, ok):
    driver = load_tool("ncu_driver")
    out = tmp_path / "op" / f"{backend}_fp16.ncu-rep"

    def fake_ncu(cmd, **kw):
        out.write_bytes(b"rep")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(driver.subprocess, "run", fake_ncu)
    monkeypatch.setattr(driver.ks, "captured_from_report", lambda path: list(captured))
    res = driver.run_one("op", "fp16", backend, {}, None, out, n_kernels_per_call=3,
                         kernel_names=["pad_kernel_kernel", "main_kernel", "main_kernel"])
    assert res["ok"] is ok and res["validated"] is ok
