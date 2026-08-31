"""CPU-only tests for the canonical NKI launch specification + atomic writes."""
import json
import os

from core.nki_profile_spec import (NkiProfileSpec, append_jsonl_locked,
                                   atomic_write_json, canonical_json,
                                   make_case_label)

BASE = dict(
    operator="vector_add",
    case_label="n1048576_fp16",
    case_params={"n": 1048576},
    dtype="fp16",
    block_size=1024,
    run_kwargs={"block_size": 1024},
    input_specs=[{"kind": "tensor", "shape": [1048576], "dtype": "torch.float16",
                  "stride": [1]}],
    autotune_enabled=True,
    autotune_replay=[{"tuner_name": "benchmarks.operators.vector_add.impl_nki.add_kernel",
                      "shape_key": [[128, 8192], "torch.float16"],
                      "config": {"free_tile_size": 2048}}],
    verify_atol=None,
    verify_rtol=None,
    warmup=2,
    repeat=3,
    nki_enabled=True,
    operator_source_sha256={"benchmarks/operators/vector_add/impl_nki.py": "aa" * 32},
    harness_source_sha256={"core/nki_timer.py": "bb" * 32},
    neuron_target="trn2",
    logical_nc_config="",
    neuron_cc_flags="",
    software_versions={"torch": "2.9.1", "nki": "0.6.0"},
)


def spec_with(**over):
    return NkiProfileSpec(**{**BASE, **over})


def test_dict_insertion_order_does_not_change_spec_id():
    a = spec_with(case_params={"n": 1048576, "m": 4})
    b = spec_with(case_params={"m": 4, "n": 1048576})
    assert a.spec_id == b.spec_id


def test_same_semantic_spec_same_id():
    assert spec_with().spec_id == spec_with().spec_id


def test_different_winner_config_changes_id():
    other = [dict(BASE["autotune_replay"][0], config={"free_tile_size": 16384})]
    assert spec_with().spec_id != spec_with(autotune_replay=other).spec_id


def test_different_operator_source_hash_changes_id():
    other = {"benchmarks/operators/vector_add/impl_nki.py": "cc" * 32}
    assert spec_with().spec_id != spec_with(operator_source_sha256=other).spec_id


def test_different_compiler_flags_change_id():
    assert spec_with().spec_id != spec_with(neuron_cc_flags="--model-type=transformer").spec_id


def test_different_dtype_changes_id():
    assert spec_with().spec_id != spec_with(dtype="bf16").spec_id


def test_canonical_json_is_key_sorted_and_compact():
    s = canonical_json({"b": 1, "a": [1, 2]})
    assert s == '{"a":[1,2],"b":1}'


def test_make_case_label_stable_and_safe():
    assert make_case_label({"n": 8, "m": 2}, "fp16") == make_case_label({"m": 2, "n": 8}, "fp16")
    assert "/" not in make_case_label({"path": "a/b c"}, "fp32")


def test_atomic_manifest_write_valid_json(tmp_path):
    path = str(tmp_path / "deep" / "manifest.json")
    atomic_write_json(path, {"spec_id": "x", "n": 1})
    with open(path) as f:
        assert json.load(f) == {"spec_id": "x", "n": 1}
    # overwrite is atomic too, and no temp litter remains
    atomic_write_json(path, {"spec_id": "y"})
    with open(path) as f:
        assert json.load(f) == {"spec_id": "y"}
    assert [p for p in os.listdir(tmp_path / "deep") if p.startswith(".tmp_")] == []


def test_append_jsonl_locked_no_partial_lines(tmp_path):
    path = str(tmp_path / "index.jsonl")
    append_jsonl_locked(path, {"a": 1})
    append_jsonl_locked(path, {"b": 2})
    lines = open(path).read().splitlines()
    assert [json.loads(l) for l in lines] == [{"a": 1}, {"b": 2}]


def test_operator_source_files_cover_transitive_repo_imports():
    """streamk_matmul's kernel lives in matmul_fp32_fp16_fp8/impl_nki.py; that
    file must enter the spec identity or manifest reuse could profile a stale
    artifact after the helper changes."""
    import importlib
    import pytest
    from core.nki_orchestrator import _operator_source_files

    impl = pytest.importorskip("benchmarks.operators.streamk_matmul.impl_nki")
    impl_torch = importlib.import_module("benchmarks.operators.streamk_matmul.impl_torch")
    files = _operator_source_files("streamk_matmul", (impl, impl_torch))
    assert "benchmarks/operators/streamk_matmul/impl_nki.py" in files
    assert "benchmarks/operators/streamk_matmul/impl_torch.py" in files
    assert "benchmarks/operators/matmul_fp32_fp16_fp8/impl_nki.py" in files
    assert all(not f.startswith("/") for f in files)  # repo-relative, stable
