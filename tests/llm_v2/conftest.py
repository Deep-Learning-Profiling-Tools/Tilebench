"""Shared fixtures: a synthetic, hash-pinned skill manifest and a synthetic
contract, installed under a temporary repo root. No GPU, no network."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tilebench.llm.v2.manifests import schema as ms
from tilebench.llm.v2.skills import loader
from tilebench.llm.v2.contracts import loader as cloader


@pytest.fixture
def study():
    return ms.load_study()


@pytest.fixture
def folds():
    return ms.load_folds()


def _write(root: Path, rel: str, text: str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return rel


@pytest.fixture
def skill_env(tmp_path, monkeypatch, study):
    """Synthetic assets + manifest under tmp_path; returns (manifest, root)."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(loader, "REPO_ROOT", root)
    manifest_path = root / "skills" / "manifest.json"
    monkeypatch.setattr(loader, "MANIFEST_PATH", manifest_path)
    m = {"schema": loader.MANIFEST_SCHEMA, "reference": {}, "device": {}, "optimization": {}, "contract": {}}
    for dsl, ver in (("triton", "3.6.0"), ("cutile", "1.5.0"), ("tilelang", "0.1.11")):
        rel = _write(root, f"skills/reference/{dsl}/{ver}/SKILL.md", f"---\nname: {dsl}\n---\n# {dsl} reference\nAPI text.\r\n\n\n")
        loader.register_asset(m, "reference", dsl, ver, rel, permission="public", status="approved", source="test")
    rel = _write(root, "skills/reference/nki/beta5/SKILL.md", "# nki private\n")
    loader.register_asset(m, "reference", "nki", "beta5", rel, permission="private", status="draft", source="test")
    for dev in ("B200", "GH200", "MI300X", "Trn2"):
        rel = _write(root, f"skills/device/{dev}/2026-10-05/SKILL.md", f"# {dev} facts\nunknown: yes\n")
        loader.register_asset(m, "device", dev, "2026-10-05", rel, permission="public", status="approved", source="test")
    for fold, status, body in (("A", "frozen", "# triton skill A\nrule.\n"), ("C", "frozen", "# triton skill C\nrule.\n"),
                               ("B", "draft", "# triton skill B (draft)\n")):
        rel = _write(root, f"skills/optimization/triton/source-B200/fold-{fold}/SKILL.md", body)
        write_opt_manifest(root, rel, dsl="triton", fold=fold, status=status)
        loader.register_asset(m, "optimization", f"triton/{fold}", "v1", rel, permission="public", status=status, source="test")
    loader.save_manifest(m, manifest_path)
    return m, root


def write_opt_manifest(root: Path, rel: str, *, dsl: str, fold: str, status: str, **overrides) -> dict:
    """A provenance manifest beside an optimization SKILL.md, valid for an
    evaluation-mode Enhanced run of (dsl, fold) unless overridden."""
    import hashlib
    from tilebench.llm.v2.manifests.schema import training_folds
    text = loader.transform((root / rel).read_bytes())
    m = {"schema": "tilebench-optimization-skill/2", "dsl": dsl, "compatible_versions": [{"triton": "3.6.0", "cutile": "1.5.0", "tilelang": "0.1.11"}[dsl]],
         "source_device": "B200", "held_out_fold": fold, "training_folds": list(training_folds(fold)),
         "evaluation_or_release": "evaluation", "source_trajectory_ids": ["t-synthetic-1", "t-synthetic-2"],
         "content_sha256": hashlib.sha256(text.encode()).hexdigest(), "status": status}
    m.update(overrides)
    (root / rel).parent.joinpath("manifest.json").write_text(json.dumps(m, indent=1))
    return m


@pytest.fixture
def contract_env(tmp_path, monkeypatch):
    """Synthetic approved contract for vector_add and a draft one for relu."""
    data = tmp_path / "contracts"
    monkeypatch.setattr(cloader, "DATA_DIR", data)

    def make(op: str, status: str, text: str | None = None):
        d = data / op
        d.mkdir(parents=True)
        (d / "contract.md").write_text(text or f"# {op}: canonical algorithm contract\n## Functional semantics\nout = a + b\n"
                                               "## Permitted PyTorch operations\n- torch.empty_like\n")
        rules = {"schema": cloader.RULES_SCHEMA if hasattr(cloader, "RULES_SCHEMA") else "tilebench-evaluator-rules/1",
                 "operator": op, "contract_revision": 1,
                 "required_stages": [{"id": "s1", "description": "elementwise"}],
                 "forbidden_substitutions": [{"pattern": r"torch\.add\(", "message": "delegated add", "level": "confirmed"}],
                 "required_evidence": [{"any_of": [r"tl\.load", r"ct\.load", r"T\.copy"], "message": "tile load", "level": "suspicious"}],
                 "allowed_torch_calls": ["torch.empty_like"],
                 "mutation": {"inputs_mutated": [], "restore_required": False},
                 "outputs": {"structure": "tensor", "count": 1, "aliasing": "none"},
                 "timing_boundary": {"includes_preprocessing": False, "notes": ""},
                 "tolerance_source": "config.verify"}
        (d / "evaluator_rules.json").write_text(json.dumps(rules))
        audit = {"schema": "tilebench-contract-audit/1", "operator": op, "source_sha": "ea04fb36", "contract_revision": 1,
                 "status": status, "approved_by": "tester" if status == "approved" else None, "approved_on": None,
                 "sources": {}, "aspects": {a: {"status": "aligned", "triton": "", "cutile": "", "note": ""}
                                            for a in ("functional_semantics", "stages", "dependencies", "reduction_scan_sort",
                                                      "precision", "preprocessing", "intermediate_storage", "mutation", "hardware_dispatch")},
                 "fq_boundary": {"bytes_expr": "", "flops_expr": "", "consistent": "true", "note": ""},
                 "human_reference_comparable": True, "review_items": [], "contract_sha256": None}
        (d / "audit.json").write_text(json.dumps(audit))
        return d
    make("vector_add", "approved")
    make("relu", "draft")
    make("mul2", "approved", text="# mul2\nUse BLOCK_M = 128 and num_warps = 4 for 0.5 ms\n")  # leaking contract
    return data
