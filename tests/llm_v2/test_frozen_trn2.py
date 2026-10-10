"""The prepared Trn2 NKI Base freeze (frozen/Trn2_base_rev4_2026-10-10.json) must match the tree it pins."""
import json
import subprocess
import sys

import pytest

from tilebench.llm.v2.manifests import schema as ms
from tilebench.paths import REPO_ROOT

FROZEN = ms.MANIFEST_DIR / "frozen" / "Trn2_base_rev4_2026-10-10.json"


@pytest.mark.skipif(not FROZEN.exists(), reason="Trn2 freeze not built")
def test_trn2_freeze_matches_the_tree():
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/llm_v2/build_trn2_freeze.py"), "--check"],
                       capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert r.returncode == 0, r.stdout + r.stderr[-3000:]


@pytest.mark.skipif(not FROZEN.exists(), reason="Trn2 freeze not built")
def test_trn2_freeze_protocol_and_scope():
    f = json.loads(FROZEN.read_text())
    p = f["protocol"]
    assert (p["revision"], p["rounds"], p["max_generations_per_round"], p["cases_per_task"], p["adjudicator"]) == \
        (4, 5, 1, 20, "claude-code")
    assert f["representative_dtypes"]["tasks_per_dsl"] == 37 and f["nominal_generations"] == 370
    assert f["trn2_eligibility"]["counts"] == {"eligible": 37, "excluded": 8}
    assert f["contracts_identical_to_b200_freeze"] is True
    assert f["evaluator_fingerprint_policy"]["timing_adapter"] == "neuron_native_trace"
    g = f["generators"]
    assert (g["gpt"]["model_id"], g["gpt"]["reasoning_effort"]) == ("gpt-6.1-sol", "xhigh")
    assert (g["claude"]["model_id"], g["claude"]["output_effort"], g["claude"]["thinking"]) == ("claude-opus-5-5", "xhigh", {"type": "adaptive"})
    for key, rec in f["skills"].items():
        if key != "optimization":
            assert rec["permission"] == "public" and sorted(rec["sendable_to"]) == ["anthropic", "openai"], key
