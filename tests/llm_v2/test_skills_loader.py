"""Manifest-driven skill loading: hashes, versions, permissions, Base/Enhanced composition."""
import pytest

from tilebench.llm.v2.skills import loader


def test_transform_rule_strips_frontmatter_and_normalizes():
    raw = b"---\nname: x\n---\r\n\r\n# T  \nline\r\n\n\n"
    assert loader.transform(raw) == "# T\nline\n"


def test_component_hashes_match_manifest(skill_env):
    m, root = skill_env
    c = loader.load_component(m, "reference", "triton", "3.6.0")
    assert c.sha256_injected == m["reference"]["triton"]["3.6.0"]["sha256_injected"]
    assert c.text.startswith("# triton reference") and "---" not in c.text


def test_hash_mismatch_fails(skill_env):
    m, root = skill_env
    (root / m["reference"]["triton"]["3.6.0"]["path"]).write_text("# tampered\n")
    with pytest.raises(loader.SkillHashError):
        loader.load_component(m, "reference", "triton", "3.6.0")


def test_missing_version_and_missing_dsl_fail_without_fallback(skill_env):
    m, _ = skill_env
    with pytest.raises(loader.SkillMissingError):
        loader.load_component(m, "reference", "triton", "3.5.0")
    with pytest.raises(loader.SkillMissingError):
        loader.load_component(m, "reference", "mojo")


def test_private_asset_is_never_sendable(skill_env):
    m, _ = skill_env
    with pytest.raises(loader.SkillPermissionError):
        loader.load_component(m, "reference", "nki", "beta5", require_status=())
    c = loader.load_component(m, "reference", "nki", "beta5", require_status=(), provider_sendable=False)
    assert c.permission == "private"


def test_draft_status_blocks_live(skill_env):
    m, _ = skill_env
    with pytest.raises(loader.SkillStatusError):
        loader.load_component(m, "optimization", "triton/B", "v1")


def test_length_limit_errors_instead_of_truncating(skill_env):
    m, _ = skill_env
    with pytest.raises(loader.SkillTooLongError):
        loader.load_component(m, "reference", "triton", "3.6.0", max_chars=5)


def test_base_and_enhanced_differ_only_by_optimization(skill_env, study):
    m, _ = skill_env
    base = loader.compose_context(m, study, dsl="triton", device="GH200", fold="A", condition="base")
    enh = loader.compose_context(m, study, dsl="triton", device="GH200", fold="A", condition="enhanced")
    assert [c.kind for c in base] == ["reference", "device"]
    assert [c.kind for c in enh] == ["reference", "device", "optimization"]
    assert loader.hash_record(base).items() <= loader.hash_record(enh).items()


def test_enhanced_without_frozen_skill_refuses(skill_env, study):
    m, _ = skill_env
    with pytest.raises(loader.SkillStatusError):
        loader.compose_context(m, study, dsl="triton", device="B200", fold="B", condition="enhanced")
    with pytest.raises(loader.SkillMissingError):
        loader.compose_context(m, study, dsl="cutile", device="B200", fold="A", condition="enhanced")


def test_same_fold_skill_hash_is_identical_across_devices(skill_env, study):
    m, _ = skill_env
    hashes = {dev: loader.compose_context(m, study, dsl="triton", device=dev, fold="A", condition="enhanced")[2].sha256_injected
              for dev in ("B200", "GH200", "MI300X")}
    assert len(set(hashes.values())) == 1


def test_reference_is_shared_across_devices_and_device_across_dsls(skill_env, study):
    m, _ = skill_env
    r1 = loader.compose_context(m, study, dsl="triton", device="B200", fold="A", condition="base")[0]
    r2 = loader.compose_context(m, study, dsl="triton", device="MI300X", fold="A", condition="base")[0]
    assert r1.sha256_injected == r2.sha256_injected
    d1 = loader.compose_context(m, study, dsl="triton", device="GH200", fold="A", condition="base")[1]
    d2 = loader.compose_context(m, study, dsl="cutile", device="GH200", fold="A", condition="base")[1]
    assert d1.sha256_injected == d2.sha256_injected


def test_transfer_list_is_enforced(skill_env, study):
    m, _ = skill_env
    with pytest.raises(loader.SkillError):
        loader.compose_context(m, study, dsl="triton", device="Trn2", fold="A", condition="enhanced")
