"""CPU-only tests for deterministic NEFF/HLO artifact identity.

No torch-xla, Neuron SDK, or Trainium required — pairs are fake byte files.
"""
import os

import pytest

from core.nki_artifact import (NEFF_PATH_ENV, NKI_HLO_MARKER,
                               NkiArtifactIdentityError, resolve_expected_pairs,
                               resolve_explicit_override, resolve_phase_pairs,
                               validate_manifest_reuse, validate_pair)
from core.nki_profile_spec import MANIFEST_SCHEMA_VERSION, sha256_file


def write_pair(root, stem, *, marker: bool, neff_bytes=b"NEFF-bytes"):
    neff = os.path.join(root, f"{stem}.neff")
    hlo = os.path.join(root, f"{stem}.hlo_module.pb")
    with open(neff, "wb") as f:
        f.write(neff_bytes + stem.encode())
    payload = b"hlo-proto " + (NKI_HLO_MARKER if marker else b"plain-torch") + stem.encode()
    with open(hlo, "wb") as f:
        f.write(payload)
    return neff, hlo


def manifest_for(spec_id, target, neff, hlo, marker=True, stem="MODULE_R"):
    return {"schema_version": MANIFEST_SCHEMA_VERSION, "spec_id": spec_id,
            "targets": {target: {"artifacts": [{
                "stem": stem, "neff_path": neff, "hlo_path": hlo,
                "neff_sha256": sha256_file(neff), "hlo_sha256": sha256_file(hlo),
                "hlo_nki_marker": marker}]}}}


def test_nki_phase_owns_every_new_pair(tmp_path):
    root = str(tmp_path)
    write_pair(root, "MODULE_PRE", marker=False)          # compiled before the phase
    neff_k, _ = write_pair(root, "MODULE_KERNEL", marker=True)
    neff_x, _ = write_pair(root, "MODULE_XLA_HELPER", marker=False)
    pairs = resolve_phase_pairs([root], nki_phase=True, exclude_stems={"MODULE_PRE"})
    assert [p.neff_path for p in pairs] == [neff_k, neff_x]   # sorted by stem, no mtime
    assert [p.has_marker for p in pairs] == [True, False]


def test_two_nki_pairs_are_both_recorded(tmp_path):
    root = str(tmp_path)
    write_pair(root, "MODULE_A", marker=True)
    write_pair(root, "MODULE_B", marker=True)
    assert [p.stem for p in resolve_phase_pairs([root], nki_phase=True)] == ["MODULE_A", "MODULE_B"]


def test_nki_phase_without_marker_pair_fails(tmp_path):
    root = str(tmp_path)
    write_pair(root, "MODULE_TORCH_ONLY", marker=False)
    with pytest.raises(NkiArtifactIdentityError, match="launched no NKI kernel"):
        resolve_phase_pairs([root], nki_phase=True)


def test_torch_phase_with_marker_pair_fails(tmp_path):
    root = str(tmp_path)
    write_pair(root, "MODULE_T", marker=False)
    write_pair(root, "MODULE_SNEAKY_NKI", marker=True)
    with pytest.raises(NkiArtifactIdentityError, match="must not launch NKI"):
        resolve_phase_pairs([root], nki_phase=False)


def test_zero_valid_pairs_fails(tmp_path):
    with pytest.raises(NkiArtifactIdentityError, match="zero valid"):
        resolve_phase_pairs([str(tmp_path)], nki_phase=True)


def test_stale_neff_outside_private_root_ignored(tmp_path):
    private = tmp_path / "private"
    stale = tmp_path / "repo_root"
    private.mkdir(); stale.mkdir()
    neff, _ = write_pair(str(private), "MODULE_MINE", marker=True)
    write_pair(str(stale), "MODULE_STALE1", marker=True)
    write_pair(str(stale), "MODULE_STALE2", marker=True)
    pairs = resolve_phase_pairs([str(private)], nki_phase=True)
    assert [p.neff_path for p in pairs] == [neff]


def test_missing_sibling_hlo_fails(tmp_path):
    root = str(tmp_path)
    with open(os.path.join(root, "MODULE_X.neff"), "wb") as f:
        f.write(b"neff")
    with pytest.raises(NkiArtifactIdentityError) as ei:
        resolve_phase_pairs([root], nki_phase=True)
    assert "missing sibling" in str(ei.value)


def test_marker_hlo_without_neff_fails(tmp_path):
    root = str(tmp_path)
    write_pair(root, "MODULE_OK", marker=True)
    with open(os.path.join(root, "MODULE_ORPHAN.hlo_module.pb"), "wb") as f:
        f.write(b"x" + NKI_HLO_MARKER)
    with pytest.raises(NkiArtifactIdentityError, match="without its NEFF"):
        resolve_phase_pairs([root], nki_phase=True)


def test_symlink_escape_fails(tmp_path):
    outside = tmp_path / "outside"
    root = tmp_path / "root"
    outside.mkdir(); root.mkdir()
    real_neff, real_hlo = write_pair(str(outside), "MODULE_E", marker=True)
    os.symlink(real_neff, root / "MODULE_E.neff")
    os.symlink(real_hlo, root / "MODULE_E.hlo_module.pb")
    with pytest.raises(NkiArtifactIdentityError) as ei:
        resolve_phase_pairs([str(root)], nki_phase=True)
    assert "escape" in str(ei.value)


def test_valid_manifest_reuse(tmp_path):
    root = str(tmp_path)
    neff, hlo = write_pair(root, "MODULE_R", marker=True)
    m = manifest_for("spec123", "nki", neff, hlo)
    pairs = validate_manifest_reuse(m, spec_id="spec123", allowed_roots=[root])
    assert [p.neff_path for p in pairs["nki"]] == [neff]


def test_manifest_reuse_per_target_states(tmp_path):
    root = str(tmp_path)
    neff, hlo = write_pair(root, "MODULE_R", marker=True)
    tneff, thlo = write_pair(root, "MODULE_T", marker=False)
    m = manifest_for("s", "nki", neff, hlo)
    torch_rec = manifest_for("s", "torch", tneff, thlo, marker=False, stem="MODULE_T")["targets"]["torch"]
    # torch compiled nothing (unsupported op): skipped, nki reused
    m["targets"]["torch"] = {"artifacts": []}
    assert list(validate_manifest_reuse(m, spec_id="s", allowed_roots=[root])) == ["nki"]
    # torch phase failed before identity: not reusable
    m["targets"]["torch"] = {"artifacts": None}
    with pytest.raises(NkiArtifactIdentityError, match="no artifact records"):
        validate_manifest_reuse(m, spec_id="s", allowed_roots=[root])
    # nki recorded only a non-marker pair: the kernel never compiled -> rebuild
    m["targets"]["torch"] = torch_rec
    m["targets"]["nki"] = {"artifacts": [dict(torch_rec["artifacts"][0])]}
    with pytest.raises(NkiArtifactIdentityError, match="no marker-bearing"):
        validate_manifest_reuse(m, spec_id="s", allowed_roots=[root])
    # nki compiled nothing -> rebuild
    m["targets"]["nki"] = {"artifacts": []}
    with pytest.raises(NkiArtifactIdentityError, match="no marker-bearing|no reusable"):
        validate_manifest_reuse(m, spec_id="s", allowed_roots=[root])


def test_manifest_reuse_rejects_other_schema_versions(tmp_path):
    root = str(tmp_path)
    neff, hlo = write_pair(root, "MODULE_R", marker=True)
    m = dict(manifest_for("spec123", "nki", neff, hlo), schema_version=1)
    with pytest.raises(NkiArtifactIdentityError, match="schema_version"):
        validate_manifest_reuse(m, spec_id="spec123", allowed_roots=[root])


def test_manifest_reuse_wrong_spec_id_fails(tmp_path):
    root = str(tmp_path)
    neff, hlo = write_pair(root, "MODULE_R", marker=True)
    m = manifest_for("specOLD", "nki", neff, hlo)
    with pytest.raises(NkiArtifactIdentityError, match="spec_id"):
        validate_manifest_reuse(m, spec_id="specNEW", allowed_roots=[root])


def test_manifest_reuse_neff_hash_mismatch_fails(tmp_path):
    root = str(tmp_path)
    neff, hlo = write_pair(root, "MODULE_R", marker=True)
    m = manifest_for("spec123", "nki", neff, hlo)
    with open(neff, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(NkiArtifactIdentityError, match="NEFF sha256 changed"):
        validate_manifest_reuse(m, spec_id="spec123", allowed_roots=[root])


def test_manifest_reuse_hlo_hash_mismatch_fails(tmp_path):
    root = str(tmp_path)
    neff, hlo = write_pair(root, "MODULE_R", marker=True)
    m = manifest_for("spec123", "nki", neff, hlo)
    with open(hlo, "ab") as f:
        f.write(NKI_HLO_MARKER)  # marker still present, content changed
    with pytest.raises(NkiArtifactIdentityError, match="HLO sha256 changed"):
        validate_manifest_reuse(m, spec_id="spec123", allowed_roots=[root])


def test_explicit_override_requires_valid_sibling(tmp_path, monkeypatch):
    root = str(tmp_path)
    lone = os.path.join(root, "MODULE_LONE.neff")
    with open(lone, "wb") as f:
        f.write(b"neff")
    monkeypatch.setenv(NEFF_PATH_ENV, lone)
    with pytest.raises(NkiArtifactIdentityError, match="sibling HLO missing"):
        resolve_explicit_override()

    neff, _hlo = write_pair(root, "MODULE_GOOD", marker=True)
    monkeypatch.setenv(NEFF_PATH_ENV, neff)
    pair = resolve_explicit_override()
    assert pair.neff_path == neff and pair.has_marker

    torch_neff, _ = write_pair(root, "MODULE_NOMARK", marker=False)
    monkeypatch.setenv(NEFF_PATH_ENV, torch_neff)
    with pytest.raises(NkiArtifactIdentityError, match="lacks the NKI marker"):
        resolve_explicit_override()

    monkeypatch.delenv(NEFF_PATH_ENV)
    assert resolve_explicit_override() is None


def test_resolver_never_calls_getmtime(tmp_path, monkeypatch):
    root = str(tmp_path)
    neff, hlo = write_pair(root, "MODULE_A", marker=True)
    write_pair(root, "MODULE_T", marker=False)

    def boom(_path):
        raise AssertionError("artifact identity consulted mtime!")

    monkeypatch.setattr(os.path, "getmtime", boom)
    pairs = resolve_phase_pairs([root], nki_phase=True)
    assert {p.neff_path for p in pairs} == {neff, neff.replace("MODULE_A", "MODULE_T")}
    validate_pair(neff, require_marker=True, allowed_roots=[root])
    validate_manifest_reuse(manifest_for("s", "nki", neff, hlo),
                            spec_id="s", allowed_roots=[root])


def test_resolve_expected_pairs_reuse_semantics(tmp_path):
    root = str(tmp_path)
    neff, _ = write_pair(root, "MODULE_KEEP", marker=True)
    write_pair(root, "MODULE_TORCH", marker=False)
    pre = {"MODULE_KEEP", "MODULE_TORCH"}
    pairs = resolve_expected_pairs([root], stems=["MODULE_KEEP"], nki_phase=True, pre_stems=pre)
    assert [p.neff_path for p in pairs] == [neff]
    with pytest.raises(NkiArtifactIdentityError, match="launched no NKI kernel"):
        resolve_expected_pairs([root], stems=["MODULE_TORCH"], nki_phase=True, pre_stems=pre)
    write_pair(root, "MODULE_NEWGRAPH", marker=True)   # graph changed during the run
    with pytest.raises(NkiArtifactIdentityError, match="new pair"):
        resolve_expected_pairs([root], stems=["MODULE_KEEP"], nki_phase=True, pre_stems=pre)
    with pytest.raises(NkiArtifactIdentityError, match="no longer present"):
        resolve_expected_pairs([root], stems=["MODULE_GONE"], nki_phase=True,
                               pre_stems=pre | {"MODULE_NEWGRAPH"})
