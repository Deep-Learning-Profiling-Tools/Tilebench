"""Deterministic NKI/torch NEFF artifact identity (fail-loud, mtime-free).

With NEURON_FRAMEWORK_DEBUG=1 the Neuron compiler dumps, into the worker's
private CWD, one pair per compiled XLA graph:

    <stem>.neff             the compiled binary the runtime loads (byte-identical
                            to the NEFF the runtime-inspect trace writes back)
    <stem>.hlo_module.pb    the post-optimization HLO for that graph

Identity rules (see core/nki_profile_worker.py for the process design):

- artifacts are searched ONLY inside explicitly allowed private roots
  (the current spec's working directory), never in global locations;
- a valid pair requires both regular files, same stem, same directory,
  no symlink escape outside the allowed roots;
- an NKI graph is recognized by the raw marker bytes
  ``AwsNeuronCustomNativeKernel`` inside the HLO (validation marker — it does
  NOT distinguish two autotune candidates; that distinction comes from exact
  winner replay in a fresh process, so the NKI phase compiles only the
  winner's graph(s));
- a phase (torch baseline / NKI run) owns EVERY pair that appeared during it —
  an operator may legitimately compile several graphs per ``run()`` (e.g. one
  per radix-sort pass). Which of them executed, how often and for how long
  comes from the runtime trace (core/nki_timer.py), matched to these pairs by
  NEFF SHA256; an executed NEFF that matches no pair is an identity error;
- the NKI phase must contain at least one marker-bearing pair and the torch
  phase none; zero pairs in a phase raises :class:`NkiArtifactIdentityError`;
- selection NEVER consults mtime, file size, sequence numbers, or glob order.
"""
from __future__ import annotations

import dataclasses
import os
from typing import Iterable, Sequence

from tilebench.core.nki_profile_spec import MANIFEST_SCHEMA_VERSION, sha256_file

NKI_HLO_MARKER = b"AwsNeuronCustomNativeKernel"
HLO_SUFFIX = ".hlo_module.pb"
NEFF_SUFFIX = ".neff"

# Expert escape hatch (validated, never silent): explicit NEFF path.
NEFF_PATH_ENV = "NKI_NEFF_PATH"


class NkiArtifactIdentityError(RuntimeError):
    """Artifact identity could not be established unambiguously."""

    def __init__(self, message: str, *, context: dict | None = None,
                 roots: Sequence[str] = (), candidates: Sequence[dict] = ()):
        self.context = dict(context or {})
        self.roots = list(roots)
        self.candidates = list(candidates)
        lines = [message]
        if self.context:
            lines.append(f"  context: {self.context}")
        lines.append(f"  searched roots: {self.roots or '(none)'}")
        if self.candidates:
            lines.append("  candidates:")
            for c in self.candidates:
                lines.append(f"    - {c}")
        else:
            lines.append("  candidates: (none found)")
        super().__init__("\n".join(lines))


@dataclasses.dataclass(frozen=True)
class ArtifactPair:
    stem: str
    neff_path: str
    hlo_path: str
    neff_sha256: str
    hlo_sha256: str
    has_marker: bool


def _is_inside(root: str, path: str) -> bool:
    root_r = os.path.realpath(root)
    path_r = os.path.realpath(path)
    return path_r == root_r or path_r.startswith(root_r + os.sep)


def _hlo_has_marker(hlo_path: str) -> bool:
    with open(hlo_path, "rb") as f:
        return NKI_HLO_MARKER in f.read()


def discover_pairs(roots: Sequence[str]) -> tuple[list[ArtifactPair], list[dict]]:
    """Enumerate NEFF/HLO stem-pairs strictly inside ``roots``.

    Returns (valid_pairs, rejections). Never orders by, or even reads, mtime.
    """
    pairs: list[ArtifactPair] = []
    rejections: list[dict] = []
    seen_neffs: set[str] = set()
    for root in roots:
        if not os.path.isdir(root):
            rejections.append({"root": root, "reason": "root does not exist"})
            continue
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
            for fn in sorted(filenames):
                if not fn.endswith(NEFF_SUFFIX):
                    continue
                neff = os.path.join(dirpath, fn)
                real = os.path.realpath(neff)
                if real in seen_neffs:
                    continue
                seen_neffs.add(real)
                stem = fn[: -len(NEFF_SUFFIX)]
                hlo = os.path.join(dirpath, stem + HLO_SUFFIX)
                reject_reason = None
                if not os.path.isfile(neff):
                    reject_reason = "NEFF is not a regular file"
                elif not _is_inside(root, neff):
                    reject_reason = "NEFF symlink escapes the allowed root"
                elif not os.path.isfile(hlo):
                    reject_reason = f"missing sibling {stem + HLO_SUFFIX}"
                elif not _is_inside(root, hlo):
                    reject_reason = "HLO symlink escapes the allowed root"
                if reject_reason:
                    rejections.append({"neff": neff, "reason": reject_reason})
                    continue
                pairs.append(ArtifactPair(
                    stem=stem, neff_path=neff, hlo_path=hlo,
                    neff_sha256=sha256_file(neff), hlo_sha256=sha256_file(hlo),
                    has_marker=_hlo_has_marker(hlo)))
    # Also fail loudly on marker-bearing HLOs whose NEFF is missing.
    paired_hlos = {os.path.realpath(p.hlo_path) for p in pairs}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
            for fn in sorted(filenames):
                if not fn.endswith(HLO_SUFFIX):
                    continue
                hlo = os.path.join(dirpath, fn)
                if os.path.realpath(hlo) in paired_hlos:
                    continue
                if os.path.isfile(hlo) and _is_inside(root, hlo) and _hlo_has_marker(hlo):
                    rejections.append({
                        "hlo": hlo,
                        "reason": "NKI-marker HLO without a sibling NEFF"})
    return pairs, rejections


def _check_phase_markers(pairs: list[ArtifactPair], *, nki_phase: bool, context,
                         roots, described) -> None:
    marked = [p for p in pairs if p.has_marker]
    if nki_phase and not marked:
        raise NkiArtifactIdentityError(
            "the NKI phase compiled no marker-bearing NEFF/HLO pair — the "
            "operator's run() launched no NKI kernel", context=context,
            roots=roots, candidates=described)
    if not nki_phase and marked:
        raise NkiArtifactIdentityError(
            f"the torch baseline phase compiled {len(marked)} NKI (marker-bearing) "
            f"pair(s) — the baseline must not launch NKI kernels", context=context,
            roots=roots, candidates=described)


def resolve_phase_pairs(roots: Sequence[str], *, nki_phase: bool,
                        exclude_stems: Iterable[str] = (),
                        context: dict | None = None) -> list[ArtifactPair]:
    """Every valid pair that appeared during a phase (all pairs in ``roots``
    minus ``exclude_stems``, sorted by stem). Raises for zero pairs, for a
    marker-bearing HLO without its NEFF, and for marker/phase mismatches
    (NKI phase without a marker pair; torch phase with one)."""
    excluded = set(exclude_stems)
    pairs, rejections = discover_pairs(roots)
    orphan_markers = [r for r in rejections
                      if r.get("reason", "").startswith("NKI-marker HLO")]
    if orphan_markers:
        raise NkiArtifactIdentityError(
            "NKI-marker HLO present without its NEFF — dump is incomplete",
            context=context, roots=roots,
            candidates=[dataclasses.asdict(p) for p in pairs] + rejections)
    new = sorted((p for p in pairs if p.stem not in excluded), key=lambda p: p.stem)
    described = [dict(dataclasses.asdict(p),
                      rejected_because=("excluded pre-existing stem" if p.stem in excluded
                                        else "NEW"))
                 for p in pairs] + rejections
    if not new:
        raise NkiArtifactIdentityError(
            "zero valid NEFF/HLO pairs appeared during this phase — cannot "
            "establish artifact identity", context=context, roots=roots,
            candidates=described)
    _check_phase_markers(new, nki_phase=nki_phase, context=context, roots=roots,
                         described=described)
    return new


def resolve_expected_pairs(roots: Sequence[str], *, stems: Sequence[str], nki_phase: bool,
                           pre_stems: Iterable[str],
                           context: dict | None = None) -> list[ArtifactPair]:
    """Reuse-mode resolution: every previously validated pair in ``stems`` must
    still exist and NO new pair may have appeared since ``pre_stems`` (a run
    that compiled a different graph would have dumped a new pair — that is an
    identity error, not a silent switch). Never selects by recency."""
    pairs, rejections = discover_pairs(roots)
    by_stem = {p.stem: p for p in pairs}
    pre = set(pre_stems)
    newcomers = [p for p in pairs if p.stem not in pre]
    described = [dataclasses.asdict(p) for p in pairs] + rejections
    if newcomers:
        raise NkiArtifactIdentityError(
            f"reuse of {list(stems)} rejected: {len(newcomers)} new pair(s) appeared "
            f"during this run ({[p.stem for p in newcomers]}) — the executed graph is "
            f"not the validated one", context=context, roots=roots, candidates=described)
    out = []
    for stem in stems:
        pair = by_stem.get(stem)
        if pair is None:
            raise NkiArtifactIdentityError(
                f"reuse of {stem!r} rejected: validated pair no longer present",
                context=context, roots=roots, candidates=described)
        out.append(pair)
    if not out:
        raise NkiArtifactIdentityError(
            "reuse rejected: no expected pairs recorded for this phase",
            context=context, roots=roots, candidates=described)
    _check_phase_markers(out, nki_phase=nki_phase, context=context, roots=roots,
                         described=described)
    return out


def validate_pair(neff_path: str, *, require_marker: bool,
                  allowed_roots: Sequence[str] | None = None,
                  context: dict | None = None) -> ArtifactPair:
    """Validate one explicit NEFF + sibling HLO (used by override and reuse)."""
    stem_dir = os.path.dirname(os.path.abspath(neff_path))
    fn = os.path.basename(neff_path)
    if not fn.endswith(NEFF_SUFFIX):
        raise NkiArtifactIdentityError(
            f"explicit NEFF {neff_path!r} does not end in {NEFF_SUFFIX}",
            context=context, roots=allowed_roots or [])
    stem = fn[: -len(NEFF_SUFFIX)]
    hlo = os.path.join(stem_dir, stem + HLO_SUFFIX)
    problems = []
    if not os.path.isfile(neff_path):
        problems.append("NEFF missing or not a regular file")
    if not os.path.isfile(hlo):
        problems.append(f"sibling HLO missing: {hlo}")
    if allowed_roots is not None:
        if not any(_is_inside(r, neff_path) for r in allowed_roots):
            problems.append("NEFF outside the allowed roots")
        if os.path.isfile(hlo) and not any(_is_inside(r, hlo) for r in allowed_roots):
            problems.append("HLO outside the allowed roots")
    if problems:
        raise NkiArtifactIdentityError(
            "explicit NEFF validation failed: " + "; ".join(problems),
            context=context, roots=allowed_roots or [],
            candidates=[{"neff": neff_path, "hlo": hlo}])
    has_marker = _hlo_has_marker(hlo)
    if require_marker and not has_marker:
        raise NkiArtifactIdentityError(
            f"explicit NEFF {neff_path!r}: sibling HLO lacks the NKI marker "
            f"{NKI_HLO_MARKER!r}", context=context, roots=allowed_roots or [],
            candidates=[{"neff": neff_path, "hlo": hlo, "marker": False}])
    return ArtifactPair(stem=stem, neff_path=neff_path, hlo_path=hlo,
                        neff_sha256=sha256_file(neff_path),
                        hlo_sha256=sha256_file(hlo), has_marker=has_marker)


def resolve_explicit_override(context: dict | None = None) -> ArtifactPair | None:
    """Validated $NKI_NEFF_PATH override, or None when the env var is unset."""
    path = os.environ.get(NEFF_PATH_ENV)
    if not path:
        return None
    return validate_pair(path, require_marker=True, context=context)


def validate_manifest_reuse(manifest: dict, *, spec_id: str,
                            allowed_roots: Sequence[str],
                            marker_targets: Sequence[str] = ("nki",)
                            ) -> dict[str, list[ArtifactPair]]:
    """Re-validate a previously written manifest for exact artifact reuse.

    Checks: manifest schema version, spec_id match, and per profiled target
    every recorded artifact (files exist, SHA256s match the manifest, NKI
    marker still present where claimed, paths inside the allowed roots).
    Artifacts are recorded whether or not the target verified, so a
    data-dependent verification failure is re-tried on reuse. Per-target
    state:

    - ``artifacts`` missing/None (the phase failed before its pairs could be
      established) -> not reusable: the private space is rebuilt, because a
      stale dump + compile-cache hit would otherwise leave that target
      unresolvable forever;
    - ``artifacts == []`` (the phase compiled nothing, e.g. an op unsupported
      on the target) -> nothing stale can exist: skipped, re-run fresh;
    - a ``marker_targets`` entry whose artifacts hold no marker-bearing pair
      -> not reusable (rebuild): the kernel graph never compiled.

    Returns the re-validated pairs per target, or raises
    NkiArtifactIdentityError. Never searches for a replacement.
    """
    ctx = {"spec_id": spec_id, "manifest": manifest.get("manifest_path", "")}
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise NkiArtifactIdentityError(
            f"manifest schema_version {manifest.get('schema_version')!r} != "
            f"current {MANIFEST_SCHEMA_VERSION!r}", context=ctx, roots=allowed_roots)
    if manifest.get("spec_id") != spec_id:
        raise NkiArtifactIdentityError(
            f"manifest spec_id {manifest.get('spec_id')!r} != current {spec_id!r}",
            context=ctx, roots=allowed_roots)
    out: dict[str, list[ArtifactPair]] = {}
    for target in manifest.get("targets", {}):
        records = manifest["targets"][target].get("artifacts")
        if records is None:
            raise NkiArtifactIdentityError(
                f"manifest target {target!r} has no artifact records (its phase "
                f"failed before identity was established) — not reusable",
                context=dict(ctx, target=target), roots=allowed_roots)
        if target in marker_targets and not any(r.get("hlo_nki_marker") for r in records):
            raise NkiArtifactIdentityError(
                f"manifest target {target!r} recorded no marker-bearing (NKI) pair — "
                f"not reusable", context=dict(ctx, target=target), roots=allowed_roots,
                candidates=list(records))
        if not records:
            continue  # compiled nothing: no stale dump, re-run fresh
        pairs = []
        for rec in records:
            pair = validate_pair(rec["neff_path"],
                                 require_marker=bool(rec.get("hlo_nki_marker")),
                                 allowed_roots=allowed_roots,
                                 context=dict(ctx, target=target))
            problems = []
            if pair.neff_sha256 != rec.get("neff_sha256"):
                problems.append(f"NEFF sha256 changed ({pair.neff_sha256} != "
                                f"{rec.get('neff_sha256')})")
            if pair.hlo_sha256 != rec.get("hlo_sha256"):
                problems.append(f"HLO sha256 changed ({pair.hlo_sha256} != "
                                f"{rec.get('hlo_sha256')})")
            if bool(rec.get("hlo_nki_marker")) != pair.has_marker:
                problems.append("NKI marker presence changed")
            if problems:
                raise NkiArtifactIdentityError(
                    f"manifest reuse validation failed for target {target!r}, "
                    f"artifact {rec.get('stem')!r}: " + "; ".join(problems),
                    context=dict(ctx, target=target), roots=allowed_roots,
                    candidates=[dataclasses.asdict(pair)])
            pairs.append(pair)
        out[target] = pairs
    if not out:
        raise NkiArtifactIdentityError(
            "manifest has no reusable targets", context=ctx, roots=allowed_roots)
    return out
