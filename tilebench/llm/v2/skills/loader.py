"""Skill loader.

The asset manifest (`skills/manifest.json` at the repository root) names
every file the framework may inject, with its raw hash, the hash of the text
after the documented transformation, its permission label, its approval
status and its sharing grants. The loader never browses directories, never
falls back to another version, DSL or device, and never truncates.

Three independent decisions are recorded per asset and checked separately:

  status        draft | approved | frozen | test-only   (content approval)
  sendable_to   list of provider names (openai, anthropic, ...) the asset
                may be sent to as prompt context; empty = not confirmed.
                `internal` and even `public` labels do not imply a grant.
  publishable   whether the body may leave the private repository (review
                bundles, publication exports)

`permission` (public | internal | private) describes the asset's origin;
`private` bodies are never stored in the repository and are never sendable
or publishable regardless of the other fields.

Transformation rule (`transform()`), applied identically to every asset:
  1. decode UTF-8;  2. CRLF/CR -> LF;  3. drop a leading YAML front-matter block
  (`---` ... `---`);  4. strip trailing whitespace on every line;  5. strip
  blank lines at both ends and end with exactly one newline.
Both sha256(raw bytes) and sha256(transformed text) are recorded."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tilebench.paths import REPO_ROOT

MANIFEST_PATH = REPO_ROOT / "skills" / "manifest.json"
MANIFEST_SCHEMA = "tilebench-skill-manifest/1"

KINDS = ("reference", "device", "optimization", "contract")
PERMISSIONS = ("public", "internal", "private")
STATUSES = ("draft", "approved", "frozen", "test-only")
PROVIDERS = ("openai", "anthropic")


class SkillError(RuntimeError):
    pass


class SkillMissingError(SkillError):
    pass


class SkillHashError(SkillError):
    pass


class SkillPermissionError(SkillError):
    pass


class SkillStatusError(SkillError):
    pass


class SkillTooLongError(SkillError):
    pass


@dataclass
class SkillComponent:
    kind: str
    key: str                 # dsl, device, or "<dsl>/<fold>" for optimization
    version: str
    path: str
    sha256_raw: str
    sha256_injected: str
    permission: str
    status: str
    text: str
    chars: int
    attachments: list[dict]
    sendable_to: list[str]
    publishable: bool
    sha256_composed: str = ""          # sha256 of the COMPLETE injected text (main body + ordered attachments)
    provenance: dict | None = None     # validated optimization-skill manifest (kind == optimization)

    def record(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


def transform(raw: bytes) -> str:
    text = raw.decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            text = text[end + 5:]
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path | None = None) -> dict:
    path = path if path is not None else MANIFEST_PATH   # resolved at call time (tests monkeypatch it)
    if not path.exists():
        raise SkillMissingError(f"skill manifest missing: {path}")
    data = json.loads(path.read_text())
    if data.get("schema") != MANIFEST_SCHEMA:
        raise SkillError(f"{path}: schema must be {MANIFEST_SCHEMA}")
    for kind in KINDS:
        data.setdefault(kind, {})
    return data


def _entry(manifest: dict, kind: str, key: str, version: str | None) -> tuple[str, dict]:
    bucket = manifest.get(kind, {}).get(key)
    if not bucket:
        raise SkillMissingError(f"no {kind} skill registered for {key!r}")
    if version is None:
        if len(bucket) != 1:
            raise SkillMissingError(f"{kind}/{key}: version must be named explicitly (registered: {sorted(bucket)})")
        version = next(iter(bucket))
    if version not in bucket:
        raise SkillMissingError(f"{kind}/{key}: version {version!r} is not registered (registered: {sorted(bucket)})")
    return version, bucket[version]


def _read_asset(rel_path: str | None, expect_raw: str | None, expect_injected: str | None,
                verify_hashes: bool) -> tuple[str, str, str]:
    if not rel_path:
        raise SkillMissingError("asset body is not stored in this repository (private asset: metadata only)")
    path = REPO_ROOT / rel_path
    if not path.exists():
        raise SkillMissingError(f"asset file missing: {rel_path}")
    raw = path.read_bytes()
    text = transform(raw)
    h_raw, h_inj = sha256_bytes(raw), sha256_bytes(text.encode("utf-8"))
    if verify_hashes:
        if expect_raw != h_raw:
            raise SkillHashError(f"{rel_path}: raw sha256 {h_raw[:12]} != manifest {str(expect_raw)[:12]}")
        if expect_injected != h_inj:
            raise SkillHashError(f"{rel_path}: injected sha256 {h_inj[:12]} != manifest {str(expect_injected)[:12]}")
    return text, h_raw, h_inj


def sendable_to(entry: dict) -> list[str]:
    if entry.get("permission") == "private":
        return []
    return [p for p in (entry.get("sendable_to") or []) if isinstance(p, str)]


def is_publishable(entry: dict) -> bool:
    if entry.get("permission") == "private":
        return False
    return bool(entry.get("publishable", False))


def load_component(manifest: dict, kind: str, key: str, version: str | None = None, *,
                   require_status: tuple[str, ...] = ("approved", "frozen"),
                   provider_sendable: bool = True, provider: str | None = None,
                   max_chars: int | None = None, verify_hashes: bool = True) -> SkillComponent:
    """provider_sendable=True means the text is being loaded to be sent to a
    model provider: private assets are refused, and when `provider` is
    named the manifest must list it in the asset's `sendable_to` grant."""
    if kind not in KINDS:
        raise SkillError(f"unknown skill kind {kind}")
    version, entry = _entry(manifest, kind, key, version)
    permission = entry.get("permission")
    status = entry.get("status")
    if permission not in PERMISSIONS:
        raise SkillPermissionError(f"{kind}/{key}@{version}: permission {permission!r} is not one of {PERMISSIONS}")
    grants = sendable_to(entry)
    if provider_sendable:
        if permission == "private":
            raise SkillPermissionError(f"{kind}/{key}@{version}: permission 'private' may not be sent to an API provider")
        if provider is not None and provider not in grants:
            raise SkillPermissionError(f"{kind}/{key}@{version}: sending to provider {provider!r} is not granted "
                                       f"(sendable_to={grants}; permission={permission!r})")
    if status not in STATUSES:
        raise SkillStatusError(f"{kind}/{key}@{version}: status {status!r} is not one of {STATUSES}")
    if require_status and status not in require_status:
        raise SkillStatusError(f"{kind}/{key}@{version}: status {status!r}, required one of {require_status}")
    text, h_raw, h_inj = _read_asset(entry["path"], entry.get("sha256_raw"), entry.get("sha256_injected"), verify_hashes)
    attachments = []
    for att in entry.get("attachments", []):
        a_text, a_raw, a_inj = _read_asset(att["path"], att.get("sha256_raw"), att.get("sha256_injected"), verify_hashes)
        attachments.append({"path": att["path"], "sha256_raw": a_raw, "sha256_injected": a_inj, "text": a_text})
        text += "\n" + a_text
    if max_chars is not None and len(text) > max_chars:
        raise SkillTooLongError(f"{kind}/{key}@{version}: {len(text)} chars exceed the limit {max_chars}")
    comp = SkillComponent(kind=kind, key=key, version=version, path=entry["path"], sha256_raw=h_raw,
                          sha256_injected=h_inj, permission=permission, status=status, text=text,
                          chars=len(text), attachments=[{k: v for k, v in a.items() if k != "text"} for a in attachments],
                          sendable_to=grants, publishable=is_publishable(entry),
                          sha256_composed=sha256_bytes(text.encode("utf-8")))
    if kind == "optimization":
        comp.provenance = read_optimization_manifest(entry["path"])
    return comp


# --------------------------------------------------------------------------
# Optimization-skill provenance (independent manifest beside SKILL.md)
# --------------------------------------------------------------------------

OPTIMIZATION_MANIFEST_SCHEMAS = ("tilebench-optimization-skill/1", "tilebench-optimization-skill/2")


def read_optimization_manifest(rel_path: str) -> dict:
    mpath = (REPO_ROOT / rel_path).parent / "manifest.json"
    if not mpath.exists():
        raise SkillError(f"optimization skill {rel_path}: no manifest.json beside it (provenance unknown)")
    try:
        m = json.loads(mpath.read_text())
    except ValueError as e:
        raise SkillError(f"optimization skill {rel_path}: manifest.json is not valid JSON: {e}")
    if m.get("schema") not in OPTIMIZATION_MANIFEST_SCHEMAS:
        raise SkillError(f"optimization skill {rel_path}: manifest schema {m.get('schema')!r} not in {OPTIMIZATION_MANIFEST_SCHEMAS}")
    return m


def validate_optimization_provenance(comp: SkillComponent, study: dict, *, dsl: str, fold: str,
                                     mode: str = "evaluation", require_status: tuple[str, ...] = ("frozen",)) -> None:
    """The Enhanced condition may inject a skill only when its own manifest
    says it was distilled for exactly this use: same DSL and compatible
    reference version, the DSL's source device, the requested mode
    (`evaluation` with THIS fold held out and the other two as training
    folds; `release` only when release is requested), non-empty source
    trajectory ids, a content hash matching the injected body, and a status
    in `require_status`. A manifest field is never inferred from the manifest
    key (`<dsl>/<fold>`): the key only selects the entry."""
    from tilebench.llm.v2.manifests.schema import training_folds
    m = comp.provenance or {}
    where = f"optimization/{comp.key}@{comp.version}"
    problems = []
    if m.get("dsl") != dsl:
        problems.append(f"dsl {m.get('dsl')!r} != {dsl!r}")
    ref_version = study["dsls"][dsl]["reference_version"]
    if ref_version not in (m.get("compatible_versions") or []):
        problems.append(f"compatible_versions {m.get('compatible_versions')!r} does not include the study reference version {ref_version!r}")
    src = study["dsls"][dsl]["source_device"]
    if m.get("source_device") != src:
        problems.append(f"source_device {m.get('source_device')!r} != {src!r}")
    got_mode = m.get("evaluation_or_release") or m.get("mode")
    if got_mode != mode:
        problems.append(f"mode {got_mode!r} != requested {mode!r}")
    if mode == "evaluation":
        if m.get("held_out_fold") != fold:
            problems.append(f"held_out_fold {m.get('held_out_fold')!r} != {fold!r}")
        if sorted(m.get("training_folds") or []) != sorted(training_folds(fold)):
            problems.append(f"training_folds {m.get('training_folds')!r} != {sorted(training_folds(fold))}")
    if not m.get("source_trajectory_ids"):
        problems.append("source_trajectory_ids empty")
    body_hash = sha256_bytes(comp.text.encode("utf-8"))
    raw_hash = sha256_bytes((REPO_ROOT / comp.path).read_bytes())
    raw_text_hash = sha256_bytes((REPO_ROOT / comp.path).read_text().encode("utf-8"))
    if m.get("content_sha256") not in (body_hash, raw_hash, raw_text_hash):
        problems.append("content_sha256 does not match the skill body")
    if m.get("status") not in require_status:
        problems.append(f"manifest status {m.get('status')!r} not in {require_status}")
    if comp.status not in require_status:
        problems.append(f"registry status {comp.status!r} not in {require_status}")
    if problems:
        raise SkillPermissionError(f"{where}: provenance rejected: " + "; ".join(problems))


def compose_context(manifest: dict, study: dict, *, dsl: str, device: str, fold: str,
                    condition: str, require_approved: bool = True,
                    reference_version: str | None = None, device_snapshot: str | None = None,
                    optimization_version: str | None = None, provider: str | None = None,
                    provider_sendable: bool = True, optimization_mode: str = "evaluation") -> list[SkillComponent]:
    """The ordered skill components of one condition. Base never includes an
    Optimization Skill; Enhanced refuses to start without the frozen one for
    (dsl, fold), and refuses a device outside the DSL's transfer list. With
    `provider` named, every component must carry that provider's grant."""
    if condition not in study["conditions"]:
        raise SkillError(f"unknown condition {condition}")
    limits = study["context_limits"]
    statuses = ("approved", "frozen") if require_approved else ("draft", "approved", "frozen", "test-only")
    common = {"provider": provider, "provider_sendable": provider_sendable}
    comps = [
        load_component(manifest, "reference", dsl, reference_version or study["dsls"][dsl]["reference_version"],
                       require_status=statuses, max_chars=limits["reference_skill"], **common),
        load_component(manifest, "device", device, device_snapshot or (study.get("device_snapshots") or {}).get(device),
                       require_status=statuses, max_chars=limits["device_context_skill"], **common),
    ]
    if condition == "enhanced":
        if device not in study["skill_transfer"][dsl]:
            raise SkillError(f"{device} is not a transfer target of {dsl} Optimization Skills")
        statuses_opt = ("frozen",) if require_approved else ("frozen", "test-only")
        opt = load_component(manifest, "optimization", f"{dsl}/{fold}", optimization_version,
                             require_status=statuses_opt, max_chars=limits["optimization_skill"], **common)
        validate_optimization_provenance(opt, study, dsl=dsl, fold=fold, mode=optimization_mode, require_status=statuses_opt)
        comps.append(opt)
    return comps


def hash_record(components: list[SkillComponent]) -> dict:
    """Component key -> sha256 of the COMPLETE injected text (main body plus
    ordered attachments). For an asset without attachments this equals the
    manifest's sha256_injected; with attachments it differs from it."""
    return {f"{c.kind}:{c.key}@{c.version}": (c.sha256_composed or c.sha256_injected) for c in components}


def hash_record_detailed(components: list[SkillComponent]) -> dict:
    return {f"{c.kind}:{c.key}@{c.version}": {"sha256_composed": c.sha256_composed, "sha256_injected_main": c.sha256_injected,
                                             "sha256_raw_main": c.sha256_raw, "attachments": c.attachments}
            for c in components}


def register_asset(manifest: dict, kind: str, key: str, version: str, rel_path: str, *,
                   permission: str, status: str, source: str, attachments: list[str] | None = None,
                   extra: dict | None = None, sendable_to: list[str] | None = None,
                   publishable: bool | None = None) -> dict:
    """Compute hashes for an asset and add/replace its manifest entry (dev CLI).
    Grants default to NOTHING: `sendable_to` empty and `publishable` false
    unless given explicitly (a private asset can never carry either)."""
    text, h_raw, h_inj = _read_asset(rel_path, None, None, verify_hashes=False)
    grants = [] if permission == "private" else list(sendable_to or [])
    for p in grants:
        if p not in PROVIDERS:
            raise SkillError(f"unknown provider {p!r} in sendable_to (known: {PROVIDERS})")
    entry = {"path": rel_path, "sha256_raw": h_raw, "sha256_injected": h_inj, "chars": len(text),
             "permission": permission, "status": status, "source": source, "attachments": [],
             "sendable_to": grants, "publishable": bool(publishable) and permission != "private"}
    for att in attachments or []:
        _, a_raw, a_inj = _read_asset(att, None, None, verify_hashes=False)
        entry["attachments"].append({"path": att, "sha256_raw": a_raw, "sha256_injected": a_inj})
    entry.update(extra or {})
    manifest.setdefault(kind, {}).setdefault(key, {})[version] = entry
    return entry


def save_manifest(manifest: dict, path: Path | None = None) -> None:
    path = path if path is not None else MANIFEST_PATH
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
