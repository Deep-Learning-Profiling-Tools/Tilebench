"""Skill loader.

The asset manifest (`skills/manifest.json` at the repository root) names
every file the framework may inject, with its raw hash, the hash of the text
after the documented transformation, its permission label and its approval
status. The loader never browses directories, never falls back to another
version, DSL or device, and never truncates.

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
# What may be sent to a third-party API provider.
API_SENDABLE = {"public", "internal"}


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


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
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


def load_component(manifest: dict, kind: str, key: str, version: str | None = None, *,
                   require_status: tuple[str, ...] = ("approved", "frozen"),
                   provider_sendable: bool = True, max_chars: int | None = None,
                   verify_hashes: bool = True) -> SkillComponent:
    if kind not in KINDS:
        raise SkillError(f"unknown skill kind {kind}")
    version, entry = _entry(manifest, kind, key, version)
    permission = entry.get("permission")
    status = entry.get("status")
    if permission not in PERMISSIONS:
        raise SkillPermissionError(f"{kind}/{key}@{version}: permission {permission!r} is not one of {PERMISSIONS}")
    if provider_sendable and permission not in API_SENDABLE:
        raise SkillPermissionError(f"{kind}/{key}@{version}: permission {permission!r} may not be sent to an API provider")
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
    return SkillComponent(kind=kind, key=key, version=version, path=entry["path"], sha256_raw=h_raw,
                          sha256_injected=h_inj, permission=permission, status=status, text=text,
                          chars=len(text), attachments=[{k: v for k, v in a.items() if k != "text"} for a in attachments])


def compose_context(manifest: dict, study: dict, *, dsl: str, device: str, fold: str,
                    condition: str, require_approved: bool = True,
                    reference_version: str | None = None, device_snapshot: str | None = None,
                    optimization_version: str | None = None) -> list[SkillComponent]:
    """The ordered skill components of one condition. Base never includes an
    Optimization Skill; Enhanced refuses to start without the frozen one for
    (dsl, fold), and refuses a device outside the DSL's transfer list."""
    if condition not in study["conditions"]:
        raise SkillError(f"unknown condition {condition}")
    limits = study["context_limits"]
    statuses = ("approved", "frozen") if require_approved else ("draft", "approved", "frozen", "test-only")
    comps = [
        load_component(manifest, "reference", dsl, reference_version or study["dsls"][dsl]["reference_version"],
                       require_status=statuses, max_chars=limits["reference_skill"]),
        load_component(manifest, "device", device, device_snapshot, require_status=statuses,
                       max_chars=limits["device_context_skill"]),
    ]
    if condition == "enhanced":
        if device not in study["skill_transfer"][dsl]:
            raise SkillError(f"{device} is not a transfer target of {dsl} Optimization Skills")
        opt = load_component(manifest, "optimization", f"{dsl}/{fold}", optimization_version,
                             require_status=("frozen",) if require_approved else ("frozen", "test-only"),
                             max_chars=limits["optimization_skill"])
        comps.append(opt)
    return comps


def hash_record(components: list[SkillComponent]) -> dict:
    return {f"{c.kind}:{c.key}@{c.version}": c.sha256_injected for c in components}


def register_asset(manifest: dict, kind: str, key: str, version: str, rel_path: str, *,
                   permission: str, status: str, source: str, attachments: list[str] | None = None,
                   extra: dict | None = None) -> dict:
    """Compute hashes for an asset and add/replace its manifest entry (dev CLI)."""
    text, h_raw, h_inj = _read_asset(rel_path, None, None, verify_hashes=False)
    entry = {"path": rel_path, "sha256_raw": h_raw, "sha256_injected": h_inj, "chars": len(text),
             "permission": permission, "status": status, "source": source, "attachments": []}
    for att in attachments or []:
        _, a_raw, a_inj = _read_asset(att, None, None, verify_hashes=False)
        entry["attachments"].append({"path": att, "sha256_raw": a_raw, "sha256_injected": a_inj})
    entry.update(extra or {})
    manifest.setdefault(kind, {}).setdefault(key, {})[version] = entry
    return entry


def save_manifest(manifest: dict, path: Path = MANIFEST_PATH) -> None:
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
