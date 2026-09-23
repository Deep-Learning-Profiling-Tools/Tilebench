#!/usr/bin/env python3
"""Download a published TileBench artifact and restore it into this checkout.

    python scripts/fetch_artifacts.py --artifact llm-aacl2026

Artifacts are described in artifacts/manifest.json: where to download the
archive, its SHA256, and which directories it provides. The archive is
downloaded to a temporary file, verified, unpacked into a temporary directory
and only then moved into place, so a failed download or a bad checksum leaves
the checkout untouched. An existing non-empty destination is never replaced
unless --force is given.

Plain HTTPS and file:// URLs work as they are; Google Drive share links are
rewritten to their direct-download form. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "artifacts" / "manifest.json"

#: Manifest values that start with this have not been filled in yet.
PLACEHOLDER = "REPLACE_WITH_"
REQUIRED_FIELDS = ("description", "url", "sha256", "archive", "destination", "provides")
_DRIVE_HOSTS = ("drive.google.com", "docs.google.com", "drive.usercontent.google.com")


class ArtifactError(Exception):
    """A problem the user can act on; reported without a traceback."""


# ── manifest ─────────────────────────────────────────────────────────────────

def load_manifest(path: Path = MANIFEST) -> dict:
    try:
        manifest = json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise ArtifactError(f"artifact manifest not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ArtifactError(f"{path} is not valid JSON: {e}") from None
    if not isinstance(manifest, dict):
        raise ArtifactError(f"{path} must map artifact names to entries")
    return manifest


def get_artifact(manifest: dict, name: str) -> dict:
    if name not in manifest:
        known = ", ".join(sorted(manifest)) or "(none)"
        raise ArtifactError(f"unknown artifact {name!r}; available: {known}")
    entry = manifest[name]
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        raise ArtifactError(f"artifact {name!r} is missing manifest field(s): {', '.join(missing)}")
    if not isinstance(entry["provides"], list) or not entry["provides"]:
        raise ArtifactError(f"artifact {name!r}: 'provides' must be a non-empty list of directories")
    for p in [entry["destination"], *entry["provides"]]:
        if PurePosixPath(p).is_absolute() or ".." in PurePosixPath(p).parts:
            raise ArtifactError(f"artifact {name!r}: path {p!r} must stay inside the repository")
    return entry


def require_published(name: str, entry: dict) -> None:
    unset = [f for f in ("url", "sha256") if str(entry[f]).startswith(PLACEHOLDER)]
    if unset:
        raise ArtifactError(
            f"artifact {name!r} has not been published yet: {' and '.join(unset)} in "
            f"artifacts/manifest.json still hold{'s' if len(unset) == 1 else ''} a placeholder. "
            f"Nothing was downloaded.")


# ── download ─────────────────────────────────────────────────────────────────

def drive_file_id(url: str) -> str | None:
    """File id of a Google Drive share link, or None for any other URL."""
    parts = urllib.parse.urlsplit(url)
    if parts.hostname not in _DRIVE_HOSTS:
        return None
    if "/folders/" in parts.path:
        raise ArtifactError(
            "the manifest URL is a Google Drive folder link, which cannot be downloaded "
            "directly; use the share link of the archive file itself")
    m = re.search(r"/d/([\w-]+)", parts.path)
    if m:
        return m.group(1)
    ids = urllib.parse.parse_qs(parts.query).get("id")
    if ids:
        return ids[0]
    raise ArtifactError(f"cannot find a file id in Google Drive URL: {url}")


def direct_url(url: str) -> str:
    file_id = drive_file_id(url)
    if file_id is None:
        return url
    return ("https://drive.usercontent.google.com/download?"
            + urllib.parse.urlencode({"id": file_id, "export": "download", "confirm": "t"}))


def drive_confirm_url(html: str) -> str | None:
    """Follow-up URL from Drive's "can't scan this file" page, if that is what html is."""
    form = re.search(r'<form[^>]*\baction="([^"]+)"', html)
    if not form:
        return None
    fields = dict(re.findall(r'<input[^>]*\bname="([^"]+)"[^>]*\bvalue="([^"]*)"', html))
    if "id" not in fields:
        return None
    return form.group(1).replace("&amp;", "&") + "?" + urllib.parse.urlencode(fields)


def _looks_like_html(path: Path) -> bool:
    with open(path, "rb") as f:
        head = f.read(512).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def download(url: str, dest: Path) -> None:
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.addheaders = [("User-Agent", "tilebench-fetch-artifacts")]
    is_drive = drive_file_id(url) is not None
    target = direct_url(url)
    for _ in range(2):                       # at most one Drive confirmation hop
        try:
            with opener.open(target, timeout=60) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
        except OSError as e:                 # URLError, HTTPError, timeouts, file://
            raise ArtifactError(f"download failed: {e}") from None
        if not (is_drive and _looks_like_html(dest)):
            return
        nxt = drive_confirm_url(dest.read_text(errors="replace"))
        if nxt is None:
            break
        target = nxt
    raise ArtifactError(
        "Google Drive returned a web page instead of the file. Check that the file is "
        "shared as 'Anyone with the link' and that the manifest URL points to the file.")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ── restore ──────────────────────────────────────────────────────────────────

def _has_files(path: Path) -> bool:
    return path.is_file() or (path.is_dir() and any(p.is_file() for p in path.rglob("*")))


def safe_members(tar: tarfile.TarFile, provides: list[str]) -> list[tarfile.TarInfo]:
    """Members that may be extracted; raises on anything that could escape or
    write outside the directories the artifact declares."""
    owned = [PurePosixPath(p) for p in provides]
    members = []
    for m in tar.getmembers():
        path = PurePosixPath(m.name)
        if path.is_absolute() or ".." in path.parts or m.name.startswith(("/", "\\")):
            raise ArtifactError(f"unsafe path in archive: {m.name!r}")
        if not (m.isfile() or m.isdir()):
            raise ArtifactError(f"archive member {m.name!r} is a link or special file; refusing")
        inside = any(path == o or o in path.parents for o in owned)
        leads_to = m.isdir() and any(path in o.parents for o in owned)
        if not (inside or leads_to):
            raise ArtifactError(
                f"archive member {m.name!r} is outside the declared directories {provides}")
        members.append(m)
    return members


def check_destination(entry: dict, repo_root: Path, force: bool) -> None:
    dest_root = repo_root / entry["destination"]
    occupied = [p for p in entry["provides"] if _has_files(dest_root / p)]
    if occupied and not force:
        raise ArtifactError(
            f"{', '.join(occupied)} already exist{'s' if len(occupied) == 1 else ''} and "
            f"{'is' if len(occupied) == 1 else 'are'} not empty. "
            f"Re-run with --force to replace {'it' if len(occupied) == 1 else 'them'}.")


def restore(entry: dict, archive: Path, repo_root: Path, force: bool) -> int:
    """Unpack archive and move the provided directories into place. Returns the file count."""
    check_destination(entry, repo_root, force)
    dest_root = repo_root / entry["destination"]
    dest_root.mkdir(parents=True, exist_ok=True)
    # Staging lives next to the destination so the final moves are same-filesystem renames.
    staging = Path(tempfile.mkdtemp(prefix=".tilebench-artifact-", dir=dest_root))
    try:
        unpacked, replaced = staging / "new", staging / "old"
        try:
            with tarfile.open(archive, "r:*") as tar:
                members = safe_members(tar, entry["provides"])
                extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                tar.extractall(unpacked, members=members, **extra)
        except tarfile.TarError as e:
            raise ArtifactError(f"cannot unpack {archive.name}: {e}") from None
        for p in entry["provides"]:
            if not _has_files(unpacked / p):
                raise ArtifactError(f"archive does not contain the declared directory {p!r}")

        moved = []                              # (target, backup or None), for rollback
        try:
            for p in entry["provides"]:
                target, backup = dest_root / p, None
                if target.exists():
                    backup = replaced / p
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, backup)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(unpacked / p, target)
                moved.append((target, backup))
        except OSError:
            for target, backup in reversed(moved):
                shutil.rmtree(target, ignore_errors=True)
                if backup is not None:
                    os.replace(backup, target)
            raise
        return sum(1 for p in entry["provides"] for f in (dest_root / p).rglob("*") if f.is_file())
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def fetch(name: str, *, manifest_path: Path = MANIFEST, repo_root: Path = REPO_ROOT,
          force: bool = False) -> dict:
    entry = get_artifact(load_manifest(manifest_path), name)
    require_published(name, entry)
    check_destination(entry, repo_root, force)      # fail before downloading anything
    with tempfile.TemporaryDirectory(prefix="tilebench-fetch-") as tmp:
        archive = Path(tmp) / entry["archive"]
        download(entry["url"], archive)
        actual = sha256_of(archive)
        if actual != entry["sha256"].lower():
            raise ArtifactError(
                f"SHA256 mismatch for {entry['archive']}: expected {entry['sha256']}, got {actual}. "
                f"Nothing was restored.")
        size = archive.stat().st_size
        count = restore(entry, archive, repo_root, force)
    return {"name": name, "archive": entry["archive"], "bytes": size, "sha256": actual,
            "restored": [str(Path(entry["destination"]) / p) for p in entry["provides"]],
            "files": count}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--artifact", required=True, metavar="NAME",
                    help="artifact name from artifacts/manifest.json, e.g. llm-aacl2026")
    ap.add_argument("--force", action="store_true",
                    help="replace the destination if it already exists and is not empty")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help=argparse.SUPPRESS)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        s = fetch(args.artifact, manifest_path=args.manifest,
                  repo_root=args.repo_root.resolve(), force=args.force)
    except ArtifactError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"artifact   {s['name']}")
    print(f"archive    {s['archive']} ({s['bytes'] / 1e6:.1f} MB)")
    print(f"sha256     {s['sha256']} (verified)")
    print(f"restored   {', '.join(s['restored'])}")
    print(f"files      {s['files']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
