#!/usr/bin/env python3
"""Package an artifact declared in artifacts/manifest.json for publication.

    python scripts/package_artifacts.py --artifact llm-aacl2026

Writes outputs/artifacts/<archive> (git-ignored) holding the artifact's
directories at their repository-relative paths, so fetch_artifacts.py can
unpack it at the repository root and everything lands in place. The archive is
deterministic: the same files always produce the same SHA256, which is printed
for artifacts/manifest.json. The manifest is never modified from here.
"""

from __future__ import annotations

import argparse
import gzip
import sys
import tarfile
from pathlib import Path

from fetch_artifacts import (MANIFEST, PLACEHOLDER, REPO_ROOT, ArtifactError,
                             get_artifact, load_manifest, sha256_of)

OUT_DIR = REPO_ROOT / "outputs" / "artifacts"
_MTIME = 1767225600          # 2026-01-01T00:00:00Z: a fixed timestamp keeps the archive reproducible


def artifact_files(root: Path) -> list[Path]:
    """Every file of an artifact directory, minus interpreter caches."""
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts)


def _normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = _MTIME
    info.mode = 0o644
    return info


def build(entry: dict, repo_root: Path, out_path: Path) -> int:
    src_root = repo_root / entry["destination"]
    files = []
    for p in entry["provides"]:
        found = artifact_files(src_root / p)
        if not found:
            raise ArtifactError(f"nothing to package: {src_root / p} is missing or empty")
        files += found
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for f in files:
            tar.add(f, arcname=f.relative_to(src_root).as_posix(), recursive=False,
                    filter=_normalized)
    return len(files)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--artifact", required=True, metavar="NAME",
                    help="artifact name from artifacts/manifest.json, e.g. llm-aacl2026")
    ap.add_argument("--output", type=Path, default=None,
                    help="archive path (default: outputs/artifacts/<archive from the manifest>)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help=argparse.SUPPRESS)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        entry = get_artifact(load_manifest(args.manifest), args.artifact)
        out = args.output or OUT_DIR / entry["archive"]
        count = build(entry, args.repo_root.resolve(), out)
    except ArtifactError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    digest = sha256_of(out)
    print(f"artifact   {args.artifact}")
    print(f"archive    {out} ({out.stat().st_size / 1e6:.1f} MB, {count} files)")
    print(f"sha256     {digest}")
    recorded = str(entry["sha256"])
    if recorded.startswith(PLACEHOLDER):
        print(f"\nartifacts/manifest.json still has a placeholder for {args.artifact!r}. After "
              f"uploading the archive, set\n  \"sha256\": \"{digest}\"\nand \"url\" to the "
              f"file's share link.")
    elif recorded.lower() != digest:
        print(f"\nwarning: artifacts/manifest.json records sha256 {recorded}, which differs from "
              f"this build. Publishing this archive requires updating the manifest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
