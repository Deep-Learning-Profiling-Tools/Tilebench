"""artifacts/manifest.json, scripts/fetch_artifacts.py and scripts/package_artifacts.py.

Everything runs against temporary directories and file:// URLs: no network, and
never this checkout's own artifact directory."""
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
LLM = "tilebench/benchmarks/llm_generated"
NAME = "llm-test"


def _load(name):
    sys.path.insert(0, str(SCRIPTS))          # package_artifacts imports fetch_artifacts
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


fa = _load("fetch_artifacts")
pa = _load("package_artifacts")


def write(path, text="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def entry(**over):
    e = {"description": "test", "url": "REPLACE_WITH_GOOGLE_DRIVE_OR_DIRECT_DOWNLOAD_URL",
         "sha256": "REPLACE_WITH_SHA256", "archive": "a.tar.gz", "destination": ".",
         "provides": [LLM]}
    return {**e, **over}


@pytest.fixture
def published(tmp_path):
    """A packaged artifact, a manifest that points at it, and an empty repo to restore into."""
    src = tmp_path / "source_repo"
    write(src / LLM / "relu/gpt/high/iter_0/prompt.md", "prompt\n")
    write(src / LLM / "relu/gpt/high/final/impl_triton.py", "kernel\n")
    write(src / LLM / "relu/gpt/high/iter_0/__pycache__/impl.cpython-310.pyc", "bytecode")
    archive = tmp_path / "a.tar.gz"
    assert pa.build(entry(), src, archive) == 2
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({NAME: entry(url=archive.as_uri(), sha256=fa.sha256_of(archive))}))
    target = tmp_path / "target_repo"
    target.mkdir()
    return manifest, target, archive


# ── manifest ─────────────────────────────────────────────────────────────────

def test_repository_manifest_is_valid_and_unpublished():
    manifest = fa.load_manifest()
    e = fa.get_artifact(manifest, "llm-aacl2026")
    assert e["provides"] == [LLM] and e["destination"] == "."
    assert e["archive"] == "tilebench-aacl2026-llm-artifacts.tar.gz"
    # No URL or checksum is invented: both stay placeholders until the upload exists.
    if e["url"].startswith(fa.PLACEHOLDER):
        with pytest.raises(fa.ArtifactError, match="not been published"):
            fa.require_published("llm-aacl2026", e)


def test_unknown_artifact_lists_the_available_ones():
    with pytest.raises(fa.ArtifactError, match="available: a, b"):
        fa.get_artifact({"a": entry(), "b": entry()}, "c")


def test_manifest_entry_must_be_complete_and_stay_inside_the_repo():
    broken = entry()
    del broken["sha256"]
    with pytest.raises(fa.ArtifactError, match="sha256"):
        fa.get_artifact({"x": broken}, "x")
    with pytest.raises(fa.ArtifactError, match="inside the repository"):
        fa.get_artifact({"x": entry(provides=["../elsewhere"])}, "x")


def test_placeholder_is_reported_before_any_download(tmp_path, monkeypatch):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({NAME: entry()}))
    monkeypatch.setattr(fa, "download", lambda *a: pytest.fail("must not download"))
    with pytest.raises(fa.ArtifactError, match="placeholder"):
        fa.fetch(NAME, manifest_path=manifest, repo_root=tmp_path)


# ── download + restore ───────────────────────────────────────────────────────

def test_fetch_restores_to_the_canonical_path(published):
    manifest, target, _ = published
    s = fa.fetch(NAME, manifest_path=manifest, repo_root=target)
    assert (target / LLM / "relu/gpt/high/iter_0/prompt.md").read_text() == "prompt\n"
    assert (target / LLM / "relu/gpt/high/final/impl_triton.py").exists()
    assert s["files"] == 2 and s["restored"] == [LLM]
    assert not list(target.rglob("*.pyc"))                       # caches are never packaged
    assert not [p for p in target.iterdir() if p.name.startswith(".tilebench-artifact-")]


def test_checksum_mismatch_restores_nothing(published):
    manifest, target, _ = published
    m = json.loads(manifest.read_text())
    m[NAME]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(m))
    with pytest.raises(fa.ArtifactError, match="SHA256 mismatch"):
        fa.fetch(NAME, manifest_path=manifest, repo_root=target)
    assert list(target.iterdir()) == []


def test_existing_destination_is_kept_unless_forced(published):
    manifest, target, _ = published
    write(target / LLM / "mine/notes.md", "local work\n")
    with pytest.raises(fa.ArtifactError, match="--force"):
        fa.fetch(NAME, manifest_path=manifest, repo_root=target)
    assert (target / LLM / "mine/notes.md").read_text() == "local work\n"

    fa.fetch(NAME, manifest_path=manifest, repo_root=target, force=True)
    assert not (target / LLM / "mine").exists()                  # replaced, not merged
    assert (target / LLM / "relu/gpt/high/iter_0/prompt.md").exists()


def test_empty_destination_directory_is_not_an_obstacle(published):
    manifest, target, _ = published
    (target / LLM).mkdir(parents=True)
    assert fa.fetch(NAME, manifest_path=manifest, repo_root=target)["files"] == 2


def test_failed_download_leaves_no_partial_artifact(published):
    manifest, target, archive = published
    archive.unlink()
    with pytest.raises(fa.ArtifactError, match="download failed"):
        fa.fetch(NAME, manifest_path=manifest, repo_root=target)
    assert list(target.iterdir()) == []


def _tar_with(tmp_path, *members):
    path = tmp_path / "evil.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return path


def _file(name, data=b"x"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def _symlink(name, target):
    info = tarfile.TarInfo(name)
    info.type, info.linkname = tarfile.SYMTYPE, target
    return info, None


@pytest.mark.parametrize("member", [
    _file("../escaped.txt"),
    _file(f"{LLM}/../../../escaped.txt"),
    _file("/tmp/absolute.txt"),
    _file("tilebench/core/engine.py"),                    # inside the repo, outside 'provides'
    _symlink(f"{LLM}/link", "/etc/passwd"),
], ids=["dotdot", "nested-dotdot", "absolute", "outside-provides", "symlink"])
def test_unsafe_archives_are_rejected(tmp_path, member):
    archive = _tar_with(tmp_path, _file(f"{LLM}/ok.md"), member)
    target = tmp_path / "repo"
    target.mkdir()
    with pytest.raises(fa.ArtifactError):
        fa.restore(entry(), archive, target, force=False)
    assert list(target.iterdir()) == []
    assert not (tmp_path / "escaped.txt").exists()


# ── Google Drive URL handling (no network) ───────────────────────────────────

@pytest.mark.parametrize("url", [
    "https://drive.google.com/file/d/1AbC_d-9/view?usp=sharing",
    "https://drive.google.com/uc?id=1AbC_d-9&export=download",
    "https://drive.google.com/open?id=1AbC_d-9",
])
def test_drive_share_links_become_direct_downloads(url):
    assert fa.drive_file_id(url) == "1AbC_d-9"
    direct = fa.direct_url(url)
    assert direct.startswith("https://drive.usercontent.google.com/download?")
    assert "id=1AbC_d-9" in direct and "confirm=t" in direct


def test_other_urls_are_left_alone_and_folder_links_are_refused():
    plain = "https://example.org/a.tar.gz"
    assert fa.drive_file_id(plain) is None and fa.direct_url(plain) == plain
    with pytest.raises(fa.ArtifactError, match="folder link"):
        fa.direct_url("https://drive.google.com/drive/folders/1LiPyNIbhA40")


def test_drive_confirmation_page_is_followed():
    html = ('<form id="download-form" action="https://drive.usercontent.google.com/download" method="get">'
            '<input type="hidden" name="id" value="1AbC"><input type="hidden" name="confirm" value="t">'
            '<input type="hidden" name="uuid" value="u-1"></form>')
    url = fa.drive_confirm_url(html)
    assert url.startswith("https://drive.usercontent.google.com/download?")
    assert "id=1AbC" in url and "uuid=u-1" in url
    assert fa.drive_confirm_url("<html><body>Sign in</body></html>") is None


# ── packaging ────────────────────────────────────────────────────────────────

def test_package_layout_is_repo_relative_and_reproducible(published, tmp_path):
    _, _, archive = published
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert names == sorted(names)
    assert all(n.startswith(LLM + "/") for n in names) and not any("pyc" in n for n in names)
    again = tmp_path / "again.tar.gz"
    pa.build(entry(), tmp_path / "source_repo", again)
    assert fa.sha256_of(again) == fa.sha256_of(archive)


def test_packaging_an_empty_artifact_fails(tmp_path):
    with pytest.raises(fa.ArtifactError, match="nothing to package"):
        pa.build(entry(), tmp_path, tmp_path / "out.tar.gz")


# ── command line + repository policy ─────────────────────────────────────────

def test_cli_runs_from_any_directory(published, tmp_path):
    manifest, target, _ = published
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    r = subprocess.run([sys.executable, str(SCRIPTS / "fetch_artifacts.py"), "--artifact", NAME,
                        "--manifest", str(manifest), "--repo-root", str(target)],
                       cwd=elsewhere, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "verified" in r.stdout and LLM in r.stdout and "files      2" in r.stdout
    again = subprocess.run([sys.executable, str(SCRIPTS / "fetch_artifacts.py"), "--artifact", NAME,
                            "--manifest", str(manifest), "--repo-root", str(target)],
                           cwd=elsewhere, capture_output=True, text=True)
    assert again.returncode == 1 and "--force" in again.stderr and "Traceback" not in again.stderr


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(),
                    reason="needs the git checkout")
def test_llm_generated_is_git_ignored_but_the_pipeline_source_is_not():
    def ignored(path):
        return subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=REPO).returncode == 0
    assert ignored(f"{LLM}/relu/gpt-5.5/high/iter_0/prompt.md")
    assert ignored(f"{LLM}/new_op/new_model/high/final/impl_triton.py")
    assert not ignored("tilebench/llm_codegen/generate.py")
    assert not ignored("artifacts/manifest.json")
    tracked = subprocess.run(["git", "ls-files", LLM], cwd=REPO, capture_output=True, text=True).stdout
    assert tracked.strip() == ""
