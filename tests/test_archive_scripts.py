"""scripts/archive_artifacts.sh and its archive_logs.sh wrapper, run against a
throwaway repository (never the real one)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "scripts" / "archive_artifacts.sh"
LOGS = REPO / "scripts" / "archive_logs.sh"
BRANCH = "archive/raw-logs-2026-09-18"
LLM = "tilebench/benchmarks/llm_generated"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

ENV = {**os.environ,
       "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
       "GIT_CONFIG_NOSYSTEM": "1"}


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, env=ENV, check=True,
                          capture_output=True, text=True).stdout.strip()


def run(script, cwd, *args):
    return subprocess.run(["bash", str(script), *args], cwd=cwd, env=ENV,
                          capture_output=True, text=True)


def archived(work):
    return set(git(work, "ls-tree", "-r", "--name-only", BRANCH).splitlines())


def write(path, text="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def work(tmp_path):
    """A clone whose archive branch already holds a legacy LLM snapshot that
    main has since dropped, plus git-ignored artifacts on disk."""
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(tmp_path, "init", "-q", str(work))
    git(work, "checkout", "-q", "-b", "main")
    git(work, "remote", "add", "origin", str(origin))
    write(work / "src.txt")
    write(work / ".gitignore", f"results/*\n{LLM}/\n__pycache__/\n*.pyc\n")
    write(work / "benchmarks/llm_generated/legacy.txt", "legacy\n")
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "main with the legacy snapshot")
    git(work, "push", "-q", "origin", "main")
    git(work, "push", "-q", "origin", f"main:refs/heads/{BRANCH}")   # archive starts here
    git(work, "rm", "-q", "-r", "benchmarks/llm_generated")
    git(work, "commit", "-q", "-m", "main drops the legacy snapshot")
    git(work, "push", "-q", "origin", "main")
    git(work, "fetch", "-q", "origin")

    write(work / "results/logs/a.json", "{}\n")
    write(work / LLM / "op/model/high/iter_0/prompt.md", "prompt\n")
    write(work / LLM / "op/model/high/iter_0/__pycache__/impl.cpython-310.pyc", "bytecode")
    return work


def test_mode_is_required(work):
    r = run(ARTIFACTS, work)
    assert r.returncode == 2 and "--logs" in r.stderr


def test_archive_logs_wrapper_archives_only_logs(work):
    r = run(LOGS, work)
    assert r.returncode == 0, r.stderr
    files = archived(work)
    assert "results/logs/a.json" in files
    assert not any(f.startswith(LLM) for f in files)


def test_llm_mode_archives_gitignored_trajectories(work):
    head, status = git(work, "rev-parse", "HEAD"), git(work, "status", "--porcelain")
    r = run(ARTIFACTS, work, "--llm")
    assert r.returncode == 0, r.stderr
    files = archived(work)
    assert f"{LLM}/op/model/high/iter_0/prompt.md" in files
    assert not any(f.endswith(".pyc") or "__pycache__" in f for f in files)
    assert "results/logs/a.json" not in files             # --llm leaves logs alone
    assert "src.txt" in files                              # main's tree is the base
    # nothing was checked out and the real index is untouched
    assert git(work, "rev-parse", "HEAD") == head
    assert git(work, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert git(work, "status", "--porcelain") == status
    assert (work / LLM / "op/model/high/iter_0/prompt.md").exists()


def test_all_mode_archives_both(work):
    assert run(ARTIFACTS, work, "--all").returncode == 0
    files = archived(work)
    assert "results/logs/a.json" in files
    assert f"{LLM}/op/model/high/iter_0/prompt.md" in files


def test_archive_is_cumulative_and_keeps_the_legacy_path(work):
    assert run(ARTIFACTS, work, "--llm").returncode == 0
    # this machine loses one trajectory and gains another
    shutil.rmtree(work / LLM / "op")
    write(work / LLM / "op2/model/high/iter_0/prompt.md", "second\n")
    assert run(ARTIFACTS, work, "--llm").returncode == 0
    assert run(LOGS, work).returncode == 0                 # a logs-only run drops nothing either
    files = archived(work)
    assert f"{LLM}/op/model/high/iter_0/prompt.md" in files
    assert f"{LLM}/op2/model/high/iter_0/prompt.md" in files
    assert "results/logs/a.json" in files
    assert "benchmarks/llm_generated/legacy.txt" in files  # main dropped it; the archive keeps it


def test_working_tree_wins_and_history_is_only_appended(work):
    assert run(ARTIFACTS, work, "--llm").returncode == 0
    first = git(work, "rev-parse", BRANCH)
    write(work / LLM / "op/model/high/iter_0/prompt.md", "revised\n")
    assert run(ARTIFACTS, work, "--llm").returncode == 0
    second = git(work, "rev-parse", BRANCH)
    assert git(work, "show", f"{BRANCH}:{LLM}/op/model/high/iter_0/prompt.md") == "revised"
    git(work, "merge-base", "--is-ancestor", first, second)   # raises unless first is an ancestor


def test_rerun_without_changes_is_a_noop(work):
    assert run(ARTIFACTS, work, "--llm").returncode == 0
    tip = git(work, "rev-parse", BRANCH)
    r = run(ARTIFACTS, work, "--llm")
    assert r.returncode == 0 and "already up to date" in r.stdout
    assert git(work, "rev-parse", BRANCH) == tip


def test_no_push_unless_asked(work):
    remote_before = git(work, "ls-remote", "origin", f"refs/heads/{BRANCH}")
    assert run(ARTIFACTS, work, "--llm").returncode == 0
    assert git(work, "ls-remote", "origin", f"refs/heads/{BRANCH}") == remote_before
    assert run(ARTIFACTS, work, "--llm", "--push").returncode == 0
    assert git(work, "ls-remote", "origin", f"refs/heads/{BRANCH}").split()[0] == git(work, "rev-parse", BRANCH)
