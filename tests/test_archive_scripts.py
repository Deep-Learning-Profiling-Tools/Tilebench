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


LEGACY_LOG = "time_measurement_logs/legacy_results.json"     # relative path, kept by the migration


@pytest.fixture
def work(tmp_path):
    """The situation the real repository is in. The archive branch holds two
    legacy snapshots, raw logs under results/logs/ and the pre-package LLM
    directory, plus the logs of another GPU. main has dropped the LLM snapshot
    but STILL tracks the legacy results/logs/. Git-ignored artifacts of two GPUs
    are on disk."""
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(tmp_path, "init", "-q", str(work))
    git(work, "checkout", "-q", "-b", "main")
    git(work, "remote", "add", "origin", str(origin))
    write(work / "src.txt")
    write(work / ".gitignore", f"results/*\n{LLM}/\n__pycache__/\n*.pyc\n")
    write(work / "benchmarks/llm_generated/legacy.txt", "legacy\n")
    write(work / f"results/logs/{LEGACY_LOG}", '{"paper": "B200"}\n')
    write(work / "results/GH200/logs/archived_g.json", '{"gpu": "GH200"}\n')
    git(work, "add", "-A")
    git(work, "add", "-f", "results/logs", "results/GH200")
    git(work, "commit", "-q", "-m", "main with the legacy snapshots")
    git(work, "push", "-q", "origin", "main")
    git(work, "push", "-q", "origin", f"main:refs/heads/{BRANCH}")   # archive starts here
    git(work, "rm", "-q", "-r", "benchmarks/llm_generated", "results/GH200")
    git(work, "commit", "-q", "-m", "main drops the LLM snapshot, still tracks results/logs/")
    git(work, "push", "-q", "origin", "main")
    git(work, "fetch", "-q", "origin")
    assert f"results/logs/{LEGACY_LOG}" in git(work, "ls-tree", "-r", "--name-only", "origin/main")

    write(work / "results/B200/logs/a.json", "{}\n")
    write(work / "results/B200/logs/nki_profiles/mul2/case0/manifest.json", "{}\n")   # NKI, same campaign
    write(work / "results/B200/logs/nki_neff_manifest.jsonl", "{}\n")
    write(work / "results/GH200/logs/g.json", "{}\n")
    write(work / "results/B200/figures/plot.png", "png")
    write(work / LLM / "op/model/high/iter_0/prompt.md", "prompt\n")
    write(work / LLM / "op/model/high/iter_0/__pycache__/impl.cpython-310.pyc", "bytecode")
    return work


def blob(work, rev, path):
    return git(work, "rev-parse", f"{rev}:{path}")


def test_mode_is_required(work):
    r = run(ARTIFACTS, work)
    assert r.returncode == 2 and "--logs" in r.stderr


@pytest.mark.parametrize("args", [(), ("--gpu",), ("--gpu", "../B200"), ("--gpu", "a/b"), ("--gpu", "")])
def test_logs_need_a_safe_gpu_label(work, args):
    tip = git(work, "rev-parse", f"origin/{BRANCH}")
    r = run(LOGS, work, *args)
    assert r.returncode == 2 and "--gpu" in r.stderr
    assert git(work, "rev-parse", f"origin/{BRANCH}") == tip
    assert subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/heads/{BRANCH}"],
                          cwd=work, env=ENV, capture_output=True).returncode != 0   # nothing was committed


def test_archive_logs_wrapper_archives_only_the_logs_of_that_gpu(work):
    r = run(LOGS, work, "--gpu", "B200")
    assert r.returncode == 0, r.stderr
    files = archived(work)
    assert "results/B200/logs/a.json" in files
    # the whole logs/ tree of the namespace goes in, NKI profiles included: no separate NKI path
    assert "results/B200/logs/nki_profiles/mul2/case0/manifest.json" in files
    assert "results/B200/logs/nki_neff_manifest.jsonl" in files
    assert "results/GH200/logs/g.json" not in files        # another GPU's logs are not swept in
    assert "results/B200/figures/plot.png" not in files    # only logs/ is archived
    assert not any(f.startswith(LLM) for f in files)


def test_llm_mode_archives_gitignored_trajectories(work):
    head, status = git(work, "rev-parse", "HEAD"), git(work, "status", "--porcelain")
    r = run(ARTIFACTS, work, "--llm")
    assert r.returncode == 0, r.stderr
    files = archived(work)
    assert f"{LLM}/op/model/high/iter_0/prompt.md" in files
    assert not any(f.endswith(".pyc") or "__pycache__" in f for f in files)
    assert "results/B200/logs/a.json" not in files        # --llm leaves logs alone
    assert "src.txt" in files                              # main's tree is the base
    # nothing was checked out and the real index is untouched
    assert git(work, "rev-parse", "HEAD") == head
    assert git(work, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert git(work, "status", "--porcelain") == status
    assert (work / LLM / "op/model/high/iter_0/prompt.md").exists()


def test_all_mode_archives_both(work):
    assert run(ARTIFACTS, work, "--all", "--gpu=B200").returncode == 0
    files = archived(work)
    assert "results/B200/logs/a.json" in files
    assert f"{LLM}/op/model/high/iter_0/prompt.md" in files


def test_archive_is_cumulative_and_keeps_the_legacy_path(work):
    assert run(ARTIFACTS, work, "--llm").returncode == 0
    # this machine loses one trajectory and gains another
    shutil.rmtree(work / LLM / "op")
    write(work / LLM / "op2/model/high/iter_0/prompt.md", "second\n")
    assert run(ARTIFACTS, work, "--llm").returncode == 0
    assert run(LOGS, work, "--gpu", "B200").returncode == 0   # a logs-only run drops nothing either
    # the B200 logs leave this machine; a GH200 run must not drop them from the archive
    shutil.rmtree(work / "results/B200")
    assert run(LOGS, work, "--gpu", "GH200").returncode == 0
    files = archived(work)
    assert f"{LLM}/op/model/high/iter_0/prompt.md" in files
    assert f"{LLM}/op2/model/high/iter_0/prompt.md" in files
    assert "results/B200/logs/a.json" in files
    assert "results/B200/logs/nki_profiles/mul2/case0/manifest.json" in files
    assert "results/GH200/logs/g.json" in files
    assert "results/GH200/logs/archived_g.json" in files   # archived earlier, never on this machine
    # the legacy LLM snapshot: main dropped it and this machine never had it; the archive keeps it
    assert "benchmarks/llm_generated/legacy.txt" in files
    assert f"results/B200/logs/{LEGACY_LOG}" in files       # the legacy raw logs live on, migrated


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



# --------------------------------------------------------------------------
# legacy results/logs/ -> results/B200/logs/
# --------------------------------------------------------------------------

def test_legacy_logs_are_migrated_byte_for_byte_and_never_come_back(work):
    before = blob(work, f"origin/{BRANCH}", f"results/logs/{LEGACY_LOG}")
    gh200 = blob(work, f"origin/{BRANCH}", "results/GH200/logs/archived_g.json")

    r = run(ARTIFACTS, work, "--llm")                      # any mode canonicalises the archive
    assert r.returncode == 0, r.stderr
    files = archived(work)
    assert blob(work, BRANCH, f"results/B200/logs/{LEGACY_LOG}") == before     # same bytes, same relative path
    # gone from the new tip, although origin/main still tracks it and is the base tree
    assert not any(f.startswith("results/logs/") for f in files)
    assert f"results/logs/{LEGACY_LOG}" in git(work, "ls-tree", "-r", "--name-only", "origin/main")
    assert blob(work, BRANCH, "results/GH200/logs/archived_g.json") == gh200   # another GPU is untouched
    assert "benchmarks/llm_generated/legacy.txt" in files                       # both LLM paths are kept
    assert f"{LLM}/op/model/high/iter_0/prompt.md" in files
    assert "migrate 1 legacy results/logs/ logs to results/B200/logs/" in git(work, "log", "-1", "--format=%s", BRANCH)

    tip = git(work, "rev-parse", BRANCH)                   # later runs have nothing left to migrate
    assert "already up to date" in run(ARTIFACTS, work, "--llm").stdout
    assert run(LOGS, work, "--gpu", "B200").returncode == 0
    assert not any(f.startswith("results/logs/") for f in archived(work))
    git(work, "merge-base", "--is-ancestor", tip, BRANCH)  # history is only appended to


def test_identical_file_at_the_destination_is_fine(work):
    write(work / f"results/B200/logs/{LEGACY_LOG}", '{"paper": "B200"}\n')      # same content as the legacy log
    r = run(LOGS, work, "--gpu", "B200")
    assert r.returncode == 0, r.stderr
    assert git(work, "show", f"{BRANCH}:results/B200/logs/{LEGACY_LOG}") == '{"paper": "B200"}'


def test_different_file_in_the_working_directory_aborts_the_migration(work):
    write(work / f"results/B200/logs/{LEGACY_LOG}", '{"stale": "local rerun"}\n')
    r = run(LOGS, work, "--gpu", "B200")
    assert r.returncode == 1
    assert f"results/B200/logs/{LEGACY_LOG}" in r.stderr and "DIFFERENT" in r.stderr
    assert subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/heads/{BRANCH}"],
                          cwd=work, env=ENV, capture_output=True).returncode != 0   # nothing was committed
    # the conflict is about B200 only: the same run for another GPU still works
    assert run(LOGS, work, "--gpu", "GH200").returncode == 0


def test_different_file_already_archived_aborts_the_migration(work, tmp_path):
    # an archive tip that holds the legacy log AND a different file at its destination
    env = {**ENV, "GIT_INDEX_FILE": str(tmp_path / "idx")}
    sh = lambda *a, **kw: subprocess.run(["git", *a], cwd=work, env=env, check=True,
                                         capture_output=True, text=True, **kw).stdout.strip()
    sh("read-tree", f"origin/{BRANCH}")
    other = sh("hash-object", "-w", "--stdin", input="other\n")
    sh("update-index", "--add", "--cacheinfo", f"100644,{other},results/B200/logs/{LEGACY_LOG}")
    commit = sh("commit-tree", sh("write-tree"), "-p", f"origin/{BRANCH}", "-m", "conflicting tip")
    git(work, "update-ref", f"refs/heads/{BRANCH}", commit)

    r = run(ARTIFACTS, work, "--llm")
    assert r.returncode == 1 and f"results/B200/logs/{LEGACY_LOG}" in r.stderr
    assert git(work, "rev-parse", BRANCH) == commit       # the branch did not move


def test_summary_csvs_come_from_main_not_from_the_archive_script(work):
    """Once main uses results/<gpu>/csv/, the archive tree shows the CSVs and the
    logs of that GPU side by side, and neither legacy directory."""
    write(work / "results/B200/csv/mul2_default.csv", "params,dtype\n")
    write(work / "results/B200/csv/notes.txt", "not tracked\n")
    git(work, "rm", "-q", "-r", "results/logs")
    git(work, "add", "-f", "results/B200/csv/mul2_default.csv")
    git(work, "commit", "-q", "-m", "main adopts results/B200/csv/ and stops tracking results/logs/")
    git(work, "push", "-q", "origin", "main")

    assert run(LOGS, work, "--gpu", "B200").returncode == 0
    files = archived(work)
    assert "results/B200/csv/mul2_default.csv" in files   # from main's tree
    assert "results/B200/csv/notes.txt" not in files      # the script copies no CSV directory
    assert "results/B200/logs/a.json" in files
    assert f"results/B200/logs/{LEGACY_LOG}" in files
    assert not any(f.startswith(("results/csv/", "results/logs/")) for f in files)


# --------------------------------------------------------------------------
# archive = public baseline + archive-only material
# --------------------------------------------------------------------------

def test_material_kept_only_on_the_archive_survives_every_run(work, tmp_path):
    """The archive can hold code merged in from a branch ahead of main, and files
    that left the public tree: cluster launch scripts and per-GPU NCU metadata."""
    env = {**ENV, "GIT_INDEX_FILE": str(tmp_path / "idx")}
    sh = lambda *a, **kw: subprocess.run(["git", *a], cwd=work, env=env, check=True,
                                         capture_output=True, text=True, **kw).stdout.strip()
    sh("read-tree", "origin/main")
    for line in git(work, "ls-tree", "-r", f"origin/{BRANCH}", "--", "benchmarks/llm_generated",
                    "results/GH200").splitlines():
        meta, path = line.split("\t")
        mode, _, sha = meta.split()
        sh("update-index", "--add", "--cacheinfo", f"{mode},{sha},{path}")
    extra = {"newer_source.py": "ahead of main\n",
             "tilebench/profiling/run_batch.sh": "#!/usr/bin/env bash\n",
             "tilebench/profiling/batch2.sbatch": "#SBATCH\n",
             "outputs/profiling/B200/kernel_counts.json": "[]"}
    for path, text in extra.items():
        sha = sh("hash-object", "-w", "--stdin", input=text)
        sh("update-index", "--add", "--cacheinfo", f"100644,{sha},{path}")
    merged = sh("commit-tree", sh("write-tree"), "-p", f"origin/{BRANCH}", "-p", "origin/main",
                "-m", "merge a newer baseline into the archive")
    git(work, "update-ref", f"refs/heads/{BRANCH}", merged)

    # main is already contained in the archive tip: the tip's own tree is the base
    assert run(LOGS, work, "--gpu", "B200").returncode == 0
    files = archived(work)
    assert set(extra) <= files and "results/B200/logs/a.json" in files
    for path, text in extra.items():
        assert git(work, "show", f"{BRANCH}:{path}") == text.strip()

    # main moves on: its tree becomes the base, and the archive-only files are still carried forward
    write(work / "main_moved.txt")
    git(work, "add", "main_moved.txt")
    git(work, "commit", "-q", "-m", "main advances")
    git(work, "push", "-q", "origin", "main")
    assert run(LOGS, work, "--gpu", "B200").returncode == 0
    files = archived(work)
    assert "main_moved.txt" in files
    assert {"tilebench/profiling/run_batch.sh", "tilebench/profiling/batch2.sbatch",
            "outputs/profiling/B200/kernel_counts.json"} <= files
    assert "results/B200/logs/a.json" in files and "benchmarks/llm_generated/legacy.txt" in files
