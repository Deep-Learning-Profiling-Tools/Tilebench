"""scripts/archive_artifacts.sh, run against throwaway repositories (never the
real one). Like the script, this file lives on the archive branch only; run it
from a checkout of that branch:  pytest tests/test_archive_artifacts.py"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

TOOL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TOOL_ROOT / "scripts" / "archive_artifacts.sh"
BRANCH = "archive/tbpp-test"
FROZEN = "archive/raw-logs-2026-09-18"
LLM = "tilebench/benchmarks/llm_generated"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")

ENV = {**os.environ,
       "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
       "GIT_CONFIG_NOSYSTEM": "1"}


def git(cwd, *args, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, env=ENV, capture_output=True, text=True)
    if check and r.returncode:
        raise AssertionError(f"git {' '.join(args)}: {r.stderr}")
    return r.stdout.strip()


def run(cwd, *args, script=SCRIPT):
    return subprocess.run(["bash", str(script), *args], cwd=cwd, env=ENV,
                          capture_output=True, text=True)


def ref(cwd, name):
    return git(cwd, "rev-parse", "-q", "--verify", name, check=False)


def remote_ref(cwd, name=BRANCH):
    out = git(cwd, "ls-remote", "origin", f"refs/heads/{name}")
    return out.split()[0] if out else ""


def tree(cwd, rev=BRANCH):
    return set(git(cwd, "ls-tree", "-r", "--name-only", rev).splitlines())


def show(cwd, path, rev=BRANCH):
    return git(cwd, "show", f"{rev}:{path}")


def trailers(cwd, rev=BRANCH):
    body = git(cwd, "log", "-1", "--format=%B", rev)
    return dict(line.split(": ", 1) for line in body.splitlines() if ": " in line)


def write(path, text="x\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def work(tmp_path):
    """A source repository with an origin, a frozen legacy archive branch, and
    git-ignored artifacts of GH200 on disk."""
    origin, work = tmp_path / "origin.git", tmp_path / "work"
    git(tmp_path, "init", "-q", "--bare", str(origin))
    git(tmp_path, "init", "-q", str(work))
    git(work, "checkout", "-q", "-b", "main")
    git(work, "remote", "add", "origin", str(origin))
    write(work / "src.py", "v1\n")
    write(work / ".gitignore", f"results/*/logs/\noutputs/\n{LLM}/\n")
    write(work / "results/GH200/csv/op_default.csv", "params,dtype\n")     # tracked summary CSV
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "source v1")
    write(work / "src.py", "v2\n")
    git(work, "commit", "-q", "-am", "source v2")
    git(work, "push", "-q", "origin", "main")
    # the frozen paper archive, which no run may touch
    git(work, "push", "-q", "origin", f"main~1:refs/heads/{FROZEN}")
    git(work, "fetch", "-q", "origin")
    git(work, "branch", "-q", FROZEN, f"origin/{FROZEN}")

    write(work / "results/GH200/logs/time_measurement_logs/op_default_triton.json", "[1]\n")
    write(work / "results/GH200/logs/provenance/op_default_triton.json", "{}\n")
    write(work / "outputs/profiling/GH200/kernel_counts.json", "[]\n")
    write(work / "outputs/ncu/GH200/op/triton_fp16.ncu-rep", "big report")
    write(work / LLM / "op/model/high/iter_0/prompt.md", "prompt\n")
    write(work / LLM / "op/model/high/iter_0/__pycache__/impl.cpython-310.pyc", "bytecode")
    return work


def init(work, *extra, base="main"):
    r = run(work, "--branch", BRANCH, "--base", base, "--init", *extra)
    assert r.returncode == 0, r.stderr
    return r


def frozen_state(work):
    return ref(work, FROZEN), remote_ref(work, FROZEN)


# --------------------------------------------------------------------------
# 1. --init
# --------------------------------------------------------------------------

def test_init_creates_the_branch_from_an_explicit_base_with_the_tool_only(work):
    assert ref(work, BRANCH) == "" and remote_ref(work) == ""
    init(work)
    assert tree(work) == {"scripts/archive_artifacts.sh", "tests/test_archive_artifacts.py"}
    assert show(work, "scripts/archive_artifacts.sh") == SCRIPT.read_text().strip()
    assert git(work, "log", "-1", "--format=%P", BRANCH) == git(work, "rev-parse", "main")
    assert git(work, "log", "-1", "--format=%s", BRANCH) == \
        f"archive: init {BRANCH} from source {git(work, 'rev-parse', '--short', 'main')}"
    t = trailers(work)
    assert t["Archive-Branch"] == BRANCH and t["Source-Base"] == git(work, "rev-parse", "main")
    assert t["Artifact-Classes"] == "none" and "Hardware" not in t
    assert remote_ref(work) == ""                                   # not pushed unless asked
    assert "100755" in git(work, "ls-tree", BRANCH, "scripts/archive_artifacts.sh") or \
        not os.access(SCRIPT, os.X_OK)


def test_init_can_push_and_can_take_artifacts(work):
    init(work, "--logs", "--gpu", "GH200", "--push")
    assert remote_ref(work) == ref(work, BRANCH)
    assert "results/GH200/logs/time_measurement_logs/op_default_triton.json" in tree(work)
    assert git(work, "log", "-1", "--format=%s", BRANCH).startswith(f"archive: init {BRANCH} with GH200 logs")


@pytest.mark.parametrize("where", ["local", "remote"])
def test_init_refuses_an_existing_branch(work, where):
    init(work, "--push" if where == "remote" else "--logs", *([] if where == "remote" else ["--gpu", "GH200"]))
    if where == "remote":
        git(work, "branch", "-q", "-D", BRANCH)
    before = ref(work, BRANCH), remote_ref(work)
    r = run(work, "--branch", BRANCH, "--base", "main", "--init")
    assert r.returncode == 1 and "already exists" in r.stderr
    assert (ref(work, BRANCH), remote_ref(work)) == before


def test_appending_to_a_missing_branch_asks_for_init(work):
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "GH200")
    assert r.returncode == 1 and "--init" in r.stderr and ref(work, BRANCH) == ""


def test_init_needs_the_tool_files_beside_the_script(work, tmp_path):
    lone = tmp_path / "elsewhere" / "aa.sh"                 # e.g. git show ... > /tmp/aa.sh
    lone.parent.mkdir()
    shutil.copy(SCRIPT, lone)
    r = run(work, "--branch", BRANCH, "--base", "main", "--init", script=lone)
    assert r.returncode == 1 and "tests/test_archive_artifacts.py" in r.stderr
    assert ref(work, BRANCH) == ""


# --------------------------------------------------------------------------
# 2-4. appending, partial artifacts, other GPUs
# --------------------------------------------------------------------------

def test_append_extends_history_and_records_hardware_and_classes(work):
    init(work)
    first = ref(work, BRANCH)
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--profiling", "--gpu", "GH200")
    assert r.returncode == 0, r.stderr
    git(work, "merge-base", "--is-ancestor", first, BRANCH)       # appended, not rewritten
    assert git(work, "log", "-1", "--format=%P", BRANCH) == first  # base already contained: one parent
    assert git(work, "log", "-1", "--format=%s", BRANCH) == \
        f"archive: GH200 logs+profiling from source {git(work, 'rev-parse', '--short', 'main')}"
    t = trailers(work)
    assert t["Hardware"] == "GH200" and t["Artifact-Classes"] == "logs profiling"
    assert t["Files-Added"] == "3" and t["Files-Updated"] == "0"
    files = tree(work)
    assert {"results/GH200/logs/time_measurement_logs/op_default_triton.json",
            "results/GH200/logs/provenance/op_default_triton.json",
            "outputs/profiling/GH200/kernel_counts.json",
            "scripts/archive_artifacts.sh"} <= files


def test_a_machine_with_partial_artifacts_keeps_every_other_gpu(work, tmp_path):
    """The archive already holds B200 and MI300X artifacts; this GH200 machine
    has none of them on disk. Its run adds GH200 and drops nothing."""
    init(work, "--push")
    # other machines archived before, through origin
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(other))
    for gpu in ("B200", "MI300X"):
        write(other / f"results/{gpu}/logs/time_measurement_logs/op.json", f"{gpu}\n")
        write(other / f"outputs/profiling/{gpu}/kernel_counts.json", f"{gpu}\n")
        r = run(other, "--branch", BRANCH, "--base", "origin/main", "--logs", "--profiling",
                "--gpu", gpu, "--push")
        assert r.returncode == 0, r.stderr
    write(other / LLM / "old/model/high/iter_0/prompt.md", "old\n")
    assert run(other, "--branch", BRANCH, "--base", "origin/main", "--llm", "--push").returncode == 0

    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "GH200", "--push")
    assert r.returncode == 0, r.stderr
    files = tree(work)
    for gpu in ("B200", "MI300X"):
        assert f"results/{gpu}/logs/time_measurement_logs/op.json" in files
        assert f"outputs/profiling/{gpu}/kernel_counts.json" in files
        assert show(work, f"results/{gpu}/logs/time_measurement_logs/op.json") == gpu
    assert f"{LLM}/old/model/high/iter_0/prompt.md" in files
    assert "results/GH200/logs/time_measurement_logs/op_default_triton.json" in files
    assert remote_ref(work) == ref(work, BRANCH)


def test_a_file_removed_from_this_machine_stays_archived_and_a_changed_one_is_updated(work):
    init(work, "--logs", "--gpu", "GH200")
    (work / "results/GH200/logs/provenance/op_default_triton.json").unlink()
    write(work / "results/GH200/logs/time_measurement_logs/op_default_triton.json", "[2]\n")
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "GH200")
    assert r.returncode == 0, r.stderr
    assert "results/GH200/logs/provenance/op_default_triton.json" in tree(work)
    assert show(work, "results/GH200/logs/time_measurement_logs/op_default_triton.json") == "[2]"
    assert trailers(work)["Files-Updated"] == "1"


def test_rerun_without_changes_is_a_noop(work):
    init(work, "--logs", "--gpu", "GH200")
    tip = ref(work, BRANCH)
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "GH200")
    assert r.returncode == 0 and "already up to date" in r.stdout and ref(work, BRANCH) == tip


# --------------------------------------------------------------------------
# 5-6. --base
# --------------------------------------------------------------------------

def test_an_arbitrary_exact_sha_is_the_recorded_base(work):
    old = git(work, "rev-parse", "main~1")                  # not the tip of any branch
    init(work, base=old)
    assert git(work, "log", "-1", "--format=%P", BRANCH) == old
    assert trailers(work)["Source-Base"] == old and trailers(work)["Source-Ref"] == old
    assert git(work, "log", "-1", "--format=%s", BRANCH).endswith(f"from source {old[:7]}")


def test_a_ref_base_is_resolved_to_its_exact_sha_and_kept_reachable(work):
    git(work, "checkout", "-q", "-b", "feature/multiarch")
    write(work / "src.py", "v3\n")
    git(work, "commit", "-q", "-am", "feature work, never merged")
    feature = git(work, "rev-parse", "HEAD")
    init(work)                                               # base = main
    r = run(work, "--branch", BRANCH, "--base", "feature/multiarch", "--logs", "--gpu", "GH200", "--push")
    assert r.returncode == 0, r.stderr
    t = trailers(work)
    assert t["Source-Base"] == feature and t["Source-Ref"] == "feature/multiarch"
    # the feature commit is a parent: it survives the feature branch being deleted
    assert feature in git(work, "log", "-1", "--format=%P", BRANCH).split()
    git(work, "checkout", "-q", "main")
    git(work, "branch", "-q", "-D", "feature/multiarch")
    git(work, "merge-base", "--is-ancestor", feature, f"origin/{BRANCH}")


def test_base_is_required_and_must_be_a_commit(work):
    r = run(work, "--branch", BRANCH, "--init")
    assert r.returncode == 2 and "--base" in r.stderr
    r = run(work, "--branch", BRANCH, "--base", "no-such-ref", "--init")
    assert r.returncode == 1 and "no-such-ref" in r.stderr and ref(work, BRANCH) == ""


# --------------------------------------------------------------------------
# 7-9. the caller's checkout and the frozen archive are never touched
# --------------------------------------------------------------------------

def test_real_index_working_tree_and_head_are_unchanged(work):
    write(work / "src.py", "local edit\n")                  # an unstaged edit
    write(work / "staged.py", "s\n")
    git(work, "add", "staged.py")                            # a staged file
    index = (work / ".git/index").read_bytes()
    status, head = git(work, "status", "--porcelain"), git(work, "rev-parse", "HEAD")
    init(work)
    assert run(work, "--branch", BRANCH, "--base", "main", "--all", "--gpu", "GH200").returncode == 0
    assert (work / ".git/index").read_bytes() == index
    assert git(work, "status", "--porcelain") == status
    assert git(work, "rev-parse", "HEAD") == head and git(work, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert (work / "src.py").read_text() == "local edit\n"
    assert (work / "results/GH200/logs/time_measurement_logs/op_default_triton.json").exists()


def test_the_frozen_archive_is_never_modified(work):
    before = frozen_state(work)
    init(work, "--push")
    assert run(work, "--branch", BRANCH, "--base", "main", "--all", "--gpu", "GH200", "--push").returncode == 0
    assert frozen_state(work) == before
    r = run(work, "--branch", FROZEN, "--base", "main", "--logs", "--gpu", "GH200", "--push")
    assert r.returncode == 2 and "frozen" in r.stderr
    assert frozen_state(work) == before


def test_a_checked_out_archive_branch_is_refused(work, tmp_path):
    init(work)
    git(work, "worktree", "add", "-q", str(tmp_path / "wt"), BRANCH)
    tip = ref(work, BRANCH)
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "GH200")
    assert r.returncode == 1 and "checked out" in r.stderr and ref(work, BRANCH) == tip


# --------------------------------------------------------------------------
# 10. branch names and labels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["", "main", "feature/tilebenchpp-multiarch", "archive/", "archive",
                                  "archive/../main", "archive/a/b", "archive/a b", "archive/x.lock",
                                  "archive/.hidden", "archive/-x", "-archive/x", "archive/a..b",
                                  "refs/heads/archive/x", "archive/x\n"])
def test_invalid_or_dangerous_branch_names_fail_safely(work, name):
    refs_before = git(work, "for-each-ref")
    r = run(work, "--branch", name, "--base", "main", "--init")
    assert r.returncode == 2 and "branch" in r.stderr
    assert git(work, "for-each-ref") == refs_before


@pytest.mark.parametrize("gpu", [(), ("--gpu",), ("--gpu", "../B200"), ("--gpu", "a/b"), ("--gpu", "")])
def test_hardware_classes_need_a_safe_label(work, gpu):
    init(work)
    tip = ref(work, BRANCH)
    for cls in ("--logs", "--profiling", "--all"):
        r = run(work, "--branch", BRANCH, "--base", "main", cls, *gpu)
        assert r.returncode == 2 and "--gpu" in r.stderr
    assert ref(work, BRANCH) == tip


def test_gpu_without_a_hardware_class_is_rejected(work):
    init(work)
    r = run(work, "--branch", BRANCH, "--base", "main", "--llm", "--gpu", "GH200")
    assert r.returncode == 2


def test_a_class_is_required(work):
    init(work)
    r = run(work, "--branch", BRANCH, "--base", "main")
    assert r.returncode == 2 and "--logs" in r.stderr


def test_a_missing_directory_is_an_error_not_an_empty_snapshot(work):
    init(work)
    tip = ref(work, BRANCH)
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "MI300X")
    assert r.returncode == 1 and "results/MI300X/logs" in r.stderr and ref(work, BRANCH) == tip


# --------------------------------------------------------------------------
# 11-13. what is and is not archived
# --------------------------------------------------------------------------

def test_summary_csvs_are_never_archived(work):
    write(work / "results/GH200/csv/untracked_extra.csv", "a\n")
    init(work, "--all", "--gpu", "GH200")
    assert not any("/csv/" in f or f.endswith(".csv") for f in tree(work))
    assert "src.py" not in tree(work)                        # nor any source file


def test_ncu_reports_and_caches_are_never_archived(work):
    write(work / "outputs/profiling/GH200/stray.ncu-rep", "report")
    init(work, "--all", "--gpu", "GH200")
    files = tree(work)
    assert not any(f.endswith(".ncu-rep") or f.startswith("outputs/ncu/") for f in files)
    assert not any(f.endswith(".pyc") or "__pycache__" in f for f in files)
    assert "outputs/profiling/GH200/kernel_counts.json" in files


def test_an_oversized_file_stops_the_run(work):
    big = work / "results/GH200/logs/huge.json"
    with open(big, "wb") as f:
        f.truncate(51 * 1024 * 1024)                         # sparse: no real disk use
    r = run(work, "--branch", BRANCH, "--base", "main", "--init", "--logs", "--gpu", "GH200")
    assert r.returncode == 1 and "results/GH200/logs/huge.json" in r.stderr
    assert ref(work, BRANCH) == ""


def test_all_means_logs_profiling_and_llm_of_one_gpu(work):
    write(work / "results/B200/logs/other_gpu.json", "{}\n")
    write(work / "outputs/profiling/B200/other_gpu.json", "{}\n")
    init(work)
    r = run(work, "--branch", BRANCH, "--base", "main", "--all", "--gpu", "GH200")
    assert r.returncode == 0, r.stderr
    files = tree(work)
    assert "results/GH200/logs/time_measurement_logs/op_default_triton.json" in files
    assert "outputs/profiling/GH200/kernel_counts.json" in files
    assert f"{LLM}/op/model/high/iter_0/prompt.md" in files
    assert not any("B200" in f and "GH200" not in f for f in files)   # only the named GPU
    assert trailers(work)["Artifact-Classes"] == "logs profiling llm"
    assert git(work, "log", "-1", "--format=%s", BRANCH).startswith("archive: GH200 logs+profiling+llm")


def test_llm_only(work):
    init(work)
    assert run(work, "--branch", BRANCH, "--base", "main", "--llm").returncode == 0
    files = tree(work)
    assert f"{LLM}/op/model/high/iter_0/prompt.md" in files
    assert not any(f.startswith(("results/", "outputs/")) for f in files)
    assert "Hardware" not in trailers(work)


def test_usage_documents_every_option():
    r = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert r.returncode == 2
    for opt in ("--branch", "--base", "--init", "--logs", "--profiling", "--llm", "--all", "--gpu", "--push"):
        assert opt in r.stderr


# --------------------------------------------------------------------------
# local / remote refs
# --------------------------------------------------------------------------

def test_only_remote_branch(work):
    init(work, "--push")
    remote = remote_ref(work)
    git(work, "branch", "-q", "-D", BRANCH)
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "GH200")
    assert r.returncode == 0, r.stderr
    assert git(work, "log", "-1", "--format=%P", BRANCH) == remote


def test_only_local_branch_is_pushed_on_request(work):
    init(work)
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "GH200", "--push")
    assert r.returncode == 0, r.stderr
    assert remote_ref(work) == ref(work, BRANCH)


def test_stale_local_branch_builds_on_origin(work, tmp_path):
    init(work, "--push")
    stale = ref(work, BRANCH)
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(other))
    write(other / "results/B200/logs/b.json", "{}\n")
    assert run(other, "--branch", BRANCH, "--base", "origin/main", "--logs", "--gpu", "B200",
               "--push").returncode == 0
    newer = remote_ref(work)
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "GH200", "--push")
    assert r.returncode == 0, r.stderr
    assert git(work, "log", "-1", "--format=%P", BRANCH) == newer
    git(work, "merge-base", "--is-ancestor", stale, BRANCH)
    assert "results/B200/logs/b.json" in tree(work)


def test_local_ahead_of_origin_builds_on_local(work):
    init(work, "--push")
    assert run(work, "--branch", BRANCH, "--base", "main", "--llm").returncode == 0   # not pushed
    ahead = ref(work, BRANCH)
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "GH200")
    assert r.returncode == 0, r.stderr
    assert git(work, "log", "-1", "--format=%P", BRANCH) == ahead


def test_diverged_branches_fail_without_changing_anything(work, tmp_path):
    init(work, "--push")
    assert run(work, "--branch", BRANCH, "--base", "main", "--llm").returncode == 0   # local-only commit
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(tmp_path / "origin.git"), str(other))
    write(other / "results/B200/logs/b.json", "{}\n")
    assert run(other, "--branch", BRANCH, "--base", "origin/main", "--logs", "--gpu", "B200",
               "--push").returncode == 0                     # origin moved elsewhere
    before = ref(work, BRANCH), remote_ref(work)
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "GH200", "--push")
    assert r.returncode == 1 and "diverged" in r.stderr
    assert (ref(work, BRANCH), remote_ref(work)) == before


def test_a_rejected_push_restores_the_local_branch(work, tmp_path):
    init(work, "--push")
    tip = ref(work, BRANCH)
    hook = tmp_path / "origin.git" / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho rejected >&2\nexit 1\n")
    hook.chmod(0o755)
    r = run(work, "--branch", BRANCH, "--base", "main", "--logs", "--gpu", "GH200", "--push")
    assert r.returncode == 1 and "rejected" in r.stderr
    assert ref(work, BRANCH) == tip and remote_ref(work) == tip
