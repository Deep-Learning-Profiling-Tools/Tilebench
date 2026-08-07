#!/usr/bin/env python3
"""Sync per-operator TileLang branches from an all-implemented worktree.

Default behavior is a dry run. Use --apply to create/update local branches.
Use --push with --apply to push those branch updates.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


BRANCH_PREFIX = "aaroosh/feature/tilelang-"
DEFAULT_BASE = "origin/main"


def run_git(repo: Path, args: list[str], *, cwd: Path | None = None, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-C", str(cwd or repo), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def ref_exists(repo: Path, ref: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def show_file(repo: Path, ref: str, path: str) -> bytes | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def rev_parse(repo: Path, ref: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", ref],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def operator_to_branch(operator: str) -> str:
    return f"{BRANCH_PREFIX}{operator.replace('_', '-')}"


def branch_to_ref(repo: Path, branch: str) -> tuple[str | None, str, bool]:
    local_exists = ref_exists(repo, f"refs/heads/{branch}")
    remote = f"origin/{branch}"
    remote_exists = ref_exists(repo, f"refs/remotes/{remote}")
    if local_exists:
        return branch, "local", remote_exists
    if remote_exists:
        return remote, "remote", remote_exists
    return None, "new", False


def local_remote_mismatch(repo: Path, branch: str) -> tuple[str, str] | None:
    if not ref_exists(repo, f"refs/heads/{branch}"):
        return None
    remote = f"origin/{branch}"
    if not ref_exists(repo, f"refs/remotes/{remote}"):
        return None
    local_sha = rev_parse(repo, branch)
    remote_sha = rev_parse(repo, remote)
    if local_sha and remote_sha and local_sha != remote_sha:
        return local_sha[:7], remote_sha[:7]
    return None


def branch_update_ref(repo: Path, branch: str) -> str:
    if ref_exists(repo, f"refs/heads/{branch}"):
        return branch, "local"
    remote = f"origin/{branch}"
    if ref_exists(repo, f"refs/remotes/{remote}"):
        return remote, "remote"
    return None, "new"


def discover_tilelang_files(repo: Path) -> dict[str, Path]:
    root = repo / "benchmarks" / "operators"
    files: dict[str, Path] = {}
    for path in sorted(root.glob("*/impl_tilelang.py")):
        files[path.parent.name] = path
    return files


def read_source_file(repo: Path, source_ref: str | None, path: Path) -> bytes:
    rel = path.relative_to(repo).as_posix()
    if source_ref:
        data = show_file(repo, source_ref, rel)
        if data is None:
            raise FileNotFoundError(f"{rel} not found in {source_ref}")
        return data
    return path.read_bytes()


def build_commit(
    repo: Path,
    base_ref: str,
    branch: str,
    operator: str,
    rel_path: str,
    content: bytes,
    *,
    apply: bool,
    push: bool,
    allow_local_remote_mismatch: bool,
    message: str | None,
) -> str:
    mismatch = local_remote_mismatch(repo, branch)
    if mismatch and not allow_local_remote_mismatch:
        local_sha, remote_sha = mismatch
        return (
            f"blocked\t{operator}\t{branch}\tlocal {local_sha} != "
            f"origin {remote_sha}; fetch/merge or pass --allow-local-remote-mismatch"
        )

    start_ref, source_kind, _ = branch_to_ref(repo, branch)
    start = start_ref or base_ref
    target_data = show_file(repo, start, rel_path) if start_ref else None

    if target_data == content:
        return f"skip\t{operator}\t{branch}\talready matches {source_kind} ref"

    if not apply:
        action = "update" if start_ref else "create"
        return f"dry-run\t{operator}\t{branch}\t{action} from {start}"

    tmp_root = Path(tempfile.mkdtemp(prefix=f"tilelang-sync-{operator}-"))
    try:
        run_git(repo, ["worktree", "add", "--detach", str(tmp_root), start])
        dst = tmp_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)
        run_git(repo, ["add", rel_path], cwd=tmp_root)

        diff_proc = subprocess.run(
            ["git", "-C", str(tmp_root), "diff", "--cached", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if diff_proc.returncode == 0:
            return f"skip\t{operator}\t{branch}\tno staged change"

        commit_message = message
        if commit_message is None:
            commit_message = f"operator({operator}): add TileLang implementation"
            if start_ref:
                commit_message = f"operator({operator}): update TileLang implementation"
        run_git(repo, ["commit", "-m", commit_message], cwd=tmp_root)
        new_commit = run_git(repo, ["rev-parse", "--short", "HEAD"], cwd=tmp_root)
        run_git(repo, ["branch", "-f", branch, "HEAD"], cwd=tmp_root)

        pushed = ""
        if push:
            run_git(repo, ["push", "-u", "origin", branch], cwd=tmp_root)
            pushed = "\tpushed"

        action = "updated" if start_ref else "created"
        return f"{action}\t{operator}\t{branch}\t{new_commit}{pushed}"
    finally:
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(tmp_root)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(tmp_root, ignore_errors=True)


def parse_operators(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {part.strip().replace("-", "_") for part in raw.split(",") if part.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create/update aaroosh/feature/tilelang-* branches from impl_tilelang.py files.",
    )
    parser.add_argument(
        "--operators",
        help="Comma-separated operators to sync, e.g. relu,l2_norm. Default: all changed targets.",
    )
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"Base ref for new branches. Default: {DEFAULT_BASE}.",
    )
    parser.add_argument(
        "--source-ref",
        default=None,
        help="Read impl_tilelang.py files from this ref instead of the current worktree.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create/update local branches. Without this, only prints what would change.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push updated branches. Requires --apply. Does not force-push.",
    )
    parser.add_argument(
        "--message",
        help="Commit message to use for every created/updated branch in this run.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Run git fetch origin before comparing local branches to origin branches.",
    )
    parser.add_argument(
        "--allow-local-remote-mismatch",
        action="store_true",
        help="Allow updating a local branch even when it differs from origin/<branch>.",
    )
    args = parser.parse_args()

    if args.push and not args.apply:
        parser.error("--push requires --apply")
    if args.message is not None and not args.message.strip():
        parser.error("--message cannot be empty")

    repo = Path(run_git(Path.cwd(), ["rev-parse", "--show-toplevel"])).resolve()
    if args.fetch:
        run_git(repo, ["fetch", "origin"])

    files = discover_tilelang_files(repo)
    selected = parse_operators(args.operators)
    if selected is not None:
        missing = sorted(selected - set(files))
        if missing:
            raise SystemExit(f"No impl_tilelang.py found for: {', '.join(missing)}")
        files = {op: path for op, path in files.items() if op in selected}

    if not ref_exists(repo, args.base):
        raise SystemExit(f"Base ref not found: {args.base}")

    changed = 0
    for operator, path in files.items():
        rel_path = path.relative_to(repo).as_posix()
        branch = operator_to_branch(operator)
        content = read_source_file(repo, args.source_ref, path)
        result = build_commit(
            repo,
            args.base,
            branch,
            operator,
            rel_path,
            content,
            apply=args.apply,
            push=args.push,
            allow_local_remote_mismatch=args.allow_local_remote_mismatch,
            message=args.message.strip() if args.message is not None else None,
        )
        print(result)
        if not result.startswith("skip"):
            changed += 1

    print(f"\noperators considered: {len(files)}")
    print(f"branches needing action: {changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
