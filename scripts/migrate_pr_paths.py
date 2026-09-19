#!/usr/bin/env python3
"""Move a pre-refactor branch's files onto the tilebench package layout.

Run inside a checkout of the PR branch, after merging or rebasing onto the
commit that introduced the package layout. It `git mv`s any file still sitting
under a pre-refactor source directory and rewrites the imports inside the moved
files. Nothing else is touched, so an NKI operator PR becomes a one-command
rebase:

    git fetch origin
    git rebase origin/main            # or: git merge origin/main
    python scripts/migrate_pr_paths.py
    git commit -am "migrate to the tilebench package layout"

This is a maintainer convenience for the open pull requests that predate the
layout change. Delete it once they are merged.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Longest prefix first: benchmarks/ must not shadow the ncu artifact rule.
# Every pull request that predates the layout change was measured on B200, so
# its profiling metadata and NCU reports land in the B200 namespace (both are
# generated data under the Git-ignored outputs/ tree).
MOVES = [
    ("tilebench_run/ncu/kernel_counts.json", "outputs/profiling/B200/kernel_counts.json"),
    ("tilebench_run/ncu_catalogue.json", "outputs/profiling/B200/ncu_catalogue.json"),
    ("tilebench_run/ncu/", "outputs/ncu/B200/"),
    ("tilebench_run/", "tilebench/profiling/"),
    ("tools/llm_codegen/", "tilebench/llm_codegen/"),
    ("benchmarks/", "tilebench/benchmarks/"),
    ("core/", "tilebench/core/"),
    ("data/", "tilebench/data/"),
]

IMPORT_RULES = [
    (r'(?m)^(\s*)from (core|data|benchmarks)\.', r'\1from tilebench.\2.'),
    (r'(?m)^(\s*)import (core|data|benchmarks)\.', r'\1import tilebench.\2.'),
    (r'(?m)^(\s*)from (core|data|benchmarks) import ', r'\1from tilebench.\2 import '),
    (r'(?m)^(\s*)from tools\.llm_codegen', r'\1from tilebench.llm_codegen'),
    (r'(?m)^(\s*)import tools\.llm_codegen', r'\1import tilebench.llm_codegen'),
    (r'(?<![\w.])("|f")benchmarks\.operators\.', r'\1tilebench.benchmarks.operators.'),
    (r'(?<![\w.])"core\.nki_profile_worker"', r'"tilebench.core.nki_profile_worker"'),
]


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def target(path: str) -> str | None:
    for old, new in MOVES:
        if path.startswith(old):
            return new + path[len(old):]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    args = ap.parse_args()

    root = Path(run("git", "rev-parse", "--show-toplevel").strip())
    tracked = run("git", "ls-files").splitlines()
    stale = [(p, t) for p in tracked if (t := target(p))]
    if not stale:
        print("nothing to migrate: no tracked file is under a pre-refactor path")
    for old, new in stale:
        print(f"  {old} -> {new}")
        if args.dry_run:
            continue
        (root / new).parent.mkdir(parents=True, exist_ok=True)
        run("git", "mv", old, new)

    # Rewrite imports in every tracked Python file this branch touches, not just
    # the moved ones: a PR may edit a file that already lives in the package.
    edited = 0
    for p in run("git", "ls-files", "*.py").splitlines():
        f = root / p
        if "llm_generated" in p or not f.exists():
            continue
        s = orig = f.read_text()
        for pat, rep in IMPORT_RULES:
            s = re.sub(pat, rep, s)
        if s != orig:
            edited += 1
            print(f"  imports rewritten: {p}")
            if not args.dry_run:
                f.write_text(s)

    print(f"\n{len(stale)} file(s) moved, {edited} file(s) had imports rewritten"
          f"{' (dry run)' if args.dry_run else ''}")
    if not args.dry_run and (stale or edited):
        print("review with `git status`, then commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
