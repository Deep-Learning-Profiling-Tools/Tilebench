#!/usr/bin/env bash
# Snapshot the local results/logs/ onto the raw-log archive branch.
#
# main does not track results/logs/ (raw per-case timing + autotune JSON); the
# archive branch does. Each run builds one commit whose tree is exactly
#     <origin/main's tree>  +  <results/logs/ from the working directory>
# so the archive branch always equals main plus the raw logs. Nothing is
# checked out: the working tree and the real index are left untouched.
#
# Usage (from anywhere inside the repo, after a benchmark run):
#     scripts/archive_logs.sh            # commit to the local archive branch
#     scripts/archive_logs.sh --push     # ...and push it to origin
set -euo pipefail

BRANCH="archive/raw-logs-2026-09-18"
BASE="origin/main"

cd "$(git rev-parse --show-toplevel)"
if [ "$(git symbolic-ref -q --short HEAD)" = "$BRANCH" ]; then
    echo "error: $BRANCH is checked out; run this from another branch" >&2
    exit 1
fi
git fetch -q origin main

# Build the tree in a throwaway index: main's files, then force-add the
# (git-ignored) logs from the working directory.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
tree="$(
    export GIT_INDEX_FILE="$tmp/index"
    git read-tree "$BASE"
    git add -f results/logs
    git write-tree
)"

parent="$(git rev-parse -q --verify "refs/heads/$BRANCH" \
       || git rev-parse -q --verify "refs/remotes/origin/$BRANCH" || true)"
parents=()
if [ -n "$parent" ]; then
    if [ "$(git rev-parse "$parent^{tree}")" = "$tree" ]; then
        echo "$BRANCH already up to date ($(git rev-parse --short "$parent"))"
        exit 0
    fi
    parents+=(-p "$parent")
    git merge-base --is-ancestor "$BASE" "$parent" || parents+=(-p "$BASE")
else
    parents+=(-p "$BASE")
fi

commit="$(git commit-tree "$tree" "${parents[@]}" \
    -m "archive: raw logs on top of main $(git rev-parse --short "$BASE")")"
git update-ref "refs/heads/$BRANCH" "$commit"
echo "$BRANCH -> $(git rev-parse --short "$commit")"

if [ "${1:-}" = "--push" ]; then
    git push origin "$BRANCH"
fi
