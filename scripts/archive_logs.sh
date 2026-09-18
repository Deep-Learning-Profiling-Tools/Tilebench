#!/usr/bin/env bash
# Snapshot the local results/logs/ onto the raw-log archive branch.
#
# main does not track results/logs/ (raw per-case timing + autotune JSON); the
# archive branch does. Each run builds one commit whose tree is
#     <origin/main's tree>  +  <logs already on the archive branch>
#                           +  <results/logs/ from the working directory>
# with the working directory winning on conflicts, so the archive branch always
# equals main plus every raw log archived so far. A machine that only holds
# some of the logs (e.g. a fresh clone on another host) adds to the archive
# instead of replacing it. Nothing is checked out: the working tree and the
# real index are left untouched.
#
# scripts/run_bench.py runs this after every benchmark (local commit only).
# Usage (from anywhere inside the repo):
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
# Offline is fine: fall back to the last fetched refs.
git fetch -q origin main 2>/dev/null || true
git fetch -q origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" 2>/dev/null || true

local_tip="$(git rev-parse -q --verify "refs/heads/$BRANCH" || true)"
remote_tip="$(git rev-parse -q --verify "refs/remotes/origin/$BRANCH" || true)"

# Build the tree in a throwaway index: main's files, then the archived logs,
# then the (git-ignored) logs from the working directory.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
tree="$(
    export GIT_INDEX_FILE="$tmp/index"
    git read-tree "$BASE"
    for tip in $remote_tip $local_tip; do
        git ls-tree -r "$tip" -- results/logs | git update-index --index-info
    done
    # --ignore-removal: archived logs missing from this machine must stay archived
    if [ -d results/logs ]; then git add -f --ignore-removal results/logs; fi
    git write-tree
)"

# Parents: the archive tips and main, minus anything another parent already contains.
parents=()
add_parent() {
    local c="$1" p kept=()
    [ -n "$c" ] || return 0
    for p in "${parents[@]}"; do
        if git merge-base --is-ancestor "$c" "$p"; then return 0; fi
        git merge-base --is-ancestor "$p" "$c" || kept+=("$p")
    done
    parents=("${kept[@]}" "$c")
}
add_parent "$local_tip"
add_parent "$remote_tip"
add_parent "$(git rev-parse "$BASE")"

if [ "${#parents[@]}" -eq 1 ] && [ "$(git rev-parse "${parents[0]}^{tree}")" = "$tree" ]; then
    commit="${parents[0]}"
    echo "$BRANCH already up to date ($(git rev-parse --short "$commit"))"
else
    commit="$(git commit-tree "$tree" $(printf -- '-p %s ' "${parents[@]}") \
        -m "archive: raw logs on top of main $(git rev-parse --short "$BASE")")"
    echo "$BRANCH -> $(git rev-parse --short "$commit")"
fi
[ "$commit" = "$local_tip" ] || git update-ref "refs/heads/$BRANCH" "$commit" "$local_tip"

if [ "${1:-}" = "--push" ]; then
    git push origin "$BRANCH"
fi
