#!/usr/bin/env bash
# Snapshot git-ignored experiment artifacts onto the archive branch.
#
# main version-controls source and the summary CSVs only. Raw benchmark logs
# and LLM-generated trajectories are generated artifacts: ignored on main,
# backed up here. Each run builds one commit whose tree is
#     <origin/main's tree>
#   + <every artifact already on the archive branch>      (cumulative)
#   + <the selected artifact directories from the working directory>
# with the working directory winning on a path collision. Artifacts that this
# machine does not hold are inherited from the archive tip, never dropped, so
# a partial checkout adds to the backup instead of replacing it. Nothing is
# checked out: the working tree and the real index are left untouched, and
# history is only ever appended to.
#
# Usage (from anywhere inside the repo):
#     scripts/archive_artifacts.sh --logs          # results/logs/
#     scripts/archive_artifacts.sh --llm           # tilebench/benchmarks/llm_generated/
#     scripts/archive_artifacts.sh --all           # both
#     ... --push                                   # also push the archive branch
#
# scripts/run_bench.py runs `archive_logs.sh` (= --logs) after every benchmark.
# LLM trajectories are archived only on request, at milestones worth keeping.
set -euo pipefail

BRANCH="archive/raw-logs-2026-09-18"
BASE="origin/main"

LOGS_PATH="results/logs"
LLM_PATH="tilebench/benchmarks/llm_generated"
# Every path the archive owns. All of them are carried forward on every run,
# whichever mode is selected. benchmarks/llm_generated/ is the pre-refactor
# location: a historical snapshot that is kept, never written to.
ARCHIVE_PATHS=("$LOGS_PATH" "benchmarks/llm_generated" "$LLM_PATH")

usage() { sed -n '15,20p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

selected=()
push=0
for arg in "$@"; do
    case "$arg" in
        --logs) selected+=("$LOGS_PATH") ;;
        --llm)  selected+=("$LLM_PATH") ;;
        --all)  selected+=("$LOGS_PATH" "$LLM_PATH") ;;
        --push) push=1 ;;
        *) echo "error: unknown option $arg" >&2; usage ;;
    esac
done
[ "${#selected[@]}" -gt 0 ] || { echo "error: choose --logs, --llm or --all" >&2; usage; }

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

# Build the tree in a throwaway index: main's files, then everything already
# archived, then the selected (git-ignored) directories from the working tree.
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
tree="$(
    export GIT_INDEX_FILE="$tmp/index"
    git read-tree "$BASE"
    for tip in $remote_tip $local_tip; do
        git ls-tree -r "$tip" -- "${ARCHIVE_PATHS[@]}" | git update-index --index-info
    done
    for path in "${selected[@]}"; do
        # update-index ignores .gitignore and only ever adds, so artifacts
        # missing from this machine stay archived. Interpreter caches are not
        # artifacts. (git add -f with exclude pathspecs is unreliable on
        # ignored directories in older git.)
        if [ -d "$path" ]; then
            find "$path" -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 \
                | git update-index --add -z --stdin
        fi
    done
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
        -m "archive: ${selected[*]} on top of main $(git rev-parse --short "$BASE")")"
    echo "$BRANCH -> $(git rev-parse --short "$commit")"
fi
[ "$commit" = "$local_tip" ] || git update-ref "refs/heads/$BRANCH" "$commit" "$local_tip"

if [ "$push" -eq 1 ]; then
    git push origin "$BRANCH"
fi
