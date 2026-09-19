#!/usr/bin/env bash
# Snapshot git-ignored experiment artifacts onto the archive branch.
#
# Maintainer tool. It lives on the archive branch only: the public branch does
# not ship it, and running a benchmark never archives anything. To use it, take
# it from this branch and run it inside a checkout of the public branch, where
# the artifacts are (the archive branch itself must not be checked out there):
#     git show origin/archive/raw-logs-2026-09-18:scripts/archive_artifacts.sh > /tmp/archive_artifacts.sh
#     bash /tmp/archive_artifacts.sh --logs --gpu B200
#
# main version-controls source and the summary CSVs only. Raw benchmark logs
# and LLM-generated trajectories are generated artifacts: ignored on main,
# backed up here. Each run builds one commit whose tree is
#     <the public code baseline, without the legacy results/logs/>
#   + <everything the archive already keeps that the baseline lacks>  (cumulative)
#   + <the selected artifact directories from the working directory>
# with the working directory winning on a path collision. The baseline is
# origin/main. When the archive tip already contains origin/main there is
# nothing new to absorb and the tip's own tree is kept instead, so code merged
# into the archive from a newer baseline (a branch ahead of main) is never
# rolled back to main by an archive run. The summary CSVs are part of the
# baseline (results/<gpu>/csv/) and are never copied by this script. Artifacts
# that this machine does not hold are inherited from the archive tip, never
# dropped, so a partial checkout adds to the backup instead of replacing it.
# Nothing is checked out: the working tree and the real index are left
# untouched, and history is only ever appended to.
#
# Usage (from anywhere inside the repo):
#     archive_artifacts.sh --logs --gpu B200   # results/B200/logs/
#     archive_artifacts.sh --llm               # tilebench/benchmarks/llm_generated/
#     archive_artifacts.sh --all --gpu B200    # both
#     ... --push                               # also push the archive branch
#
# Raw logs are scoped by hardware (results/<gpu>/logs/), so --logs and --all
# take the label of the GPU whose logs to snapshot; the logs of every other GPU
# already on the archive branch are carried forward untouched. The whole logs/
# tree goes in, including the NKI profiles recorded with that campaign
# (logs/nki_profiles/): there is no separate NKI archive path. LLM trajectories
# are archived only on request, at milestones worth keeping.
set -euo pipefail

BRANCH="archive/raw-logs-2026-09-18"
BASE="origin/main"

LLM_PATH="tilebench/benchmarks/llm_generated"
# Fixed paths the archive owns, carried forward on every run whichever mode is
# selected. benchmarks/llm_generated/ is a legacy location (before the
# tilebench/ package): a historical snapshot that is kept and never written to.
# outputs/profiling/ holds the NCU metadata (catalogue, kernel counts) of each
# GPU: generated data that the public tree does not track. The last two are
# this script and its tests: the public tree does not have them, so they must
# carry themselves forward or the first run on a newer main would delete them.
ARCHIVE_PATHS=("benchmarks/llm_generated" "$LLM_PATH" "outputs/profiling"
               "scripts/archive_artifacts.sh" "tests/test_archive_scripts.py")
# Legacy raw-log location, from before results were scoped by hardware. Every
# log ever stored there was measured on B200 (the paper campaign), so it is
# canonicalised to results/B200/logs/<same relative path> and never kept:
#   - main may still track it until the layout change lands, so it is dropped
#     from the base tree instead of being inherited;
#   - an archive tip that still holds it is migrated, byte for byte;
#   - a migrated file that meets a DIFFERENT file at its destination (already
#     archived, or in the working directory) aborts the run: nothing is guessed.
LEGACY_LOGS="results/logs"
LEGACY_LOGS_DEST="results/B200/logs"

usage() { sed -n '/^# Usage/,/--push/p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

want_logs=0
want_llm=0
gpu=""
push=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --logs) want_logs=1 ;;
        --llm)  want_llm=1 ;;
        --all)  want_logs=1; want_llm=1 ;;
        --gpu)  [ "$#" -ge 2 ] || { echo "error: --gpu needs a hardware label" >&2; usage; }
                gpu="$2"; shift ;;
        --gpu=*) gpu="${1#--gpu=}" ;;
        --push) push=1 ;;
        *) echo "error: unknown option $1" >&2; usage ;;
    esac
    shift
done
[ "$((want_logs + want_llm))" -gt 0 ] || { echo "error: choose --logs, --llm or --all" >&2; usage; }

selected=()
if [ "$want_logs" -eq 1 ]; then
    # Same rule as tilebench.paths.hardware_label: one safe path component.
    if ! [[ "$gpu" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
        echo "error: --logs needs --gpu <label> (e.g. --gpu B200); got '${gpu}'" >&2
        usage
    fi
    selected+=("results/$gpu/logs")
fi
[ "$want_llm" -eq 0 ] || selected+=("$LLM_PATH")

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
# Four throwaway indexes keep the sources apart until they are reconciled:
#   index    the tree being built
#   legacy   results/logs/ of the archive tips, re-rooted at results/B200/logs/
#   canon    results/<gpu>/logs/ already on the archive tips
#   work     the selected directories of the working directory
in_index() { GIT_INDEX_FILE="$tmp/$1" git "${@:2}"; }
entries() { in_index "$1" ls-files -s | sed 's/^[0-9]* \([0-9a-f]*\) [0-9]\t/\1\t/' | LC_ALL=C sort -t $'\t' -k2; }

base_tree="$BASE"
for tip in $remote_tip $local_tip; do
    if git merge-base --is-ancestor "$BASE" "$tip"; then base_tree="$tip"; fi
done
in_index index read-tree "$base_tree"
# main's copy of the legacy logs is not inherited (see LEGACY_LOGS above).
in_index index ls-files -z -- "$LEGACY_LOGS" | in_index index update-index -z --force-remove --stdin

for tip in $remote_tip $local_tip; do
    git ls-tree -r "$tip" -- "${ARCHIVE_PATHS[@]}" | in_index index update-index --index-info
    # Files that left the public tilebench/profiling/ (cluster launch scripts,
    # campaign-specific harnesses) and are kept here only: whatever the tip
    # holds there that the base tree does not.
    mapfile -t retired < <(LC_ALL=C comm -13 \
        <(in_index index ls-files -- tilebench/profiling | LC_ALL=C sort) \
        <(git ls-tree -r --name-only "$tip" -- tilebench/profiling | LC_ALL=C sort))
    if [ "${#retired[@]}" -gt 0 ]; then
        git ls-tree -r "$tip" -- "${retired[@]}" | in_index index update-index --index-info
    fi
    # the raw logs of every GPU archived so far: results/<gpu>/logs/
    git ls-tree -r -z "$tip" -- results \
        | { grep -z -E $'\tresults/[^/]+/logs/' || true; } \
        | in_index canon update-index -z --index-info
    # legacy raw logs, re-rooted; a later tip wins over an earlier one
    git ls-tree -r -z "$tip" -- "$LEGACY_LOGS" \
        | sed -z "s|\t$LEGACY_LOGS/|\t$LEGACY_LOGS_DEST/|" \
        | in_index legacy update-index -z --index-info
done
for path in "${selected[@]}"; do
    # update-index ignores .gitignore and only ever adds, so artifacts
    # missing from this machine stay archived. Interpreter caches are not
    # artifacts. (git add -f with exclude pathspecs is unreliable on
    # ignored directories in older git.)
    if [ -d "$path" ]; then
        find "$path" -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 \
            | in_index work update-index --add -z --stdin
    fi
done

# A migrated legacy log may only meet an identical file at its destination.
conflicts="$(
    for other in canon work; do
        LC_ALL=C join -t $'\t' -j 2 <(entries legacy) <(entries "$other") | awk -F'\t' '$2 != $3 {print $1}'
    done | sort -u
)"
if [ -n "$conflicts" ]; then
    {
        echo "error: legacy $LEGACY_LOGS/ logs cannot be migrated to $LEGACY_LOGS_DEST/:"
        echo "these destinations already hold a DIFFERENT file, and neither version is dropped silently:"
        echo "$conflicts" | sed 's/^/    /'
        echo "Nothing was committed. Decide which version to keep, then run again."
    } >&2
    exit 1
fi

# Lowest to highest priority: migrated legacy, archived, working directory.
for src in legacy canon work; do
    in_index "$src" ls-files -s -z | in_index index update-index -z --index-info
done
tree="$(in_index index write-tree)"

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
    note=""
    migrated="$(entries legacy | wc -l)"
    [ "$migrated" -eq 0 ] || note="; migrate $migrated legacy $LEGACY_LOGS/ logs to $LEGACY_LOGS_DEST/"
    commit="$(git commit-tree "$tree" $(printf -- '-p %s ' "${parents[@]}") \
        -m "archive: ${selected[*]} on top of main $(git rev-parse --short "$BASE")$note")"
    echo "$BRANCH -> $(git rev-parse --short "$commit")"
fi
[ "$commit" = "$local_tip" ] || git update-ref "refs/heads/$BRANCH" "$commit" "$local_tip"

if [ "$push" -eq 1 ]; then
    git push origin "$BRANCH"
fi
