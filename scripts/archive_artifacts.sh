#!/usr/bin/env bash
# Snapshot git-ignored experiment artifacts onto an archive branch.
#
# Maintainer tool. It lives on the archive branch it maintains, never on main or
# a development branch. Take it from there and run it inside a checkout of the
# source tree that holds the artifacts (the archive branch itself must not be
# checked out anywhere):
#     git show origin/archive/tilebenchpp-2026-10:scripts/archive_artifacts.sh > /tmp/archive_artifacts.sh
#     bash /tmp/archive_artifacts.sh --branch archive/tilebenchpp-2026-10 \
#         --base <commit the artifacts were produced from> --logs --gpu GH200 --push
#
# Usage:
#     archive_artifacts.sh --branch archive/<name> --base <commit-ish> [--init] [CLASS...] [--gpu LABEL] [--push]
#
#     --branch B     the archive branch: archive/<one path component>. Frozen
#                    branches (archive/raw-logs-2026-09-18) are refused.
#     --base REF     the source commit the artifacts were produced from (a SHA,
#                    or a ref such as feature/tilebenchpp-multiarch). Required;
#                    resolved to an exact SHA that is recorded in the commit and
#                    added as a parent, so it stays reachable from the archive.
#     --init         create the branch (it must not exist, locally or on origin).
#                    Its first commit holds this tool and its tests, taken from
#                    the directory above this script, plus any CLASS given.
#     --logs         results/<gpu>/logs/            (needs --gpu)
#     --profiling    outputs/profiling/<gpu>/       (needs --gpu)
#     --llm          tilebench/benchmarks/llm_generated/
#     --all          --logs --profiling --llm       (needs --gpu)
#     --gpu LABEL    hardware label, one safe path component (B200, GH200, MI300X)
#     --push         push the branch to origin (never forced)
#
# The archive tree holds archive material only: this tool, its tests and the
# artifacts. Source code and the summary CSVs (results/<gpu>/csv/) are not
# copied; the source is the --base commit. Each run builds one commit whose
# tree is the archive tip's tree plus the selected directories of the working
# directory, the working directory winning on a path collision. Nothing is
# ever removed: artifacts this machine does not hold (another GPU's logs, an
# older trajectory) are carried forward. NCU reports (*.ncu-rep), interpreter
# caches and files over 50 MiB are never archived. Nothing is checked out: the
# caller's working tree and index are left untouched, and history is only
# appended to. A local branch that has diverged from origin, or a push that is
# rejected, stops the run instead of being resolved by overwriting.
set -euo pipefail

FROZEN_BRANCHES=("archive/raw-logs-2026-09-18")
LOGS_DIR="results/%s/logs"
PROFILING_DIR="outputs/profiling/%s"
LLM_DIR="tilebench/benchmarks/llm_generated"
TOOL_FILES=("scripts/archive_artifacts.sh" "tests/test_archive_artifacts.py")
MAX_FILE_BYTES=$((50 * 1024 * 1024))

usage() { sed -n '/^# Usage/,/^#     --push/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }
die() { echo "error: $*" >&2; exit 1; }
bad_usage() { echo "error: $*" >&2; usage; }

branch=""; base=""; gpu=""; init=0; push=0
want_logs=0; want_profiling=0; want_llm=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --branch) [ "$#" -ge 2 ] || bad_usage "--branch needs a value"; branch="$2"; shift ;;
        --branch=*) branch="${1#--branch=}" ;;
        --base) [ "$#" -ge 2 ] || bad_usage "--base needs a value"; base="$2"; shift ;;
        --base=*) base="${1#--base=}" ;;
        --gpu) [ "$#" -ge 2 ] || bad_usage "--gpu needs a hardware label"; gpu="$2"; shift ;;
        --gpu=*) gpu="${1#--gpu=}" ;;
        --init) init=1 ;;
        --logs) want_logs=1 ;;
        --profiling) want_profiling=1 ;;
        --llm) want_llm=1 ;;
        --all) want_logs=1; want_profiling=1; want_llm=1 ;;
        --push) push=1 ;;
        -h|--help) usage ;;
        *) bad_usage "unknown option $1" ;;
    esac
    shift
done

# --- arguments ---------------------------------------------------------------
[ -n "$branch" ] || bad_usage "--branch is required (e.g. --branch archive/tilebenchpp-2026-10)"
if ! [[ "$branch" =~ ^archive/[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || [[ "$branch" == *..* ]] \
        || [[ "$branch" == *.lock ]] || ! git check-ref-format "refs/heads/$branch" 2>/dev/null; then
    bad_usage "invalid archive branch '$branch': use archive/<name>, one component of letters, digits, '.', '_' or '-'"
fi
for frozen in "${FROZEN_BRANCHES[@]}"; do
    [ "$branch" != "$frozen" ] || bad_usage "$branch is frozen and is never written to"
done
[ -n "$base" ] || bad_usage "--base is required: the source commit the artifacts were produced from"
if [ $((want_logs + want_profiling)) -gt 0 ] && ! [[ "$gpu" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
    bad_usage "--logs, --profiling and --all need --gpu <label> (e.g. --gpu GH200); got '${gpu}'"
fi
if [ -n "$gpu" ] && [ $((want_logs + want_profiling)) -eq 0 ]; then
    bad_usage "--gpu is only used with --logs, --profiling or --all"
fi
[ $((init + want_logs + want_profiling + want_llm)) -gt 0 ] || bad_usage "choose --logs, --profiling, --llm or --all (or --init)"

# Resolved before the cd below: the tool files --init records sit beside this script.
tool_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
top="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a Git working tree"
cd "$top"

base_sha="$(git rev-parse -q --verify "${base}^{commit}" 2>/dev/null)" || die "--base '$base' is not a commit here"

classes=(); selected=()
if [ "$want_logs" -eq 1 ]; then classes+=(logs); selected+=("$(printf "$LOGS_DIR" "$gpu")"); fi
if [ "$want_profiling" -eq 1 ]; then classes+=(profiling); selected+=("$(printf "$PROFILING_DIR" "$gpu")"); fi
if [ "$want_llm" -eq 1 ]; then classes+=(llm); selected+=("$LLM_DIR"); fi
for path in "${selected[@]}"; do
    [ -d "$path" ] || die "$path does not exist in $top; nothing to archive for it (select the classes this machine holds)"
done

if git worktree list --porcelain | grep -qxF "branch refs/heads/$branch"; then
    die "$branch is checked out in a worktree; archive it from a checkout of another branch"
fi

# --- tip: local and origin must agree or one must contain the other --------
git remote get-url origin >/dev/null 2>&1 || die "no 'origin' remote"
set +e
remote_line="$(git ls-remote --exit-code origin "refs/heads/$branch")"
ls_status=$?
set -e
case "$ls_status" in
    0) remote_tip="${remote_line%%[[:space:]]*}"
       git fetch -q origin "+refs/heads/$branch:refs/remotes/origin/$branch" ;;
    2) remote_tip="" ;;
    *) die "cannot reach origin to check $branch (exit $ls_status); nothing was changed" ;;
esac
local_tip="$(git rev-parse -q --verify "refs/heads/$branch" || true)"

if [ "$init" -eq 1 ]; then
    [ -z "$local_tip$remote_tip" ] || die "$branch already exists; drop --init to append to it"
    tip=""
elif [ -z "$local_tip$remote_tip" ]; then
    die "$branch does not exist locally or on origin; create it with --init"
elif [ -z "$remote_tip" ] || [ "$local_tip" = "$remote_tip" ]; then
    tip="$local_tip"
elif [ -z "$local_tip" ] || git merge-base --is-ancestor "$local_tip" "$remote_tip"; then
    tip="$remote_tip"                       # local missing or stale: build on origin
elif git merge-base --is-ancestor "$remote_tip" "$local_tip"; then
    tip="$local_tip"                        # local ahead (not pushed yet)
else
    die "local $branch ($(git rev-parse --short "$local_tip")) and origin/$branch ($(git rev-parse --short "$remote_tip")) have diverged; reconcile them by hand, nothing was changed"
fi

# --- tree ----------------------------------------------------------------
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
in_index() { GIT_INDEX_FILE="$tmp/$1" git "${@:2}"; }
entries() { in_index "$1" ls-files -s | sed 's/^[0-9]* \([0-9a-f]*\) [0-9]\t/\1\t/' | LC_ALL=C sort -t $'\t' -k2; }

if [ -n "$tip" ]; then in_index index read-tree "$tip"; else in_index index read-tree --empty; fi

if [ "$init" -eq 1 ]; then
    for f in "${TOOL_FILES[@]}"; do
        [ -f "$tool_root/$f" ] || die "--init records the tool from $tool_root, which has no $f; run it from a directory holding ${TOOL_FILES[*]}"
    done
    for f in "${TOOL_FILES[@]}"; do
        mode=100644; [ -x "$tool_root/$f" ] && mode=100755
        in_index index update-index --add --cacheinfo "$mode,$(git hash-object -w "$tool_root/$f"),$f"
    done
fi

in_index work read-tree --empty
for path in "${selected[@]}"; do
    too_large="$(find "$path" -type f -size +"$((MAX_FILE_BYTES / 1024))"k ! -name '*.ncu-rep' | LC_ALL=C sort)"
    [ -z "$too_large" ] || die "files over $((MAX_FILE_BYTES / 1024 / 1024)) MiB belong in external artifact storage, not on $branch:
$(echo "$too_large" | sed 's/^/    /')"
    # update-index ignores .gitignore; only regular files, no caches or NCU reports
    find "$path" -type f ! -name '*.pyc' ! -path '*/__pycache__/*' ! -name '*.ncu-rep' -print0 \
        | in_index work update-index --add -z --stdin
done
added="$(LC_ALL=C join -t $'\t' -1 2 -2 2 -v 1 <(entries work) <(entries index) | wc -l)"
updated="$(LC_ALL=C join -t $'\t' -1 2 -2 2 <(entries work) <(entries index) | awk -F'\t' '$2 != $3' | wc -l)"
in_index work ls-files -s -z | in_index index update-index -z --index-info
tree="$(in_index index write-tree)"

# --- commit ----------------------------------------------------------------
short="$(git rev-parse --short "$base_sha")"
if [ -n "$tip" ] && [ "$(git rev-parse "$tip^{tree}")" = "$tree" ]; then
    echo "$branch already up to date ($(git rev-parse --short "$tip")); no artifact changed"
    commit="$tip"
else
    parents=()
    [ -z "$tip" ] || parents+=(-p "$tip")
    if [ -z "$tip" ] || ! git merge-base --is-ancestor "$base_sha" "$tip"; then parents+=(-p "$base_sha"); fi
    what="${classes[*]:-}"; what="${what// /+}"
    if [ "$init" -eq 1 ]; then
        subject="archive: init $branch${what:+ with ${gpu:+$gpu }$what} from source $short"
    else
        subject="archive: ${gpu:+$gpu }$what from source $short"
    fi
    body="Archive-Branch: $branch
Source-Base: $base_sha
Source-Ref: $base"
    [ -z "$gpu" ] || body+="
Hardware: $gpu"
    body+="
Artifact-Classes: ${classes[*]:-none}
Files-Added: $added
Files-Updated: $updated"
    commit="$(git commit-tree "$tree" "${parents[@]}" -m "$subject" -m "$body")"
    echo "$branch -> $(git rev-parse --short "$commit")  ($subject; $added added, $updated updated)"
fi

# Compare-and-swap: fails if the local branch moved while this ran.
if [ "$commit" != "$local_tip" ]; then
    git update-ref -m "archive_artifacts" "refs/heads/$branch" "$commit" "$local_tip"
fi

if [ "$push" -eq 1 ] && [ "$commit" != "$remote_tip" ]; then
    if ! git push origin "refs/heads/$branch:refs/heads/$branch"; then
        if [ -n "$local_tip" ]; then
            git update-ref "refs/heads/$branch" "$local_tip" "$commit"
        else
            git update-ref -d "refs/heads/$branch" "$commit"
        fi
        die "push of $branch was rejected (origin moved?); the local branch was restored, run again"
    fi
fi
