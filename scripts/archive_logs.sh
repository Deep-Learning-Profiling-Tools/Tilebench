#!/usr/bin/env bash
# Snapshot results/logs/ onto the raw-log archive branch.
#
# Compatibility entry point: scripts/run_bench.py calls this after every
# benchmark. It archives the raw logs only; LLM-generated trajectories are
# archived on request with `archive_artifacts.sh --llm`.
#
#     scripts/archive_logs.sh            # commit to the local archive branch
#     scripts/archive_logs.sh --push     # ...and push it to origin
exec "$(dirname "$0")/archive_artifacts.sh" --logs "$@"
