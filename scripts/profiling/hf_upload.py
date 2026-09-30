"""Upload one GPU's NCU `.ncu-rep` binaries (outputs/ncu/<gpu>/) to the
HuggingFace dataset `bcui2/NCU_report`, preserving the per-op directory layout
under a per-GPU prefix, ncu_report_main/<gpu>/<op>/, so that the reports of
different GPUs can never overwrite each other.

The released paper reports (B200) predate this layout and sit directly under
ncu_report_main/<op>/. This script never writes there.

The `.md` reports (comparison.md / SUMMARY.md) are NOT uploaded — they live in
git now (the NCU_analysis branch). Only the large, git-ignored `.ncu-rep`
binaries go to HF. `upload_folder` is diff-based, so unchanged reports are
skipped.

Auth: $HUGGING_FACE holds the HF API token.

Run:
  python scripts/profiling/hf_upload.py --gpu GH200            # all ops
  python scripts/profiling/hf_upload.py --gpu GH200 1d_conv    # just one op (incremental)
"""
import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without setting PYTHONPATH
import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilebench.paths import REPO_ROOT, hardware_label, ncu_output_dir  # noqa: E402

ROOT = REPO_ROOT
REPO_ID = "bcui2/NCU_report"

def main() -> None:
    """Upload the .ncu-rep binaries. Side effects only when run as a script:
    this module is importable as tilebench.profiling.hf_upload, so importing it
    must not touch the remote dataset."""
    ap = argparse.ArgumentParser(description="Upload one GPU's NCU reports.")
    ap.add_argument("--gpu", type=hardware_label, required=True, metavar="LABEL",
                    help="Hardware label (e.g. GH200): uploads outputs/ncu/<gpu>/")
    ap.add_argument("op", nargs="?", default=None,
                    help="Single operator, for incremental updates (default: all)")
    args = ap.parse_args()
    op = args.op

    token = os.environ.get("HUGGING_FACE")
    if not token:
        sys.exit("error: $HUGGING_FACE is not set")

    ncu_dir = ncu_output_dir(args.gpu)
    folder = ncu_dir / op if op else ncu_dir
    if not folder.is_dir():
        sys.exit(f"error: {folder} is not a directory")
    # Reports live under ncu_report_main/<gpu>/ in the repo (ncu_tma/ is a separate set).
    path_in_repo = f"ncu_report_main/{args.gpu}" + (f"/{op}" if op else "")

    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)

    print(f"uploading {'op ' + op if op else 'ALL ops'} → {REPO_ID} (.ncu-rep only) …",
          flush=True)
    result = api.upload_folder(
        folder_path=str(folder),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        # both forms so a per-op folder (files at root) and the all-ops folder
        # (files one level down) are matched.
        allow_patterns=["*.ncu-rep", "**/*.ncu-rep"],
        commit_message=f"NCU .ncu-rep ({args.gpu}): {op or 'all ops'}",
    )
    print(f"done: {result}", flush=True)


if __name__ == "__main__":
    main()
