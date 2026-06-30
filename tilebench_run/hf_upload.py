"""Upload NCU `.ncu-rep` binaries to the private HuggingFace dataset
`bcui2/NCU_report`, preserving the per-op directory layout.

The `.md` reports (comparison.md / SUMMARY.md) are NOT uploaded — they live in
git now (the NCU_analysis branch). Only the large, git-ignored `.ncu-rep`
binaries go to HF. `upload_folder` is diff-based, so unchanged reports are
skipped.

Auth: $HUGGING_FACE holds the HF API token.

Run:
  PYTHONPATH=. python tilebench_run/hf_upload.py            # all ops
  PYTHONPATH=. python tilebench_run/hf_upload.py 1d_conv    # just one op (incremental)
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path("/projects/kzhou6/bcui2/research/tilebench/Tilebench")
NCU_DIR = ROOT / "tilebench_run" / "ncu"
REPO_ID = "bcui2/NCU_report"

token = os.environ.get("HUGGING_FACE")
if not token:
    sys.exit("error: $HUGGING_FACE is not set")

# Optional single-operator argument for incremental updates.
op = sys.argv[1] if len(sys.argv) > 1 else None
folder = NCU_DIR / op if op else NCU_DIR
if not folder.is_dir():
    sys.exit(f"error: {folder} is not a directory")
# Reports live under ncu_report_main/ in the repo (ncu_tma/ is a separate set).
path_in_repo = f"ncu_report_main/{op}" if op else "ncu_report_main"

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
    commit_message=f"NCU .ncu-rep: {op or 'all ops'}",
)
print(f"done: {result}", flush=True)
