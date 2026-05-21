"""Upload all NCU reports (.ncu-rep) and comparison docs (.md) to the
private HuggingFace dataset `bcui2/NCU_report`, preserving directory layout.

Auth: $HUGGING_FACE (env var holds the HF API token).
Code (.py / .json / .out under tilebench_run/ncu/) is NOT uploaded.

Run:
  PYTHONPATH=. python tilebench_run/hf_upload.py
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

api = HfApi(token=token)

api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
print(f"repo ready: {REPO_ID} (dataset, private)", flush=True)

# Upload tilebench_run/ncu/  with structure preserved.
# allow: every .ncu-rep + every .md
# (sweep_log.json, *.py, *.out are NOT in allow_patterns → skipped)
print("uploading … (this may take a while; 6.7 GB across ~270 files)",
      flush=True)
result = api.upload_folder(
    folder_path=str(NCU_DIR),
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo=".",
    allow_patterns=["**/*.ncu-rep", "**/*.md"],
    commit_message="NCU sweep: 222 reports + 45 comparison.md + SUMMARY",
)
print(f"\ndone: {result}", flush=True)
