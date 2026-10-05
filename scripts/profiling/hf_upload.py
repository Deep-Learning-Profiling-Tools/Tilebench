"""Upload one GPU's Nsight Compute `.ncu-rep` binaries to the Hugging Face
dataset `bcui2/NCU_report`.

Two namespaces, given separately and never derived from each other:

  --gpu LABEL         local TileBench hardware namespace: the reports are read
                      from outputs/ncu/<gpu>/ (tilebench.paths.ncu_output_dir).
  --hf-folder NAME    top-level folder of the dataset the reports go to, one per
                      hardware (e.g. NVIDIA_B200, NVIDIA_GH200). It must be one
                      path component; there is no list of accepted names and no
                      mapping from --gpu, so the caller always names the
                      destination.

The per-operator layout is kept: outputs/ncu/<gpu>/<op>/ goes to
<hf-folder>/<op>/. With an operator argument only that operator is uploaded.

This is an NCU uploader only: it uploads `*.ncu-rep` files and nothing else
(the `.md` write-ups live in Git). Profiler output of other hardware (AMD,
Neuron) is not handled here. `upload_folder` is diff-based, so unchanged
reports are skipped; nothing in the dataset is deleted or moved.

Auth: $HUGGING_FACE holds the HF API token.

Run:
  python scripts/profiling/hf_upload.py --gpu GH200 --hf-folder NVIDIA_GH200            # all ops
  python scripts/profiling/hf_upload.py --gpu GH200 --hf-folder NVIDIA_GH200 1d_conv    # one op
"""
import argparse
import os
import re
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
#: Both forms, so a per-op folder (files at its root) and the all-ops folder
#: (files one level down) are matched.
ALLOW_PATTERNS = ["*.ncu-rep", "**/*.ncu-rep"]

_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


def _path_component(kind: str):
    def check(value: str) -> str:
        if not _PATH_COMPONENT.fullmatch(value):
            raise argparse.ArgumentTypeError(
                f"invalid {kind} {value!r}: use one path component made of letters, digits, "
                f"'.', '_', '+' or '-', starting with a letter or digit")
        return value
    return check


def upload_plan(gpu: str, hf_folder: str, op: str | None = None) -> tuple[Path, str]:
    """(local source folder, folder in the dataset) of one upload."""
    if op is None:
        return ncu_output_dir(gpu), hf_folder
    return ncu_output_dir(gpu) / op, f"{hf_folder}/{op}"


def main() -> None:
    """Upload the .ncu-rep binaries. Side effects only when run as a script:
    importing this module must not touch the remote dataset."""
    ap = argparse.ArgumentParser(description="Upload one GPU's NCU reports to the Hugging Face dataset.")
    ap.add_argument("--gpu", type=hardware_label, required=True, metavar="LABEL",
                    help="Local TileBench hardware label (e.g. GH200): reads outputs/ncu/<gpu>/")
    ap.add_argument("--hf-folder", type=_path_component("--hf-folder"), required=True, metavar="NAME",
                    help="Top-level folder of the dataset to upload to (e.g. NVIDIA_GH200), "
                         "one path component; independent of --gpu")
    ap.add_argument("op", nargs="?", default=None, type=_path_component("operator"),
                    help="Single operator, for incremental updates (default: all)")
    args = ap.parse_args()
    op = args.op

    token = os.environ.get("HUGGING_FACE")
    if not token:
        sys.exit("error: $HUGGING_FACE is not set")

    folder, path_in_repo = upload_plan(args.gpu, args.hf_folder, op)
    if not folder.is_dir():
        sys.exit(f"error: {folder} is not a directory")

    api = HfApi(token=token)
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)

    print(f"uploading {'op ' + op if op else 'ALL ops'} from {folder} → "
          f"{REPO_ID}/{path_in_repo} (.ncu-rep only) …", flush=True)
    result = api.upload_folder(
        folder_path=str(folder),
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=ALLOW_PATTERNS,
        commit_message=f"NCU .ncu-rep ({args.gpu} → {args.hf_folder}): {op or 'all ops'}",
    )
    print(f"done: {result}", flush=True)


if __name__ == "__main__":
    main()
