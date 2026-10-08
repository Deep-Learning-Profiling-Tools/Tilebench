"""Upload one GPU's ROCm Compute Profiler artifacts to the Hugging Face dataset
`bcui2/NCU_report` (one top-level folder per hardware).

Two namespaces, given separately and never derived from each other:

  --gpu LABEL         local TileBench hardware namespace: the artifacts are read
                      from outputs/rocprof_compute/<gpu>/
                      (tilebench.paths.rocprof_compute_output_dir).
  --hf-folder NAME    top-level folder of the dataset they go to (e.g.
                      AMD_MI300X). One path component; there is no mapping from
                      --gpu, the caller always names the destination.

The upload unit is the complete directory of one profiled pair,
<op>/<backend>_<dtype>/ (workload/, pc_sampling/, analysis/, logs/,
capture.json): every file in it, whatever its extension, so a download can be
re-analyzed with `rocprof-compute analyze -p <...>/workload` without running the
kernel again. A pair is uploaded only when the sweep log records it ok AND its
workloads validate again against the recorded launch sequence; nothing else
under the GPU folder is sent. sweep_log.json and coverage.json go along.
With an operator argument only that operator's pairs are uploaded.

This uploader handles ROCm Compute Profiler output only; NCU reports have
their own uploader (hf_upload.py). `upload_folder` is diff-based, so unchanged
files are skipped; nothing in the dataset is deleted or moved.

Auth: $HUGGING_FACE holds the HF API token.

Run:
  python scripts/profiling/hf_upload_rocm_compute.py --gpu MI300X --hf-folder AMD_MI300X
  python scripts/profiling/hf_upload_rocm_compute.py --gpu MI300X --hf-folder AMD_MI300X vector_add
  ... --verify     # then download what was uploaded and compare every file
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
# Run from anywhere: put the repository root on sys.path so `tilebench` imports
# without setting PYTHONPATH
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tilebench.paths import hardware_label, rocprof_compute_output_dir  # noqa: E402
from tilebench.profiling import rocprof_compute as rc  # noqa: E402

REPO_ID = "bcui2/NCU_report"
#: GPU-level records uploaded with the pairs.
RECORDS = ("sweep_log.json", "coverage.json")
#: Never uploaded: interrupted atomic writes and interpreter caches.
IGNORE_PATTERNS = ["*.tmp", "**/__pycache__/*"]

_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


def _path_component(kind: str):
    def check(value: str) -> str:
        if not _PATH_COMPONENT.fullmatch(value):
            raise argparse.ArgumentTypeError(
                f"invalid {kind} {value!r}: use one path component made of letters, digits, "
                f"'.', '_', '+' or '-', starting with a letter or digit")
        return value
    return check


def uploadable_pairs(gpu: str, op: str | None = None) -> tuple[list[str], list[str]]:
    """(pair directories relative to the GPU folder that may be uploaded,
    reasons for the recorded pairs that may not)."""
    root = rocprof_compute_output_dir(gpu)
    log_path = root / "sweep_log.json"
    if not log_path.is_file():
        return [], [f"no sweep log at {log_path}"]
    ok, refused = [], []
    for e in json.loads(log_path.read_text()):
        if op is not None and e["op"] != op:
            continue
        rel = f"{e['op']}/{e['backend']}_{e['dtype']}"
        if not e.get("ok"):
            refused.append(f"{rel}: not recorded ok")
            continue
        expected = e.get("expected_names") or []
        pair = root / rel
        v = rc.validate_workload(pair / "workload", expected)
        if not v["ok"]:
            refused.append(f"{rel}: workload does not validate: {v['problems'][:2]}")
            continue
        if e.get("pc_sampling") and not rc.validate_pc_sampling(pair / "pc_sampling", expected)["ok"]:
            refused.append(f"{rel}: pc_sampling does not validate")
            continue
        ok.append(rel)
    return ok, refused


def upload_plan(gpu: str, hf_folder: str, pairs: list[str]) -> dict:
    """upload_folder arguments for these pairs: the GPU folder goes to
    <hf-folder>/, restricted to the pair directories and the records."""
    return {"folder_path": str(rocprof_compute_output_dir(gpu)),
            "path_in_repo": hf_folder,
            "allow_patterns": [f"{p}/*" for p in pairs] + list(RECORDS),
            "ignore_patterns": IGNORE_PATTERNS}


def local_files(gpu: str, pairs: list[str]) -> dict[str, str]:
    """{path relative to the GPU folder: sha256} of every file of the pairs."""
    root = rocprof_compute_output_dir(gpu)
    out = {}
    for p in pairs:
        for f in sorted((root / p).rglob("*")):
            if f.is_file() and not f.name.endswith(".tmp") and "__pycache__" not in f.parts:
                out[f.relative_to(root).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def verify(api, gpu: str, hf_folder: str, pairs: list[str]) -> list[str]:
    """Download the uploaded pairs into a temporary directory and compare every
    file with the local one. Returns the mismatches (empty when identical)."""
    from huggingface_hub import snapshot_download
    want = local_files(gpu, pairs)
    with tempfile.TemporaryDirectory(prefix="hf_verify_") as tmp:
        snapshot_download(REPO_ID, repo_type="dataset", token=api.token, local_dir=tmp,
                          allow_patterns=[f"{hf_folder}/{p}/*" for p in pairs])
        base = Path(tmp) / hf_folder
        got = {f.relative_to(base).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
               for p in pairs for f in (base / p).rglob("*") if f.is_file()}
    problems = [f"missing remotely: {k}" for k in want if k not in got]
    problems += [f"differs: {k}" for k in want if k in got and got[k] != want[k]]
    problems += [f"extra remotely: {k}" for k in got if k not in want]
    return problems


def main() -> None:
    """Upload the artifacts. Side effects only when run as a script: importing
    this module must not touch the remote dataset."""
    ap = argparse.ArgumentParser(description="Upload one GPU's ROCm Compute Profiler artifacts "
                                             "to the Hugging Face dataset.")
    ap.add_argument("--gpu", type=hardware_label, required=True, metavar="LABEL",
                    help="Local TileBench hardware label (e.g. MI300X): reads "
                         "outputs/rocprof_compute/<gpu>/")
    ap.add_argument("--hf-folder", type=_path_component("--hf-folder"), required=True, metavar="NAME",
                    help="Top-level folder of the dataset to upload to (e.g. AMD_MI300X), "
                         "one path component; independent of --gpu")
    ap.add_argument("op", nargs="?", default=None, type=_path_component("operator"),
                    help="Single operator, for incremental updates (default: all)")
    ap.add_argument("--verify", action="store_true",
                    help="After the upload, download the uploaded pairs and compare every file")
    args = ap.parse_args()

    token = os.environ.get("HUGGING_FACE")
    if not token:
        sys.exit("error: $HUGGING_FACE is not set")
    folder = rocprof_compute_output_dir(args.gpu)
    if not folder.is_dir():
        sys.exit(f"error: {folder} is not a directory")

    pairs, refused = uploadable_pairs(args.gpu, args.op)
    for r in refused:
        print(f"  not uploaded: {r}", flush=True)
    if not pairs:
        sys.exit("error: no validated pair to upload")

    plan = upload_plan(args.gpu, args.hf_folder, pairs)
    api = HfApi(token=token)
    print(f"uploading {len(pairs)} pair(s) of {args.gpu} → {REPO_ID}/{args.hf_folder}: "
          f"{', '.join(pairs)}", flush=True)
    result = api.upload_folder(
        repo_id=REPO_ID, repo_type="dataset",
        commit_message=f"ROCm Compute Profiler ({args.gpu} → {args.hf_folder}): "
                       f"{args.op or 'all ops'}, {len(pairs)} pair(s)",
        **plan)
    print(f"done: {result}", flush=True)
    if args.verify:
        problems = verify(api, args.gpu, args.hf_folder, pairs)
        if problems:
            sys.exit("verification FAILED:\n  " + "\n  ".join(problems[:50]))
        print(f"verified: {len(local_files(args.gpu, pairs))} files identical after download",
              flush=True)


if __name__ == "__main__":
    main()
