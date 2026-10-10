"""Fetch one revision-pinned TileBench NVIDIA report and optionally extract it."""

import argparse
import hashlib
import json
from pathlib import Path
import re


REPO = "bcui2/NCU_report"
HARDWARE = {"B200": "NVIDIA_B200", "GH200": "NVIDIA_GH200",
            "NVIDIA_B200": "NVIDIA_B200", "NVIDIA_GH200": "NVIDIA_GH200"}


def remote_path(hardware, operator, backend, dtype):
    if hardware not in HARDWARE:
        raise ValueError("Only B200/GH200 NVIDIA reports are supported; AMD artifacts are not NCU reports")
    for value in (operator, backend, dtype):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
            raise ValueError(f"Invalid artifact name: {value!r}")
    return f"{HARDWARE[hardware]}/{operator}/{backend}_{dtype}.ncu-rep"


def fetch(api, download, hardware, operator, backend, dtype, revision, output_dir):
    path = remote_path(hardware, operator, backend, dtype)
    commit = api.repo_info(REPO, repo_type="dataset", revision=revision).sha
    identity = {"dataset": REPO, "revision": commit, "remote_path": path}
    output_dir = Path(output_dir)
    receipt_path = output_dir / "download.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("Output directory belongs to another case/revision; use a new directory")
    entries = api.get_paths_info(REPO, [path], repo_type="dataset", revision=commit)
    entry = next((e for e in entries if e.path == path and hasattr(e, "size")), None)
    if entry is None:
        raise FileNotFoundError(f"No report at {REPO}@{commit}/{path}")
    local = Path(download(REPO, path, repo_type="dataset", revision=commit,
                          local_dir=output_dir / "artifacts"))
    digest = hashlib.sha256()
    with local.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    sha256 = digest.hexdigest()
    remote_sha256 = entry.lfs.sha256 if getattr(entry, "lfs", None) else None
    if local.stat().st_size != entry.size:
        raise ValueError("Downloaded report size does not match Hub metadata")
    if remote_sha256 and sha256 != remote_sha256:
        raise ValueError("Downloaded report SHA-256 does not match Hub metadata")
    receipt = {**identity, "requested_revision": revision,
               "local_path": str(local.resolve()), "bytes": local.stat().st_size,
               "sha256": sha256, "remote_sha256": remote_sha256,
               "remote_hash_verified": bool(remote_sha256),
               "identity_status": "Filename selected; capture shape/config/source still require verification"}
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return local, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hardware", choices=sorted(HARDWARE))
    parser.add_argument("operator")
    parser.add_argument("backend")
    parser.add_argument("dtype")
    parser.add_argument("--revision", default="main", help="Resolved to an immutable commit")
    parser.add_argument("--output-dir", type=Path, required=True, help="One case/revision per directory")
    parser.add_argument("--extract", action="store_true", help="Also export scalar metrics/action inventory")
    parser.add_argument("--metric", action="append", help="Exact metric; repeat; implies --extract")
    args = parser.parse_args()
    try:
        remote_path(args.hardware, args.operator, args.backend, args.dtype)
    except ValueError as error:
        parser.error(str(error))
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        parser.error("Hugging Face client unavailable; use an approved client environment, not an automatic install")
    should_extract = args.extract or args.metric is not None
    if should_extract:
        try:
            import ncu_report
            from ncu_extract import extract
        except ImportError:
            parser.error("NCU report API unavailable; configure its module path or omit --extract")
    local, receipt = fetch(HfApi(), hf_hub_download, args.hardware, args.operator,
                           args.backend, args.dtype, args.revision, args.output_dir)
    summary = {"receipt": str(args.output_dir / "download.json"),
               "report": str(local), "revision": receipt["revision"],
               "remote_hash_verified": receipt["remote_hash_verified"]}
    if should_extract:
        report = ncu_report.load_report(str(local.resolve()))
        result = extract(report, local, args.metric)
        result["report_sha256"] = receipt["sha256"]
        output = args.output_dir / "ncu.json"
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        summary.update({"extraction": str(output), "actions": len(result["actions"]),
                        "metrics": len(result["metrics"]), "errors": len(result["errors"])})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
