"""Inventory of the NVIDIA (B200, GH200) .ncu-rep reports used by the paper-figure extraction.

Offline only: hashes local report files and compares them with the Hugging Face LFS metadata of
bcui2/NCU_report at a pinned revision (no download, no GPU). Writes <cache>/inventory_<device>.json.

Usage:
  python scripts/paper_figures/nvidia_inventory.py --device B200 --cache <dir> \
      --report-root <dir-with-<op>/<dsl>_<dtype>.ncu-rep> [--report-root <dir> ...]
  python scripts/paper_figures/nvidia_inventory.py --device GH200 --cache <dir> --report-root <dir>
"""
import argparse, hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HF_REPO = "bcui2/NCU_report"
HF_FOLDER = {"B200": "NVIDIA_B200", "GH200": "NVIDIA_GH200"}
DSLS = ("triton", "cutile", "tilelang")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_listing(device, revision):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HUGGING_FACE") or os.environ.get("HF_TOKEN"))
    rev = revision or api.dataset_info(HF_REPO).sha
    files = [p for p in api.list_repo_files(HF_REPO, repo_type="dataset", revision=rev)
             if p.startswith(HF_FOLDER[device] + "/") and p.endswith(".ncu-rep")]
    meta = {}
    for i in range(0, len(files), 100):
        for f in api.get_paths_info(HF_REPO, files[i:i + 100], repo_type="dataset", revision=rev):
            meta[f.path] = {"size": f.size, "sha256": f.lfs.sha256 if f.lfs else None}
    return rev, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=sorted(HF_FOLDER))
    ap.add_argument("--cache", required=True)
    ap.add_argument("--report-root", action="append", required=True)
    ap.add_argument("--revision", default=None, help="HF dataset revision (default: current head)")
    a = ap.parse_args()
    rev, meta = hf_listing(a.device, a.revision)
    found = {}
    for root in a.report_root:
        for p in Path(root).glob("*/*.ncu-rep"):
            dsl, dtype = p.stem.split("_", 1)
            if dsl not in DSLS:
                continue
            key = f"{HF_FOLDER[a.device]}/{p.parent.name}/{p.name}"
            if key in found:
                sys.exit(f"duplicate report for {key}: {found[key]} and {p}")
            found[key] = str(p)
    with ThreadPoolExecutor(8) as ex:
        hashes = dict(zip(found, ex.map(sha256, found.values())))
    rows = []
    for key in sorted(set(found) | set(meta)):
        op, fname = key.split("/")[1:]
        dsl, dtype = fname[:-len(".ncu-rep")].split("_", 1)
        loc = found.get(key)
        hf = meta.get(key, {})
        rows.append({"hf_path": key, "device": a.device, "operator": op, "dsl": dsl, "dtype": dtype,
                     "local_path": loc, "size": os.path.getsize(loc) if loc else None,
                     "sha256": hashes.get(key), "hf_sha256": hf.get("sha256"), "hf_size": hf.get("size"),
                     "hf_match": bool(loc and hf and hashes.get(key) == hf.get("sha256"))})
    out = {"device": a.device, "hf_repo": HF_REPO, "hf_revision": rev, "reports": rows}
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.cache) / f"inventory_{a.device}.json", "w"), indent=1)
    n = len(rows)
    print(a.device, "rev", rev, "reports", n, "local", sum(1 for r in rows if r["local_path"]),
          "hf", sum(1 for r in rows if r["hf_sha256"]), "match", sum(r["hf_match"] for r in rows))


if __name__ == "__main__":
    main()
