#!/usr/bin/env python3
"""Export the published profiling reports into compact JSON the dashboard agent can query.

Source: the Hugging Face dataset bcui2/NCU_report (gated; uses the locally stored HF token).
  NVIDIA_B200, NVIDIA_GH200   <op>/<backend>_<dtype>.ncu-rep      needs the `ncu` CLI
  AMD_MI300X                  <op>/<backend>_<dtype>/analysis/…   rocprof-compute CSVs

Output: data/profiles_json/<PLATFORM>/<op>/<backend>_<dtype>.json.gz and manifest.json.
Each report is downloaded, converted and deleted again, so the raw 30 GB never sits on disk.

    python scripts/export_profiles.py                    # everything not exported yet
    python scripts/export_profiles.py --platform GH200 --op vector_add
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

DASHBOARD = Path(__file__).resolve().parents[1]
OUT = DASHBOARD / "data" / "profiles_json"
DATASET = "bcui2/NCU_report"
API = f"https://huggingface.co/api/datasets/{DATASET}"
NVIDIA = {"B200": "NVIDIA_B200", "GH200": "NVIDIA_GH200"}
AMD = {"MI300X": "AMD_MI300X"}
NCU = "ncu"
MAX_KERNELS = 8
MAX_HOT = 40
# Launch bookkeeping that is not a metric.
NCU_IDENTITY = ("ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context",
                "Stream", "Block Size", "Grid Size", "Device", "CC")
BACKENDS = ("triton", "cutile", "tilelang", "torch")
csv.field_size_limit(1 << 30)


def number(text: str):
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text.replace(",", ""))
    except ValueError:
        return text
    return int(value) if value.is_integer() and abs(value) < 1e15 else value


def split_name(stem: str) -> tuple[str, str]:
    for backend in BACKENDS:
        if stem.startswith(backend + "_"):
            return backend, stem[len(backend) + 1:]
    raise ValueError(f"no backend prefix in {stem}")


def listing(hf_path: str) -> list[dict]:
    entries: list[dict] = []
    url: str | None = f"{API}/tree/main/{hf_path}?recursive=true&limit=1000"
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            entries += [e for e in json.load(response) if e["type"] == "file"]
            link = response.headers.get("Link") or ""
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
    return entries


def fetch(path: str, into: Path) -> Path:
    return Path(hf_hub_download(DATASET, path, repo_type="dataset", local_dir=into))


def ncu_page(report: Path, page: str) -> list[list[str]]:
    done = subprocess.run([NCU, "--import", str(report), "--page", page, "--csv"],
                          capture_output=True, text=True, timeout=900)
    if done.returncode != 0:
        raise RuntimeError(f"ncu --page {page} failed: {done.stderr[-300:]}")
    return list(csv.reader(io.StringIO(done.stdout)))


def ncu_hot(report: Path) -> dict[str, list[dict]]:
    """Kernel -> hottest SASS instructions, when this ncu can disassemble the report."""
    hot: dict[str, list[dict]] = {}
    kernel, header, rows = None, None, []

    def flush() -> None:
        if not kernel or not header or "# Samples" not in header:
            return
        col = {name: i for i, name in enumerate(header)}
        keep = []
        for row in rows:
            if len(row) <= col["# Samples"] or row[col["Source"]].strip() in ("", "???"):
                continue
            samples = number(row[col["# Samples"]])
            if not isinstance(samples, (int, float)) or samples <= 0:
                continue
            item = {"instruction": row[col["Source"]].strip(), "samples": samples}
            for field, key in (("Warp Stall Sampling (Not-issued Samples)", "not_issued_samples"),
                               ("Instructions Executed", "instructions_executed")):
                if field in col:
                    item[key] = number(row[col[field]])
            keep.append(item)
        if keep:
            total = sum(k["samples"] for k in keep)
            keep.sort(key=lambda k: -k["samples"])
            hot.setdefault(kernel, [{**k, "pct_of_samples": round(100 * k["samples"] / total, 2)}
                                    for k in keep[:MAX_HOT]])

    for row in ncu_page(report, "source"):
        if len(row) >= 2 and row[0] == "Kernel Name":
            flush()
            kernel, header, rows = row[1], None, []
        elif row and row[0] == "Address":
            header = row
        elif header:
            rows.append(row)
    flush()
    return hot


def export_nvidia(platform: str, entry: dict, tmp: Path) -> dict:
    _, op, filename = entry["path"].split("/")
    backend, dtype = split_name(filename.removesuffix(".ncu-rep"))
    report = fetch(entry["path"], tmp)
    try:
        raw = ncu_page(report, "raw")
        details = ncu_page(report, "details")
        try:
            hot = ncu_hot(report)
        except Exception:
            hot = {}
    finally:
        report.unlink(missing_ok=True)

    header, units, launches = raw[0], raw[1], raw[2:]
    col = {name: i for i, name in enumerate(header)}
    rules: dict[str, list[dict]] = {}
    sections: dict[str, dict[str, dict]] = {}
    if details:
        d = {name: i for i, name in enumerate(details[0])}

        def cell(row: list[str], name: str) -> str:
            # Metric rows are shorter than rule rows, and some captures have no rule columns.
            i = d.get(name)
            return row[i] if i is not None and i < len(row) else ""

        for row in details[1:]:
            # The curated, human-named metrics the Nsight Compute UI shows per section.
            if cell(row, "Metric Name"):
                value = number(cell(row, "Metric Value"))
                if value is not None:
                    unit = cell(row, "Metric Unit")
                    sections.setdefault(cell(row, "ID"), {}).setdefault(
                        cell(row, "Section Name"), {})[cell(row, "Metric Name")] = (
                            [value, unit] if unit else [value])
            if not cell(row, "Rule Name"):
                continue
            rule = {"section": cell(row, "Section Name"), "rule": cell(row, "Rule Name"),
                    "type": cell(row, "Rule Type"), "description": cell(row, "Rule Description")}
            if cell(row, "Estimated Speedup"):
                rule["estimated_speedup_pct"] = number(cell(row, "Estimated Speedup"))
                rule["estimated_speedup_type"] = cell(row, "Estimated Speedup Type")
            rules.setdefault(cell(row, "ID"), []).append(rule)

    # One entry per distinct kernel; repeated launches of the same kernel are counted, not kept.
    kernels: list[dict] = []
    seen: dict[str, dict] = {}
    for row in launches:
        name = row[col["Kernel Name"]]
        if name in seen:
            seen[name]["launches"] += 1
            continue
        metrics = {}
        for i, key in enumerate(header):
            if key in NCU_IDENTITY or i >= len(row):
                continue
            value = number(row[i])
            if value is not None:
                metrics[key] = [value, units[i]] if units[i] else [value]
        kernel = {"name": name, "launches": 1, "block_size": row[col["Block Size"]],
                  "grid_size": row[col["Grid Size"]], "compute_capability": row[col["CC"]],
                  "sections": sections.get(row[col["ID"]], {}),
                  "metrics": metrics, "findings": rules.get(row[col["ID"]], []),
                  "hot_instructions": hot.get(name, [])}
        seen[name] = kernel
        if len(kernels) < MAX_KERNELS:
            kernels.append(kernel)
    return {"platform": platform, "op": op, "backend": backend, "dtype": dtype,
            "profiler": "Nsight Compute", "source_path": entry["path"],
            "source_bytes": entry["size"], "distinct_kernels": len(seen), "kernels": kernels}


def export_amd(platform: str, run: str, tmp: Path) -> dict:
    _, op, name = run.split("/")
    backend, dtype = split_name(name)
    files = {key: fetch(f"{run}/{rel}", tmp) for key, rel in (
        ("capture", "capture.json"),
        ("kernel", "analysis/workload_csv/kernel.csv"),
        ("metric", "analysis/workload_csv/kernel_metric.csv"),
        ("hot", "analysis/pc_sampling_instructions.csv"))}
    capture = json.loads(files["capture"].read_text())
    kernels: dict[str, dict] = {}
    with files["kernel"].open(newline="") as handle:
        for row in csv.DictReader(handle):
            kernels[row["kernel_name"]] = {
                "name": row["kernel_name"], "launches": number(row["dispatch_count"]),
                "duration_ns_mean": number(row["duration_ns_mean"]),
                "metrics": {}, "findings": [], "hot_instructions": []}
    descriptions: dict[str, str] = {}
    with files["metric"].open(newline="") as handle:
        for row in csv.DictReader(handle):
            kernel = kernels.get(row["kernel_name"])
            value = number(row["value"])
            if kernel is None or value is None:
                continue
            base = " / ".join(dict.fromkeys(
                p for p in (row["table_name"], row["sub_table_name"], row["metric_name"]) if p))
            key = f"{base} [{row['value_name']}]"
            kernel["metrics"][key] = [value, row["unit"]] if row["unit"] else [value]
            if row["description"]:
                descriptions[base] = row["description"]
    with files["hot"].open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["operator_kernel"] == "True"]
    totals: dict[str, int] = {}
    for row in rows:
        totals[row["kernel_name"]] = totals.get(row["kernel_name"], 0) + int(row["samples"])
    for row in sorted(rows, key=lambda r: -int(r["samples"])):
        kernel = kernels.get(row["kernel_name"])
        if kernel is None or len(kernel["hot_instructions"]) >= MAX_HOT:
            continue
        line = re.sub(r"^.*/operators/", "", row["source_line"])
        kernel["hot_instructions"].append({
            "instruction": row["instruction"], "source_line": line,
            "samples": int(row["samples"]), "stalled": int(row["stalled"]),
            "pct_of_samples": round(100 * int(row["samples"]) / totals[row["kernel_name"]], 2),
            "stall_reasons": json.loads(row["stall_reasons"] or "{}")})
    shutil.rmtree(tmp / run, ignore_errors=True)
    return {"platform": platform, "op": op, "backend": backend, "dtype": dtype,
            "profiler": "rocprof-compute", "source_path": run,
            "capture": {"params": capture.get("params"), "autotune_winner": capture.get("winner")},
            "distinct_kernels": len(kernels), "kernels": list(kernels.values())[:MAX_KERNELS],
            "_descriptions": descriptions}


def write(report: dict) -> dict:
    target = OUT / report["platform"] / report["op"] / f"{report['backend']}_{report['dtype']}.json.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(report, separators=(",", ":")).encode()
    with gzip.GzipFile(target, "wb", mtime=0) as handle:
        handle.write(body)
    return {"platform": report["platform"], "op": report["op"], "backend": report["backend"],
            "dtype": report["dtype"], "profiler": report["profiler"],
            "file": str(target.relative_to(OUT)), "bytes": target.stat().st_size,
            "distinct_kernels": report["distinct_kernels"],
            "kernels": [{"name": k["name"], "metrics": len(k["metrics"]),
                         "section_metrics": sum(len(v) for v in k.get("sections", {}).values()),
                         "findings": len(k["findings"]),
                         "hot_instructions": len(k["hot_instructions"])}
                        for k in report["kernels"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--platform", action="append", choices=[*NVIDIA, *AMD])
    parser.add_argument("--op", action="append")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ncu", default="ncu", help="ncu binary; B200 reports need 2025+ for units and SASS")
    parser.add_argument("--force", action="store_true", help="re-export reports already present")
    args = parser.parse_args()
    global NCU
    NCU = args.ncu
    platforms = args.platform or [*NVIDIA, *AMD]

    jobs = []  # (platform, kind, key, entry)
    for platform in platforms:
        if platform in NVIDIA:
            for entry in listing(NVIDIA[platform]):
                if entry["path"].endswith(".ncu-rep"):
                    jobs.append((platform, "nvidia", entry["path"], entry))
        else:
            runs = sorted({"/".join(e["path"].split("/")[:3]) for e in listing(AMD[platform])
                           if e["path"].endswith("/capture.json")})
            jobs += [(platform, "amd", run, None) for run in runs]
    if args.op:
        jobs = [j for j in jobs if j[2].split("/")[1] in args.op]

    manifest_path = OUT / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = {(r["platform"], r["op"], r["backend"], r["dtype"]): r
                    for r in json.loads(manifest_path.read_text())["reports"]}
    descriptions: dict[str, dict[str, str]] = {}
    desc_path = OUT / "metric_descriptions.json"
    if desc_path.exists():
        descriptions = json.loads(desc_path.read_text())

    def key_of(job) -> tuple:
        _, op, name = job[2].split("/")[:3]
        return (job[0], op, *split_name(name.removesuffix(".ncu-rep")))

    todo = [j for j in jobs if args.force or key_of(j) not in manifest]
    print(f"{len(jobs)} reports selected, {len(todo)} to export", flush=True)
    failures = []

    def run(job):
        platform, kind, key, entry = job
        with tempfile.TemporaryDirectory(prefix="profexp_") as tmp:
            if kind == "nvidia":
                return export_nvidia(platform, entry, Path(tmp))
            return export_amd(platform, key, Path(tmp))

    def save() -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(
            {"dataset": DATASET, "reports": [manifest[k] for k in sorted(manifest)]}, indent=1) + "\n")
        desc_path.write_text(json.dumps(descriptions, indent=1, sort_keys=True) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job): job for job in todo}
        for n, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                report = future.result()
            except Exception as error:  # keep going; report at the end
                failures.append((job[2], f"{type(error).__name__}: {error}"[:300]))
                print(f"[{n}/{len(todo)}] FAILED {job[2]}: {failures[-1][1]}", flush=True)
                continue
            descriptions.setdefault(report["platform"], {}).update(report.pop("_descriptions", {}))
            manifest[key_of(job)] = write(report)
            if n % 20 == 0 or n == len(todo):
                save()
                print(f"[{n}/{len(todo)}] exported", flush=True)
    save()
    print(f"done: {len(manifest)} reports in manifest, {len(failures)} failures", flush=True)
    for path, error in failures:
        print(f"  {path}: {error}")


if __name__ == "__main__":
    main()
