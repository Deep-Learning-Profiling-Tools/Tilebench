# Discover Released Evidence Before Reprofiling

Dataset: [bcui2/NCU_report](https://huggingface.co/datasets/bcui2/NCU_report/tree/main).
It contains NVIDIA reports and AMD artifacts, despite its NCU name. Discover the
current selected revision; do not assume every platform/backend/dtype is present.
Do not fetch the entire dataset to analyze one operator.

## Saved-Evidence Workflow

1. Resolve hardware/operator/backend/dtype and the requested shape/mode using the
   repository map. Prefer a supplied report or matching local artifact.
2. If absent locally and remote reads/downloads are permitted, list just the
   relevant hardware/operator directory on the Hub. Use an existing authenticated
   client; do not print credentials or install a client without authorization.
3. Identify the exact saved artifact. NVIDIA uses
   `NVIDIA_<hardware>/<op>/<backend>_<dtype>.ncu-rep`. MI300X uses
   `AMD_MI300X/<op>/<backend>_<dtype>/` with ROCm workloads, analysis, and capture
   metadata. A directory is not an NCU report.
4. Pin the resolved dataset commit and download only the needed file(s) into the
   task's output/cache location. Record dataset ID, commit, remote path, local
   path, and file checksum. A moving `main` reference is not durable provenance.
5. Verify case/config/source identity and kernel coverage using associated metadata
   and report attributes. The NVIDIA filename encodes only backend and dtype: it
   does not establish shape, source version, configuration, replay, or cache policy.
   If these remain unknown, state the limits rather than inventing them or treating
   current catalogue entries as proof of an old capture's identity.
6. Extract and analyze the saved evidence. Only propose new collection if the
   requested case is absent/mismatched or a material signal cannot be recovered.
   Explain what is missing and obtain authorization before using collection tools.

## Bundled Download and Extraction Tool

Use `scripts/hf_ncu_report.py` instead of writing an ad hoc downloader. It fetches
exactly one NVIDIA report, resolves the requested revision to a commit, verifies
size and the remote SHA-256 when provided, and writes `download.json`. It reuses
the existing Hugging Face login; tokens are never embedded in commands or outputs.
One case/revision belongs in each output directory; conflicting reuse is refused.

```bash
<python> <skill-dir>/scripts/hf_ncu_report.py B200 gaussian_blur tilelang fp16 --output-dir <output>/blur --extract
```

`--extract` also writes `ncu.json` with the same exact metrics/action inventory as
`ncu_extract.py`. Repeated `--metric <exact-name>` selects counters and implies
extraction. Without those flags it only downloads and records provenance, so the
fetch environment need not contain NCU. `--revision <commit-or-ref>` selects a
specific dataset revision. The fetcher supports B200/GH200, not AMD directories.

Dependencies are an available `huggingface_hub` client, plus the NCU Python API
when extracting. Neither needs GPU access or installed DSL compilers. Do not
install either automatically or change the workspace's benchmark stack. If tools
are unavailable, identify that tooling gap rather than suggesting a new profile.
The download receipt verifies file provenance, not the captured shape/config/source.

## Client Pattern for Other Artifact Discovery

Run with the approved interpreter/client environment. Adapt the example to the
resolved hardware/operator/backend/dtype; output location is task-supplied.
The example lists one directory, pins its revision, and downloads one report.
It is a fallback/discovery example; prefer the bundled tool for NVIDIA downloads.

```python
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()  # Uses the user's existing login; never embed a token.
repo = "bcui2/NCU_report"
revision = api.repo_info(repo, repo_type="dataset", revision="main").sha
directory = "NVIDIA_B200/gaussian_blur"
entries = list(api.list_repo_tree(repo, repo_type="dataset",
                                revision=revision, path_in_repo=directory))
path = directory + "/tilelang_fp16.ncu-rep"
if path not in {entry.path for entry in entries}:
    raise RuntimeError("Requested report is not available at this revision")
local_path = hf_hub_download(repo, path, repo_type="dataset",
                             revision=revision, local_dir=output_directory)
```

`output_directory` must be set to an authorized task location. The download alone
does not validate the capture's case identity. NVIDIA saved-report analysis needs
compatible NCU import tools, not GPU access or installed DSL compilers. AMD parsing
capabilities depend on the supplied formats; this skill currently only locates
those artifacts and does not require installing ROCm to navigate them.
