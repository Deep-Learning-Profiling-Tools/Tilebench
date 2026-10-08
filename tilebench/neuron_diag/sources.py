"""Per-operator NKI source manifest and pinned source overlays.

The NKI implementations are not on ``main``: each one lives on an unmerged PR
branch (``bowen/nki/<op>``), and a few of those heads still use the pre-package
layout (``benchmarks/``, ``core/``). Nothing is merged or cherry-picked. For every
operator this module

  * pins the branch head commit (``git rev-parse``) and reads the operator files
    from that commit with ``git show``;
  * records file hashes before and after the minimal import-path compatibility
    patch (``core.`` -> ``tilebench.core.`` etc.);
  * runs a *static* scan of impl_nki.py / impl_torch.py (what the source contains,
    not what the device executes);
  * materializes an overlay tree: this checkout's ``tilebench`` package with the
    operator directory replaced by the pinned files, so the existing XLA
    orchestrator and the native runtime import exactly the same algorithm.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml

from tilebench.paths import PACKAGE_ROOT, REPO_ROOT, list_operators

BRANCH_PREFIX = "origin/bowen/nki/"

# old layout -> package layout (import statements only)
_IMPORT_PATCHES = (
    (re.compile(r"^(\s*)from core\.", re.M), r"\1from tilebench.core."),
    (re.compile(r"^(\s*)import core\.", re.M), r"\1import tilebench.core."),
    (re.compile(r"^(\s*)from benchmarks\.", re.M), r"\1from tilebench.benchmarks."),
    (re.compile(r"^(\s*)from data\.", re.M), r"\1from tilebench.data."),
)

_WRAPPER_PATTERNS = {
    "reshape": r"\.reshape\(",
    "view": r"\.view\(",
    "contiguous": r"\.contiguous\(",
    "pad": r"(?:F|functional)\.pad\(",
    "to()": r"\.to\(",
    "float()/half()/bfloat16()": r"\.(?:float|half|bfloat16)\(\)",
    "clone": r"\.clone\(",
    "cat/stack": r"torch\.(?:cat|stack)\(",
    "permute/transpose": r"\.(?:permute|transpose|t)\(",
    "flatten": r"\.flatten\(",
    "slice": r"[\w\)\]]\[[^\]\n]*:[^\]\n]*\]",
    "item()/cpu()/tolist()": r"\.(?:item|cpu|tolist)\(\)",
}
_XLA_PATTERNS = {
    "import torch_xla": r"^\s*(?:import|from)\s+torch_xla",
    "xm.*": r"\bxm\.",
    "mark_step": r"mark_step\(",
    "torch_xla.sync": r"torch_xla\.sync\(",
    "device.type ==/!= 'xla'": r"device\.type\s*[=!]=\s*[\"']xla[\"']",
}


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout


def _git_bytes(*args: str, cwd: Path = REPO_ROOT) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True).stdout


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def patch_imports(text: str) -> str:
    for pat, rep in _IMPORT_PATCHES:
        text = pat.sub(rep, text)
    return text


def _run_body(text: str) -> str:
    """Source of the module-level ``run`` function (empty when absent)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return ast.get_source_segment(text, node) or ""
    return ""


def _count(patterns: dict, text: str) -> dict:
    return {k: len(re.findall(p, text, re.M)) for k, p in patterns.items()
            if re.search(p, text, re.M)}


def scan_impl_nki(text: str) -> dict:
    """Static facts about an impl_nki.py. Presence in source only: a reshape here is
    not evidence of a device copy."""
    run = _run_body(text)
    kernels = re.findall(r"@nki\.jit\s*\n\s*def\s+(\w+)", text)
    guarded = bool(re.search(r"^if nki is not None:\s*\n(?:[ \t]+.*\n|\s*\n)*?[ \t]+@nki\.jit",
                             text, re.M))
    unguarded_jit = bool(re.search(r"^@nki\.jit", text, re.M))
    launch_vars = {name for name, rhs in re.findall(r"^(\w+)\s*=\s*(.+)$", text, re.M)
                   if any(re.search(rf"\b{k}\b", rhs) for k in kernels)}
    run_calls = [v for v in launch_vars if re.search(rf"\b{v}\(", run)]
    direct = re.findall(r"\b(\w+)\[[^\]]*\]\(", run) + [k for k in kernels if re.search(rf"\b{k}\(", run)]
    default_cfg = re.search(r"^_DEFAULT_CONFIG\s*=\s*(.+)$", text, re.M)
    lnc = {
        "lnc_helper": "_lnc_degree" in text,
        "hardcoded_num_cores": re.findall(r"^NUM_CORES\s*=\s*(\d+)", text, re.M),
        "program_id": "program_id" in text,
        "num_programs": "num_programs" in text,
        "launch_subscripts": sorted(set(re.findall(r"\w+\[(_LNC|_lnc_degree\(\)|NUM_CORES|lnc|\d)\]", text))),
    }
    return {
        "kernels": kernels,
        "jit_guarded_by_nki_none": guarded,
        "module_level_unguarded_jit": unguarded_jit,
        "launch_sites_in_run_static": sorted(set(run_calls) | set(direct)),
        "loop_in_run": bool(re.search(r"^\s+for\s", run, re.M)),
        "device_loops": sorted(set(re.findall(r"nl\.(dynamic_range|fori_loop|while_loop)", text))),
        "xla_dependencies": _count(_XLA_PATTERNS, text),
        "wrapper_ops_in_run_static": _count(_WRAPPER_PATTERNS, run),
        "lnc": lnc,
        "default_config": default_cfg.group(1).strip() if default_cfg else None,
        "autotuner": {"imports_NkiAutotuner": "NkiAutotuner" in text,
                      "module_level_construction": bool(re.search(r"^_tuner\w*\s*=\s*NkiAutotuner", text, re.M)),
                      "lazy_holder": "_tuner_holder" in text},
        "declared_unsupported": re.findall(r"NotImplementedError\(\s*\n?\s*f?[\"']([^\"']{0,120})", text),
    }


def scan_impl_torch(text: str) -> dict:
    return {"xla_dependencies": _count(_XLA_PATTERNS, text),
            "uses_torch_compile": "torch.compile" in text}


def _generator_name(operator: str) -> str | None:
    from tilebench.data.tensors import GENERATORS

    fn = GENERATORS.get(operator)
    return getattr(fn, "__name__", None)


def _generator_source(tensors_text: str, operator: str) -> str | None:
    """Source of the generator function that main registers for `operator`."""
    want = _generator_name(operator)
    if want is None:
        return None
    try:
        tree = ast.parse(tensors_text)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == want:
            return ast.get_source_segment(tensors_text, node)
    return None


def _branch_layout(ref: str, operator: str) -> tuple[str, str] | None:
    names = _git("ls-tree", "-r", "--name-only", ref).splitlines()
    for prefix in ("tilebench/benchmarks/operators", "benchmarks/operators"):
        if f"{prefix}/{operator}/impl_nki.py" in names:
            return prefix, ("package" if prefix.startswith("tilebench/") else "old")
    return None


def _branch_op_files(ref: str, prefix: str, operator: str) -> list[str]:
    return sorted(n for n in _git("ls-tree", "-r", "--name-only", ref,
                                  f"{prefix}/{operator}/").splitlines()
                  if not n.endswith((".pyc",)) and "__pycache__" not in n)


def _cross_op_imports(text: str, operator: str) -> list[str]:
    return sorted({m for m in re.findall(r"benchmarks\.operators\.(\w+)", text) if m != operator})


def _dep_source(sha: str, prefix: str, dep: str) -> dict:
    """Where a cross-operator dependency's files come from: the importing branch when
    it carries the dependency's impl_nki.py, else the dependency's own pinned PR head."""
    names = set(_git("ls-tree", "-r", "--name-only", sha).splitlines())
    if f"{prefix}/{dep}/impl_nki.py" in names:
        return {"commit": sha, "prefix": prefix, "from": "importing branch"}
    ref = BRANCH_PREFIX + dep
    try:
        dep_sha = _git("rev-parse", ref).strip()
    except subprocess.CalledProcessError:
        return {"commit": None, "prefix": None, "from": f"missing: no {ref}"}
    lay = _branch_layout(dep_sha, dep)
    if lay is None:
        return {"commit": None, "prefix": None, "from": f"missing: {ref} has no impl_nki.py"}
    return {"commit": dep_sha, "prefix": lay[0], "from": ref.replace("origin/", "")}


def build_entry(operator: str, prs: dict | None = None, ref: str | None = None) -> dict:
    """Manifest entry for one operator (reads git only; writes nothing). ``ref`` defaults to
    the PR head ``origin/bowen/nki/<operator>``; a local branch or commit may be given."""
    ref = ref or BRANCH_PREFIX + operator
    main_dir = PACKAGE_ROOT / "benchmarks" / "operators" / operator
    entry: dict = {"operator": operator, "branch": ref.replace("origin/", ""),
                   "pr": (prs or {}).get(ref.replace("origin/", "")), "status": None}
    try:
        sha = _git("rev-parse", ref).strip()
    except subprocess.CalledProcessError:
        entry.update(status="no_nki_impl", reason=f"no ref {ref}")
        return entry
    entry["commit"] = sha
    lay = _branch_layout(sha, operator)
    if lay is None:
        entry.update(status="no_nki_impl", reason=f"{ref} has no impl_nki.py for {operator}")
        return entry
    prefix, layout = lay
    entry["layout"] = layout
    files = {}
    for path in _branch_op_files(sha, prefix, operator):
        raw = _git_bytes("show", f"{sha}:{path}")
        rel = path[len(prefix) + len(operator) + 2:]
        rec = {"branch_path": path, "sha256": sha256_bytes(raw)}
        if path.endswith(".py"):
            patched = patch_imports(raw.decode())
            if patched != raw.decode():
                rec["patched_sha256"] = sha256_bytes(patched.encode())
        files[rel] = rec
    entry["files"] = files
    nki_text = patch_imports(_git("show", f"{sha}:{prefix}/{operator}/impl_nki.py"))
    torch_text = _git("show", f"{sha}:{prefix}/{operator}/impl_torch.py") if "impl_torch.py" in files else ""
    cfg_text = _git("show", f"{sha}:{prefix}/{operator}/config.yaml") if "config.yaml" in files else ""
    entry["impl_nki"] = scan_impl_nki(nki_text)
    entry["impl_torch"] = scan_impl_torch(torch_text)
    entry["cross_operator_imports"] = _cross_op_imports(nki_text, operator)
    entry["cross_operator_sources"] = {dep: _dep_source(sha, prefix, dep)
                                       for dep in entry["cross_operator_imports"]}
    cfg = yaml.safe_load(cfg_text) if cfg_text else {}
    entry["config"] = {"benchmark": cfg.get("benchmark"), "verify": cfg.get("verify"),
                       "case_grid": cfg.get("case_grid"), "case_defaults": cfg.get("case_defaults")}
    main_cfg = main_dir / "config.yaml"
    entry["config_differs_from_main"] = (main_cfg.is_file()
                                         and yaml.safe_load(main_cfg.read_text()) != cfg)
    main_torch = main_dir / "impl_torch.py"
    entry["impl_torch_differs_from_main"] = bool(main_torch.is_file()
                                                 and main_torch.read_text() != torch_text)
    tensors_path = "tilebench/data/tensors.py" if layout == "package" else "data/tensors.py"
    try:
        branch_tensors = _git("show", f"{sha}:{tensors_path}")
    except subprocess.CalledProcessError:
        branch_tensors = ""
    main_tensors = (PACKAGE_ROOT / "data" / "tensors.py").read_text()
    gb, gm = _generator_source(branch_tensors, operator), _generator_source(main_tensors, operator)
    entry["generator"] = {
        "used": "main tilebench/data/tensors.py",
        "function": _generator_name(operator),
        "main_sha256": sha256_bytes(main_tensors.encode()),
        "function_found_by_name": gm is not None,
        "branch_function_differs": (gb != gm) if (gb is not None and gm is not None) else None,
    }
    entry["status"] = "found"
    return entry


def framework_hash() -> str:
    """Hash of the framework modules the overlay shares with this checkout."""
    h = hashlib.sha256()
    for sub in ("core", "data", "neuron_diag"):
        for p in sorted((PACKAGE_ROOT / sub).rglob("*.py")):
            h.update(str(p.relative_to(PACKAGE_ROOT)).encode())
            h.update(p.read_bytes())
    for p in ("paths.py", "backends.py", "__init__.py"):
        h.update(p.encode())
        h.update((PACKAGE_ROOT / p).read_bytes())
    return h.hexdigest()


def source_hash(entry: dict) -> str:
    """Identity of the code a measurement ran: pinned operator files (after the
    import patch), cross-operator dependencies and the shared framework."""
    h = hashlib.sha256()
    h.update(entry.get("commit", "").encode())
    for rel, rec in sorted((entry.get("files") or {}).items()):
        h.update(rel.encode())
        h.update(rec.get("patched_sha256", rec["sha256"]).encode())
    for dep in entry.get("cross_operator_imports", []):
        src = (entry.get("cross_operator_sources") or {}).get(dep) or {}
        h.update(f"dep:{dep}:{src.get('commit')}".encode())
    h.update(framework_hash().encode())
    return h.hexdigest()


def build_manifest(prs: dict | None = None) -> dict:
    ops = list_operators()
    entries = [build_entry(op, prs) for op in ops]
    for e in entries:
        if e["status"] == "found":
            e["source_hash"] = source_hash(e)
    return {"main_commit": _git("rev-parse", "HEAD").strip(),
            "framework_sha256": framework_hash(), "operators": entries}


def materialize_overlay(entry: dict, overlays_root: Path) -> Path:
    """Directory whose ``tilebench`` package is this checkout's framework plus the
    operator's pinned files. Reused when it already exists for the same source hash;
    never modified afterwards."""
    op = entry["operator"]
    dest = Path(overlays_root) / f"{op}-{entry['source_hash'][:16]}"
    marker = dest / "OVERLAY.json"
    if marker.is_file():
        return dest
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(PACKAGE_ROOT, tmp / "tilebench",
                    ignore=shutil.ignore_patterns("__pycache__", "llm_generated", "*.pyc"))
    sha = entry["commit"]
    prefix = "tilebench/benchmarks/operators" if entry["layout"] == "package" else "benchmarks/operators"
    deps = [(op, sha, prefix)]
    for dep in entry.get("cross_operator_imports", []):
        src = (entry.get("cross_operator_sources") or {}).get(dep) or {}
        if src.get("commit"):
            deps.append((dep, src["commit"], src["prefix"]))
    for dep, dsha, dprefix in deps:
        dst_dir = tmp / "tilebench" / "benchmarks" / "operators" / dep
        for path in _branch_op_files(dsha, dprefix, dep):
            raw = _git_bytes("show", f"{dsha}:{path}")
            if path.endswith(".py"):
                raw = patch_imports(raw.decode()).encode()
            out = dst_dir / path[len(dprefix) + len(dep) + 2:]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
    (tmp / "OVERLAY.json").write_text(json.dumps(
        {"operator": op, "commit": sha, "source_hash": entry["source_hash"],
         "framework_sha256": framework_hash()}, indent=1))
    tmp.rename(dest)
    return dest


# ---------------------------------------------------------------- runtime compatibility patches
# Minimal edits that let an unchanged NKI implementation run on the native stack. They may only
# touch framework calls (e.g. XLA graph cuts), never the algorithm, decomposition or tuning.
RUNTIME_COMPAT_PATCHES = {
    "radix_sort": {
        "file": "impl_nki.py",
        "kind": "runtime_compatibility",
        "description": ("run() cuts the XLA graph after every radix pass with xm.mark_step(); the "
                        "native 'neuron' device has no lazy graph and no torch_xla. The cut is kept "
                        "for XLA tensors and skipped for any other device. Radix width, passes, "
                        "kernel and tuning are unchanged."),
        "edits": [
            ("def _mark_step() -> None:\n"
             "    from torch_xla.core import xla_model as xm\n"
             "    xm.mark_step()\n",
             "def _mark_step(device=None) -> None:\n"
             "    # runtime compatibility patch (neuron diagnostics): XLA keeps its graph cut; the\n"
             "    # native neuron device executes eagerly and has no torch_xla.\n"
             "    if device is not None and device.type != \"xla\":\n"
             "        return\n"
             "    from torch_xla.core import xla_model as xm\n"
             "    xm.mark_step()\n"),
            ("        work = radix_pass(work, shift_t, S, n_blocks)\n        _mark_step()\n",
             "        work = radix_pass(work, shift_t, S, n_blocks)\n        _mark_step(device)\n"),
        ],
    },
}


def apply_runtime_compat_patch(entry: dict, overlays_root: Path) -> dict:
    """New manifest entry for ``entry`` with its declared runtime compatibility patch applied
    to a copy of its overlay. The new source hash is derived from the old one and the patch,
    so every other operator keeps its hash."""
    op = entry["operator"]
    spec = RUNTIME_COMPAT_PATCHES[op]
    src_overlay = Path(entry["overlay"])
    rel = Path("tilebench") / "benchmarks" / "operators" / op / spec["file"]
    before = (src_overlay / rel).read_text()
    after = before
    for old, new in spec["edits"]:
        if after.count(old) != 1:
            raise ValueError(f"{op}: patch anchor not found exactly once: {old[:60]!r}")
        after = after.replace(old, new)
    patch_sha = sha256_bytes(json.dumps(spec["edits"]).encode())
    new_hash = sha256_bytes((entry["source_hash"] + patch_sha).encode())
    dest = Path(overlays_root) / f"{op}-{new_hash[:16]}"
    if not (dest / "OVERLAY.json").is_file():
        tmp = dest.with_name(dest.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src_overlay, tmp, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (tmp / rel).write_text(after)
        meta = json.loads((tmp / "OVERLAY.json").read_text())
        meta.update(source_hash=new_hash, base_source_hash=entry["source_hash"],
                    runtime_compat_patch_sha256=patch_sha)
        (tmp / "OVERLAY.json").write_text(json.dumps(meta, indent=1))
        tmp.rename(dest)
    out = dict(entry)
    out.update(source_hash=new_hash, overlay=str(dest), base_source_hash=entry["source_hash"],
               runtime_compat_patch={"kind": spec["kind"], "file": spec["file"],
                                     "description": spec["description"], "sha256": patch_sha,
                                     "patched_file_sha256": sha256_bytes(after.encode()),
                                     "base_file_sha256": sha256_bytes(before.encode()),
                                     "algorithm_change": False})
    return out
