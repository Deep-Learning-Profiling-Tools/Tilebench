"""Build introspection_evidence.json for SKILL.md (TileLang 0.1.11 reference).

For every `T.*` / `tilelang.*` name mentioned in SKILL.md (sections 1-18) this script
records: whether it resolves in the installed package, the module that defines it,
its `inspect.signature` (or why it is not introspectable), the installed source
file:line, and the corresponding file:line in the v0.1.11 checkout (with a check that
the checkout line matches the installed line). Names listed in section 19
("Names That Do Not Exist") are verified to be ABSENT.

Run:  env -u LD_LIBRARY_PATH <tilebench_env python> introspect.py
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import inspect
import json
import os
import platform
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.join(HERE, "SKILL.md")
CHECKOUT = os.path.join(os.path.dirname(HERE), "tilelang_src")
OUT = os.path.join(HERE, "introspection_evidence.json")

import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import tilelang.autotuner  # noqa: E402,F401
import tilelang.layout  # noqa: E402,F401
import tilelang.jit  # noqa: E402,F401

SITE_TILELANG = os.path.dirname(os.path.abspath(tilelang.__file__))  # .../site-packages/tilelang

# Names written in SKILL.md without a `T.` / `tilelang.` prefix (methods, enum members).
EXTRA_PRESENT = [
    "T.GemmWarpPolicy.FullRow",
    "T.GemmWarpPolicy.FullCol",
    "tilelang.PassConfigKey.TL_ENABLE_FAST_MATH",
    "tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS",
    "tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER",
    "tilelang.jit.JITKernel.__call__",
    "tilelang.jit.JITKernel.get_kernel_source",
    "tilelang.jit.JITKernel.get_host_source",
    "tilelang.jit.JITImpl.compile",
    "tilelang.jit.JITImpl.get_kernel_source",
    "tilelang.jit.JITImpl.__call__",
    "tilelang.language.eager.builder.JITFunc",
    "tilelang.jit.adapter.tvm_ffi.TVMFFIKernelAdapter",
    "tilelang.jit.adapter.base.BaseKernelAdapter.get_current_stream_functor",
    "tilelang.jit.adapter.base.BaseKernelAdapter.get_current_device_functor",
    "tilelang.jit.execution_backend.resolve_execution_backend",
    "tilelang.env.Environment",
]

NAME_RE = re.compile(r"(?<![\w.])((?:T|tilelang)(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?![\w*])")


def split_sections(text: str) -> tuple[str, str, str]:
    """Return (sections 1-18, section-19 absent-name cells, section-19 remaining text)."""
    idx = text.index("## 19. Names That Do Not Exist")
    present_text, sec19 = text[:idx], text[idx:]
    absent_cells, rest = [], []
    for line in sec19.splitlines():
        if line.startswith("| `"):
            cells = line.split("|")
            absent_cells.append(cells[1])
            rest.append("|".join(cells[2:]))
        else:
            rest.append(line)
    return present_text, "\n".join(absent_cells), "\n".join(rest)


def collect(text: str) -> list[str]:
    names = set(NAME_RE.findall(text))
    # drop attribute access on results that is not part of the API path (none expected), keep sorted
    return sorted(names)


def resolve(path: str):
    """Resolve a dotted name: first by attribute access from the root (T = tilelang.language),
    then, if that fails, by importing the longest module prefix and getattr on the rest
    (needed where an attribute shadows a submodule, e.g. the function `tilelang.jit`
    vs the module `tilelang.jit`)."""
    parts = path.split(".")
    root_name = "tilelang.language" if parts[0] == "T" else "tilelang"
    root = T if parts[0] == "T" else tilelang
    obj = root
    try:
        for p in parts[1:]:
            obj = getattr(obj, p)
        return obj, root_name
    except AttributeError:
        pass
    full = [root_name] + parts[1:]
    for i in range(len(full), 0, -1):
        modname = ".".join(full[:i])
        try:
            obj = importlib.import_module(modname)
        except Exception:
            continue
        try:
            for p in full[i:]:
                obj = getattr(obj, p)
            return obj, modname
        except AttributeError:
            continue
    raise AttributeError(f"{path}: cannot be resolved")


def installed_to_repo(path: str) -> str | None:
    path = os.path.abspath(path)
    if path.startswith(SITE_TILELANG + os.sep):
        rel = os.path.relpath(path, SITE_TILELANG)
        if rel.startswith("3rdparty" + os.sep):
            return rel  # submodule path inside the repo (3rdparty/tvm/...)
        return os.path.join("tilelang", rel)
    return None


def checkout_line(rel: str, lineno: int) -> str | None:
    p = os.path.join(CHECKOUT, rel)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].rstrip("\n")
    return None


def grep_definition(name: str, rel_hint_files: list[str]) -> tuple[str, int, str] | None:
    """Find the line binding `name` in the checkout. Assignments are preferred over
    `def`/`class` lines, and one-line typing stubs (`class X(...): ...`) are skipped."""
    pats = [
        re.compile(rf"^\s*{re.escape(name)}\s*=\s*"),
        re.compile(rf"^\s*{re.escape(name)}\s*:\s*[^=]+=\s*"),
        re.compile(rf"^\s*(def|class)\s+{re.escape(name)}\b"),
    ]
    files = []
    for rel in rel_hint_files:
        p = os.path.join(CHECKOUT, rel)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                files.append((rel, f.readlines()))
    for pt in pats:
        for rel, lines in files:
            for i, line in enumerate(lines, 1):
                if pt.search(line) and not line.rstrip().endswith(": ..."):
                    return rel, i, line.rstrip("\n")
    return None


def binding_candidates(obj, leaf: str) -> list[str]:
    """Modules (tilelang.* only) whose namespace binds `leaf` to exactly this object."""
    out = []
    for name, mod in sorted(sys.modules.items()):
        if mod is None or not name.startswith("tilelang"):
            continue
        try:
            if vars(mod).get(leaf, None) is obj:
                out.append(name)
        except Exception:
            continue
    return out


def module_to_files(modname: str) -> list[str]:
    base = modname.replace(".", os.sep)
    return [base + ".py", os.path.join(base, "__init__.py")]


def signature_of(obj) -> str:
    try:
        if inspect.isclass(obj) or inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj):
            return str(inspect.signature(obj))
        if callable(obj) and not isinstance(obj, str):
            return "__call__" + str(inspect.signature(type(obj).__call__))
    except (TypeError, ValueError) as e:
        return f"not introspectable: {type(e).__name__}: {e}"
    if isinstance(obj, str):
        return f"not introspectable: value object of type {type(obj).__module__}.{type(obj).__qualname__} (value {str(obj)!r})"
    return f"not introspectable: non-callable object of type {type(obj).__module__}.{type(obj).__qualname__}"


def defining_module(obj, path: str) -> str:
    target = inspect.unwrap(obj) if callable(obj) else obj
    m = getattr(target, "__module__", None)
    if isinstance(m, str) and not isinstance(obj, str) and not inspect.ismodule(obj):
        if inspect.isclass(obj) or inspect.isfunction(target) or inspect.ismethod(target) or inspect.isbuiltin(target):
            return m
    if inspect.ismodule(obj):
        return obj.__name__
    # instances (dtype objects, proxies, enum members): find the module whose namespace binds this object
    leaf = path.split(".")[-1]
    best = None
    for name, mod in sorted(sys.modules.items()):
        if not name.startswith(("tilelang", "tvm")) or mod is None:
            continue
        try:
            if vars(mod).get(leaf, None) is obj:
                if best is None or len(name) > len(best):
                    best = name
        except Exception:
            continue
    return best or f"{type(obj).__module__} (instance)"


def evidence_for(path: str) -> dict:
    rec: dict = {"symbol": path}
    try:
        obj, _ = resolve(path)
    except Exception as e:
        rec.update(exists=False, error=f"{type(e).__name__}: {e}")
        return rec
    rec["exists"] = True
    rec["type"] = f"{type(obj).__module__}.{type(obj).__qualname__}"
    rec["defined_in_module"] = defining_module(obj, path)
    rec["signature"] = signature_of(obj)
    # source location
    target = obj
    if not (inspect.isfunction(obj) or inspect.isclass(obj) or inspect.ismethod(obj) or inspect.ismodule(obj)):
        target = None
    src_file = src_line = None
    if target is not None:
        try:
            u = inspect.unwrap(target)
            src_file = inspect.getsourcefile(u)
            src_line = inspect.getsourcelines(u)[1] if not inspect.ismodule(u) else 1
        except Exception as e:
            rec["installed_source"] = f"unavailable: {type(e).__name__}: {e}"
    if src_file:
        rec["installed_source"] = f"{src_file}:{src_line}"
        rel = installed_to_repo(src_file)
        if rel is None:
            rec["v0.1.11_source"] = "outside tilelang package (third-party dependency)"
        elif rel.startswith("3rdparty"):
            rec["v0.1.11_source"] = f"{rel}:{src_line}"
            rec["v0.1.11_source_note"] = "file belongs to the 3rdparty/tvm git submodule pinned by v0.1.11 (commit ec7f7bd92c185d8c0b84a5dc17709a0860aa5714); not present in the shallow checkout, located via the installed wheel"
        else:
            line = checkout_line(rel, src_line)
            with open(src_file, encoding="utf-8", errors="replace") as f:
                inst_lines = f.readlines()
            inst_line = inst_lines[src_line - 1].rstrip("\n") if src_line and src_line <= len(inst_lines) else None
            rec["v0.1.11_source"] = f"{rel}:{src_line}"
            rec["v0.1.11_line_matches_installed"] = (line is not None and line == inst_line)
            if line is None:
                rec["v0.1.11_source_note"] = "file or line not found in checkout"
    else:
        # instance / enum member: locate the binding by grep in the checkout, trying every
        # tilelang module that binds this exact object, then the owning class's module
        leaf = path.split(".")[-1]
        hints = []
        cands = binding_candidates(obj, leaf)
        for c in cands:
            hints += module_to_files(c)
        owner = None
        if len(path.split(".")) >= 3:
            try:
                owner, _ = resolve(".".join(path.split(".")[:-1]))
            except Exception:
                owner = None
        if inspect.isclass(owner):
            hints += module_to_files(owner.__module__)
        hit = grep_definition(leaf, hints)
        if hit:
            rec["v0.1.11_source"] = f"{hit[0]}:{hit[1]}"
            rec["v0.1.11_source_text"] = hit[2].strip()
            rec["defined_in_module"] = hit[0][:-3].replace(os.sep, ".").removesuffix(".__init__")
        elif "installed_source" not in rec:
            rec["v0.1.11_source"] = "binding not located by grep"
    if isinstance(obj, str):
        rec["value"] = str(obj)
    return rec


def main() -> None:
    text = open(SKILL, encoding="utf-8").read()
    present_text, absent_text, sec19_rest = split_sections(text)
    present = sorted(set(collect(present_text)) | set(collect(sec19_rest)) | set(EXTRA_PRESENT))
    absent = sorted(set(collect(absent_text)))

    records = [evidence_for(n) for n in present]
    absent_records = []
    for n in absent:
        try:
            resolve(n)
            absent_records.append({"symbol": n, "absent": False})
        except Exception as e:
            absent_records.append({"symbol": n, "absent": True, "error": f"{type(e).__name__}: {e}"})

    head = subprocess.run(["git", "-C", CHECKOUT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {
        "generated_by": "introspect.py",
        "skill_file": "SKILL.md",
        "versions": {
            "python": platform.python_version(),
            "tilelang": tilelang.__version__,
            "apache-tvm-ffi": md.version("apache-tvm-ffi"),
            "torch": md.version("torch"),
            "installed_tilelang_path": SITE_TILELANG,
        },
        "checkout": {
            "repo": "https://github.com/tile-ai/tilelang",
            "tag": "v0.1.11",
            "commit": head,
            "expected_commit": "cd37ed5fc35ae7a60a1277c8eb49028174ac51e6",
            "commit_ok": head == "cd37ed5fc35ae7a60a1277c8eb49028174ac51e6",
        },
        "summary": {
            "symbols_mentioned": len(records),
            "symbols_existing": sum(1 for r in records if r.get("exists")),
            "symbols_missing": [r["symbol"] for r in records if not r.get("exists")],
            "checkout_line_mismatches": [r["symbol"] for r in records if r.get("v0.1.11_line_matches_installed") is False],
            "absent_names_checked": len(absent_records),
            "absent_names_unexpectedly_present": [r["symbol"] for r in absent_records if not r["absent"]],
        },
        "symbols": records,
        "documented_absent_names": absent_records,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
