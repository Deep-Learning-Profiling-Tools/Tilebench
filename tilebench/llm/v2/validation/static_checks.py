"""Static evidence collection on a generated file.

Static analysis can only PROVIDE EVIDENCE. The verdict levels are:

- confirmed  : unambiguous by construction (an autotuner decorator or call
               resolved through the file's own import aliases, a forbidden
               reference-library call that computes the operator in host
               code, a subprocess / network / evaluator import). Triggers
               the same-round compliance repair.
- suspicious : patterns that need a human or execution-based decision
               (module-level caches keyed by tensor identity, loops that time
               several configurations, timing primitives, file/system
               access). Verdict review_required.
- none       : no evidence found.

Context model (this is what keeps kernel arithmetic from being mistaken for
host-side PyTorch delegation):

- import aliases are resolved (`import triton.language as tl`,
  `from triton import autotune as tune`, `import cuda.tile as ct`), so every
  dotted name is compared in canonical form;
- functions decorated with a DSL kernel decorator (`@triton.jit`,
  `@ct.kernel`, `@T.prim_func`, `@tilelang.jit`, `@nki.jit`, ...) and
  everything nested in them are KERNEL scope; the rest of the file is HOST
  scope. `x + y` inside a kernel is tile arithmetic; the same text in host
  scope operates on tensors;
- regex rules from the contract run on CODE-ONLY text (comments and string
  literals blanked, line structure preserved), in the scope the rule
  declares (`host` by default, `kernel`, or `any`).

Operator-specific forbidden/required patterns come from the evaluator rules
of the contract (validation.contract_checks)."""
from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field

CONFIRMED = "confirmed"
SUSPICIOUS = "suspicious"

# Canonical dotted names of kernel decorators per DSL (union used for scope).
KERNEL_DECORATORS = {
    "triton": {"triton.jit", "triton.language.jit", "triton.runtime.jit.jit"},
    "cutile": {"cuda.tile.kernel", "cuda.tile.jit", "cuda.tile.function"},
    "tilelang": {"tilelang.jit", "tilelang.language.prim_func", "tilelang.language.macro", "tilelang.prim_func",
                 "tvm.script.tir.prim_func", "tvm.script.prim_func"},
    "nki": {"neuronxcc.nki.jit", "nki.jit", "neuronxcc.nki.language.jit"},
}
_ALL_KERNEL_DECORATORS = set().union(*KERNEL_DECORATORS.values())

# Canonical autotuning / hidden-search entry points (suffix match on the canonical name).
_AUTOTUNE_CANONICAL = {
    "triton.autotune", "triton.runtime.autotuner.autotune", "triton.runtime.autotuner.Autotuner",
    "triton.runtime.Autotuner", "triton.testing.do_bench", "triton.testing.do_bench_cudagraph",
    "triton.testing.perf_report", "triton.testing.Benchmark",
    "cuda.tile.tune.exhaustive_search", "cuda.tile.tune.search", "cuda.tile_experimental.autotune_launch",
    "tilelang.autotune", "tilelang.autotuner.AutoTuner", "tilelang.autotuner.autotune",
    "tilelang.autotuner.set_autotune_inputs", "tilelang.set_autotune_inputs",
}
_AUTOTUNE_TAILS = ("autotune", "Autotuner", "AutoTuner", "exhaustive_search", "autotune_launch",
                   "set_autotune_inputs", "do_bench", "do_bench_cudagraph")

# Modules a generated file must never import (evaluator, reference, I/O, process control).
_FORBIDDEN_MODULES = {
    "subprocess", "socket", "urllib", "requests", "http", "ctypes", "multiprocessing",
    "tilebench.llm", "tilebench.core.timer", "tilebench.core.verifier", "tilebench.core.engine",
    "tilebench.benchmarks", "impl_torch", "impl_triton", "impl_cutile", "impl_tilelang", "impl_nki",
    "triton.profiler", "proton",
}
_FORBIDDEN_MODULE_PREFIXES = ("tilebench.llm", "tilebench.benchmarks", "tilebench.profiling")
_IO_MODULES = ("os", "sys", "pathlib", "shutil", "importlib", "inspect")

# torch calls that would delegate the operator's computation to a library (host scope).
_DELEGATION_CALLS = {
    "torch.nn.functional", "torch.matmul", "torch.mm", "torch.bmm", "torch.baddbmm", "torch.addmm",
    "torch.einsum", "torch.softmax", "torch.log_softmax", "torch.sort", "torch.argsort", "torch.topk",
    "torch.histc", "torch.bincount", "torch.cumsum", "torch.conv1d", "torch.conv2d", "torch.conv3d",
    "torch.nn.", "torch._scaled_mm", "torch.linalg", "torch.fft", "torch.scaled_dot_product_attention",
    "torch.layer_norm", "torch.rms_norm", "torch.batch_norm", "torch.kl_div", "torch.cross_entropy",
    "torch.max_pool2d", "torch.sigmoid", "torch.relu", "torch.nn.functional.",
}
_EVALUATOR_TAMPERING = ("torch.cuda.synchronize", "torch.cuda.Event", "torch.cuda.graph", "torch.cuda.CUDAGraph",
                        "torch.cuda.Stream", "time.perf_counter", "time.time")
_DYNAMIC_CALLS = ("setattr", "eval", "exec", "compile", "open", "__import__", "getattr", "globals", "vars")


@dataclass
class Evidence:
    level: str          # confirmed | suspicious
    category: str       # autotune | delegation | forbidden_import | cache | tampering | io | rule
    message: str
    line: int | None = None
    scope: str | None = None      # host | kernel
    matched: str | None = None    # text the rule matched (regex rules)


@dataclass
class StaticReport:
    evidence: list[Evidence] = field(default_factory=list)
    kernel_lines: set = field(default_factory=set)
    aliases: dict = field(default_factory=dict)
    code_text: str = ""                       # comments and strings blanked
    code_text_with_strings: str = ""          # comments blanked, string literals kept
    computational_lines: set = field(default_factory=set)
    run_params: list = field(default_factory=list)

    @property
    def confirmed(self) -> list[Evidence]:
        return [e for e in self.evidence if e.level == CONFIRMED]

    @property
    def suspicious(self) -> list[Evidence]:
        return [e for e in self.evidence if e.level == SUSPICIOUS]

    def verdict(self) -> str:
        if self.confirmed:
            return "confirmed_violation"
        if self.suspicious:
            return "review_required"
        return "clear"

    def scope_of(self, line: int) -> str:
        return "kernel" if line in self.kernel_lines else "host"

    def to_dict(self) -> dict:
        return {"verdict": self.verdict(), "evidence": [e.__dict__ for e in self.evidence],
                "kernel_line_count": len(self.kernel_lines), "aliases": dict(self.aliases)}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def import_aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> canonical dotted module/object name."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    aliases[a.asname] = a.name
                else:
                    head = a.name.split(".")[0]
                    aliases[head] = head
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                aliases[a.asname or a.name] = f"{node.module}.{a.name}"
    return aliases


def canonical(dotted: str | None, aliases: dict[str, str]) -> str:
    if not dotted:
        return ""
    head, *rest = dotted.split(".")
    if head in aliases:
        return ".".join([aliases[head]] + rest)
    return dotted


def code_only(source: str, *, keep_strings: bool = False) -> str:
    """Comments (and, unless keep_strings, string literals) blanked with
    spaces of the same length, newlines kept, so regex rules match code,
    not prose. Forbidden-substitution rules use the fully blanked text; the
    required-evidence rules keep strings because evidence such as
    float("-inf") lives inside a literal."""
    out = list(source)
    kinds = (tokenize.COMMENT,) if keep_strings else (tokenize.COMMENT, tokenize.STRING)
    try:
        toks = tokenize.generate_tokens(io.StringIO(source).readline)
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for ln in lines:
            offsets.append(offsets[-1] + len(ln))
        for tok in toks:
            if tok.type in kinds:
                (sr, sc), (er, ec) = tok.start, tok.end
                start = offsets[sr - 1] + sc
                end = offsets[er - 1] + ec
                for i in range(start, min(end, len(out))):
                    if out[i] != "\n":
                        out[i] = " "
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return source
    return "".join(out)


def _is_kernel_decorated(node: ast.AST, aliases: dict[str, str]) -> bool:
    for dec in getattr(node, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = canonical(_dotted(target), aliases)
        if name in _ALL_KERNEL_DECORATORS:
            return True
    return False


def kernel_line_set(tree: ast.AST, aliases: dict[str, str]) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_kernel_decorated(node, aliases):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            lines.update(range(start, (node.end_lineno or node.lineno) + 1))
    return lines


def _computational_lines(tree: ast.AST) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Call, ast.BinOp, ast.AugAssign, ast.Return, ast.Subscript, ast.Compare, ast.UnaryOp)):
            if hasattr(node, "lineno"):
                lines.update(range(node.lineno, (getattr(node, "end_lineno", None) or node.lineno) + 1))
    return lines


def _run_params(tree: ast.AST) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return [a.arg for a in node.args.posonlyargs + node.args.args]
    return []


def is_autotune_name(name: str) -> bool:
    if name in _AUTOTUNE_CANONICAL:
        return True
    tail = name.split(".")[-1]
    return tail in _AUTOTUNE_TAILS


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def analyze(source: str, dsl: str, *, allowed_torch_calls: tuple[str, ...] = ()) -> StaticReport:
    rep = StaticReport()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        rep.evidence.append(Evidence(SUSPICIOUS, "io", f"unparsable source: {e}", e.lineno))
        rep.code_text = source
        rep.code_text_with_strings = source
        return rep
    aliases = import_aliases(tree)
    rep.aliases = aliases
    rep.kernel_lines = kernel_line_set(tree, aliases)
    rep.code_text = code_only(source)
    rep.code_text_with_strings = code_only(source, keep_strings=True)
    rep.computational_lines = _computational_lines(tree)
    rep.run_params = _run_params(tree)
    allowed = set(allowed_torch_calls)

    # imports
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for n in names:
            if n in _FORBIDDEN_MODULES or n.startswith(_FORBIDDEN_MODULE_PREFIXES) or n.split(".")[0] in _FORBIDDEN_MODULES:
                rep.evidence.append(Evidence(CONFIRMED, "forbidden_import", f"import of {n!r}", node.lineno, "host"))
            if n.split(".")[0] in _IO_MODULES:
                rep.evidence.append(Evidence(SUSPICIOUS, "io", f"import of {n!r} (file/system access)", node.lineno, "host"))
            if is_autotune_name(n) and n.split(".")[0] in ("triton", "tilelang", "cuda"):
                rep.evidence.append(Evidence(CONFIRMED, "autotune", f"import of autotuning entry point {n!r}", node.lineno, "host"))

    # decorators and calls
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                name = canonical(_dotted(target), aliases)
                if is_autotune_name(name):
                    rep.evidence.append(Evidence(CONFIRMED, "autotune", f"autotune decorator {name}", node.lineno, rep.scope_of(node.lineno)))
        if isinstance(node, ast.Call):
            raw = _dotted(node.func) or ""
            name = canonical(raw, aliases)
            scope = rep.scope_of(node.lineno)
            if is_autotune_name(name) and (name.split(".")[0] in ("triton", "tilelang", "cuda") or name in _AUTOTUNE_CANONICAL
                                           or name.split(".")[-1] in ("exhaustive_search", "autotune_launch", "set_autotune_inputs")):
                rep.evidence.append(Evidence(CONFIRMED, "autotune", f"call to {name}", node.lineno, scope))
            if name.startswith("torch.") and name not in allowed and scope == "host":
                if any(name == c or name.startswith(c) for c in _DELEGATION_CALLS):
                    rep.evidence.append(Evidence(CONFIRMED, "delegation", f"reference-library computation {name}", node.lineno, scope))
            if scope == "host" and any(name == t or name.startswith(t + ".") for t in _EVALUATOR_TAMPERING):
                rep.evidence.append(Evidence(SUSPICIOUS, "tampering", f"timing/sync primitive {name}", node.lineno, scope))
            if name in _DYNAMIC_CALLS:
                rep.evidence.append(Evidence(SUSPICIOUS, "io", f"dynamic/IO call {name}", node.lineno, scope))
            if name.startswith("torch.") and name.split(".")[-1] in ("manual_seed", "seed"):
                rep.evidence.append(Evidence(SUSPICIOUS, "tampering", "re-seeding torch RNG", node.lineno, scope))

    # module-level mutable caches keyed by tensors (output caching evidence);
    # empty containers whose name says "config" are the mutable-dict config
    # record pattern, not a result cache.
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, (ast.Dict, ast.List)) and len(getattr(value, "keys", getattr(value, "elts", []))) == 0:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and not any(k in t.id.lower() for k in ("config", "cfg")):
                        rep.evidence.append(Evidence(SUSPICIOUS, "cache", f"module-level mutable container {t.id!r} (possible result cache)", node.lineno, "host"))
    code = rep.code_text
    for pat, msg in ((r"WeakTensorKeyDictionary", "tensor-identity dictionary"),
                     (r"functools\.lru_cache|functools\.cache\b|@lru_cache|@cache\b", "function result cache")):
        for m in re.finditer(pat, code):
            line = code.count("\n", 0, m.start()) + 1
            rep.evidence.append(Evidence(SUSPICIOUS, "cache", msg, line, rep.scope_of(line), m.group(0)))
    for line in data_ptr_key_uses(tree):
        rep.evidence.append(Evidence(SUSPICIOUS, "cache", "tensor address (data_ptr) used as a key or kept in a variable",
                                     line, rep.scope_of(line), ".data_ptr()"))
    return rep


def data_ptr_key_uses(tree: ast.AST) -> list[int]:
    """Lines where a `.data_ptr()` value can key a cache: used as a subscript
    index, passed to get/setdefault/pop/__contains__, tested with `in`, or
    stored in a variable. A bare comparison such as
    `assert x.data_ptr() == y.data_ptr()` (a no-op-contiguity guard) is not
    evidence of caching."""
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    lines: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "data_ptr"):
            continue
        cur, parent = node, parents.get(id(node))
        suspicious = False
        while parent is not None and not suspicious:
            if isinstance(parent, ast.Subscript) and parent.slice is cur:
                suspicious = True
            elif isinstance(parent, ast.Call) and cur in parent.args and isinstance(parent.func, ast.Attribute) \
                    and parent.func.attr in ("get", "setdefault", "pop", "__contains__", "add", "append"):
                suspicious = True
            elif isinstance(parent, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops):
                suspicious = True
            elif isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) and getattr(parent, "value", None) is cur:
                suspicious = True
            elif isinstance(parent, (ast.Dict, ast.Tuple, ast.List, ast.Set)):
                cur, parent = parent, parents.get(id(parent))
                continue
            elif isinstance(parent, (ast.Compare, ast.Assert, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.If, ast.IfExp, ast.Expr)):
                break
            else:
                cur, parent = parent, parents.get(id(parent))
                continue
        if suspicious:
            lines.append(node.lineno)
    return sorted(set(lines))


def hidden_search_evidence(source: str, aliases: dict[str, str] | None = None) -> list[Evidence]:
    """Loops that iterate over several configurations and time them inside
    host code: treated as suspicious (needs execution/human confirmation)."""
    out = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    aliases = aliases if aliases is not None else import_aliases(tree)
    kernel = kernel_line_set(tree, aliases)
    timing_names = {"perf_counter", "time", "Event", "elapsed_time", "synchronize", "do_bench"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)) and node.lineno not in kernel:
            body_src = ast.unparse(node)
            if any(t in body_src for t in timing_names) and ("config" in body_src.lower() or "BLOCK" in body_src):
                out.append(Evidence(SUSPICIOUS, "autotune", "loop over configurations with timing calls", node.lineno, "host"))
    return out
