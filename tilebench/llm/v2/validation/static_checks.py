"""Static evidence collection on a generated file (checker v2).

Static analysis can only PROVIDE EVIDENCE. Evidence levels and the
compliance verdict they lead to (study.yaml `compliance`):

- confirmed : unambiguous by construction (an autotuner decorator or call
              resolved through the file's own import/assignment aliases, a
              forbidden reference-library call that computes the operator in
              host code, a subprocess / network / evaluator import, a change
              of process-wide precision state). -> confirmed_violation: the
              round closes, no regeneration.
- review    : HIGH-RISK positive evidence the checker cannot settle:
              persistent state written with tensor data or tensor identity
              (possible cross-call result cache), a tensor-identity
              dictionary, a functools cache called with tensors, dynamic code
              execution / file access, work moved onto another stream or
              graph, a configuration loop with timing calls, a dynamic
              attribute lookup on torch. -> review_required (blocks the
              trajectory until a human decision).
- audit     : recorded for audit, never blocking: local metadata use of
              data_ptr, persistent state holding only non-tensor values
              (compiled-kernel / configuration caches), plain timing/sync
              calls, os/sys/pathlib imports, getattr. -> audit_only (the
              candidate is evaluated normally).

A missing positive pattern is never evidence of a violation (contract
required-evidence misses are audit_only, validation.contract_checks).

Context model (keeps kernel arithmetic from being mistaken for host-side
PyTorch delegation):

- import aliases are resolved (`import triton.language as tl`,
  `from triton import autotune as tune`, `import cuda.tile as ct`), and so
  are plain assignment aliases of torch objects (`F = torch.nn.functional`,
  `mm = torch.matmul`) and `getattr(<module>, "<literal>")`, so every dotted
  name is compared in canonical form;
- functions decorated with a DSL kernel decorator (`@triton.jit`,
  `@ct.kernel`, `@T.prim_func`, `@tilelang.jit`, `@nki.jit`, ...) and
  everything nested in them are KERNEL scope; the rest of the file is HOST
  scope. `x + y` inside a kernel is tile arithmetic;
- persistent state: module-level names (containers, objects, functions,
  classes), `global` names and functools-cached functions. A write into
  persistent state is classified by a data-flow analysis of the host
  functions: a value is TENSOR-DERIVED when it comes from a tensor argument of
  `run()` (or of a helper called with one), from a torch call that produces a
  tensor, or from a tensor address (`data_ptr()`, `id()`); shape, dtype,
  device, stride and numel reads are metadata and break the derivation;
- regex rules from the contract run on CODE-ONLY text (comments and string
  literals blanked, line structure preserved), in the scope the rule
  declares (`host` by default, `kernel`, or `any`)."""
from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field

CONFIRMED = "confirmed"
REVIEW = "review"
AUDIT = "audit"
LEVELS = (CONFIRMED, REVIEW, AUDIT)

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
_AUDIT_IO_MODULES = ("os", "sys", "pathlib")                 # environment/path reads: audit only
_REVIEW_IO_MODULES = ("shutil", "importlib", "inspect")       # can load or inspect other code (incl. the evaluator)

# torch calls that would delegate the operator's computation to a library (host scope).
_DELEGATION_CALLS = {
    "torch.nn.functional", "torch.matmul", "torch.mm", "torch.bmm", "torch.baddbmm", "torch.addmm",
    "torch.einsum", "torch.softmax", "torch.log_softmax", "torch.sort", "torch.argsort", "torch.topk",
    "torch.histc", "torch.bincount", "torch.cumsum", "torch.conv1d", "torch.conv2d", "torch.conv3d",
    "torch.nn.", "torch._scaled_mm", "torch.linalg", "torch.fft", "torch.scaled_dot_product_attention",
    "torch.layer_norm", "torch.rms_norm", "torch.batch_norm", "torch.kl_div", "torch.cross_entropy",
    "torch.max_pool2d", "torch.sigmoid", "torch.relu", "torch.nn.functional.",
}
_STREAM_GRAPH = ("torch.cuda.graph", "torch.cuda.CUDAGraph", "torch.cuda.Stream", "torch.cuda.stream",
                 "torch.cuda.graphs")          # may move work out of the evaluator's captured / timed stream
_TIMING_SYNC = ("torch.cuda.synchronize", "torch.cuda.Event", "time.perf_counter", "time.time", "time.sleep")
_REVIEW_DYNAMIC = ("eval", "exec", "compile", "open", "__import__", "setattr", "globals", "vars", "delattr")
_AUDIT_DYNAMIC = ("getattr",)
# process-wide numerical state: changing it alters the reference and later candidates (evaluator state)
_PRECISION_STATE_CALLS = ("torch.set_float32_matmul_precision", "torch.use_deterministic_algorithms", "torch.set_default_dtype",
                          "torch.set_default_device", "torch.backends.cudnn.flags", "torch.backends.cuda.matmul.flags",
                          "torch.set_flush_denormal")
# torch calls that return metadata or handles, not tensor data
_TORCH_NON_TENSOR = ("torch.cuda.current_stream", "torch.cuda.current_device", "torch.cuda.device", "torch.device",
                     "torch.Size", "torch.finfo", "torch.iinfo", "torch.cuda.get_device_properties",
                     "torch.cuda.get_device_capability", "torch.cuda.get_device_name", "torch.get_default_dtype",
                     "torch.cuda.is_available", "torch.is_tensor", "torch.cuda.device_count", "torch.dtype",
                     "torch.cuda.set_device")
METADATA_ATTRS = {"shape", "dtype", "device", "ndim", "is_contiguous", "stride", "size", "numel", "element_size",
                  "layout", "requires_grad", "is_cuda", "nbytes", "itemsize", "dim", "get_device", "storage_offset",
                  "is_floating_point"}
_SCALAR_ANNOTATIONS = {"int", "float", "bool", "str", "None"}
_MUTATORS = {"update", "setdefault", "append", "add", "insert", "extend", "__setitem__", "appendleft"}
_CACHE_DECORATORS = {"functools.lru_cache", "functools.cache", "functools.cached_property", "lru_cache", "cache"}
_IDENTITY_DICTS = ("WeakTensorKeyDictionary", "WeakKeyDictionary", "WeakValueDictionary")


@dataclass
class Evidence:
    level: str          # confirmed | review | audit
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
    degenerate_exit_lines: set = field(default_factory=set)   # host lines only reached for inputs of size <= 1
    matmult_lines: set = field(default_factory=set)           # lines spanned by an AST infix `@` (MatMult) expression
    decorator_lines: set = field(default_factory=set)         # lines of decorator expressions
    defined_names: set = field(default_factory=set)           # names the candidate binds itself (defs, args, targets)
    imported_names: set = field(default_factory=set)          # local names bound by import statements
    import_lines: set = field(default_factory=set)
    guarded_contiguous_lines: set = field(default_factory=set)  # `.contiguous()` under an is_contiguous() test
    name_tokens: list = field(default_factory=list)           # (line, col_start, col_end, name) of every NAME token
    keyword_names: set = field(default_factory=set)           # keyword-argument names used in calls (parameter names)

    @property
    def confirmed(self) -> list[Evidence]:
        return [e for e in self.evidence if e.level == CONFIRMED]

    @property
    def review(self) -> list[Evidence]:
        return [e for e in self.evidence if e.level == REVIEW]

    @property
    def audit(self) -> list[Evidence]:
        return [e for e in self.evidence if e.level == AUDIT]

    def verdict(self) -> str:
        if self.confirmed:
            return "confirmed_violation"
        if self.review:
            return "review_required"
        if self.audit:
            return "audit_only"
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


def _literal_getattr(node: ast.AST) -> tuple[str, str] | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" \
            and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
        base = _dotted(node.args[0])
        if base:
            return base, node.args[1].value
    return None


def _assignment_aliases(tree: ast.AST, aliases: dict[str, str]) -> dict[str, str]:
    """`F = torch.nn.functional`, `mm = torch.matmul`, `op = getattr(torch, "matmul")`: plain (non-call)
    bindings of torch objects are aliases too, so a call through them resolves canonically."""
    extra: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = None
            v = node.value
            if isinstance(v, (ast.Attribute, ast.Name)):
                target = canonical(_dotted(v), {**aliases, **extra})
            elif _literal_getattr(v):
                base, attr = _literal_getattr(v)
                target = f"{canonical(base, {**aliases, **extra})}.{attr}"
            if target and target.split(".")[0] == "torch" and node.targets[0].id not in aliases:
                extra[node.targets[0].id] = target
    return extra


def call_name(node: ast.Call, aliases: dict[str, str]) -> str:
    lit = _literal_getattr(node.func) if isinstance(node.func, ast.Call) else None
    if lit:
        return f"{canonical(lit[0], aliases)}.{lit[1]}"
    return canonical(_dotted(node.func), aliases)


def code_only(source: str, *, keep_strings: bool = False) -> str:
    """Comments (and, unless keep_strings, string literals) blanked with
    spaces of the same length, newlines kept, so regex rules match code,
    not prose."""
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


def matmult_lines(tree: ast.AST) -> set[int]:
    """Lines between the left and right operand of every infix `@` (where the operator token can be)."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
            a = (node.left.end_lineno if isinstance(node, ast.BinOp) else node.target.end_lineno) or node.lineno
            b = (node.right.lineno if isinstance(node, ast.BinOp) else node.value.lineno) or node.lineno
            out.update(range(min(a, b), max(a, b) + 1))
    return out


def decorator_line_set(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        for dec in getattr(node, "decorator_list", []) or []:
            out.update(range(dec.lineno, (dec.end_lineno or dec.lineno) + 1))
    return out


def defined_name_set(tree: ast.AST, aliases: dict[str, str]) -> set[str]:
    """Names the candidate binds itself: function/class names (nested included), parameters, assignment /
    loop / with / comprehension targets and attribute-assignment names. A plain alias of an imported object
    (`f = lib.op`) and every import-bound name are NOT candidate-defined."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            names.add(node.attr)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Name, ast.Attribute)):
            root = (_dotted(node.value) or "").split(".")[0]
            if root in aliases:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.discard(t.id)
    return names - set(aliases)


def _mentions_is_contiguous(expr: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "is_contiguous"
               for n in ast.walk(expr))


def guarded_contiguous_line_set(tree: ast.AST) -> set[int]:
    """Lines of `.contiguous()` calls that only run under a test involving `.is_contiguous()`
    (`if not x.is_contiguous(): x = x.contiguous()`, `x if x.is_contiguous() else x.contiguous()`)."""
    parents = _parents(tree)
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "contiguous":
            cur = node
            while id(cur) in parents:
                p = parents[id(cur)]
                if isinstance(p, (ast.If, ast.IfExp, ast.While)) and cur is not p.test and _mentions_is_contiguous(p.test):
                    out.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
                    break
                if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module, ast.ClassDef, ast.Lambda)):
                    break
                cur = p
    return out


def name_token_spans(source: str) -> list[tuple[int, int, int, str]]:
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.NAME:
                out.append((tok.start[0], tok.start[1], tok.end[1], tok.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


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


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def degenerate_exit_lines(tree: ast.AST, kernel: set[int]) -> set[int]:
    """Host lines inside an `if` whose test compares a size against a
    constant <= 1 (`N <= 1`, `n < 2`, `x.numel() == 0`, ...) and whose body
    returns: they run only for degenerate inputs, never for a benchmark task
    (every task's problem size is > 1)."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or node.lineno in kernel:
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1):
            continue
        op, rhs = test.ops[0], test.comparators[0]
        if not (isinstance(rhs, ast.Constant) and isinstance(rhs.value, int) and not isinstance(rhs.value, bool)):
            continue
        c = rhs.value
        degenerate = (isinstance(op, ast.LtE) and c <= 1) or (isinstance(op, ast.Lt) and c <= 2) or \
                     (isinstance(op, ast.Eq) and c in (0, 1))
        if degenerate and node.body and isinstance(node.body[-1], ast.Return):
            out.update(range(node.body[0].lineno, (node.body[-1].end_lineno or node.body[-1].lineno) + 1))
    return out


# --------------------------------------------------------------------------
# data flow: tensor-derived values and persistent state (host scope)
# --------------------------------------------------------------------------

class _Flow:
    def __init__(self, tree: ast.Module, aliases: dict[str, str], kernel: set[int]):
        self.tree, self.aliases, self.kernel = tree, aliases, kernel
        self.parents = _parents(tree)
        self.functions: dict[str, ast.FunctionDef] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno not in kernel:
                self.functions[n.name] = n
            elif isinstance(n, ast.ClassDef):
                self.classes[n.name] = n
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.lineno not in kernel:
                        self.functions[f"{n.name}.{m.name}"] = m
        self.module_names = self._module_names()
        self.all_defs = {n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.tainted: dict[str, set[str]] = {}
        self._propagate()

    def _module_names(self) -> set[str]:
        names: set[str] = set()
        for n in self.tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            names.add(x.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
        return names

    # -- expression predicates ------------------------------------------------
    def refs(self, expr: ast.AST, names: set[str]) -> bool:
        """expr uses a tensor-derived name other than through metadata."""
        for n in ast.walk(expr):
            if isinstance(n, ast.Name) and n.id in names and isinstance(n.ctx, ast.Load):
                p = self.parents.get(id(n))
                if isinstance(p, ast.Attribute) and p.value is n and p.attr in METADATA_ATTRS:
                    continue
                if isinstance(p, ast.Call) and isinstance(p.func, ast.Name) and p.func.id in ("len", "isinstance", "type"):
                    continue
                return True
        return False

    def has_address(self, expr: ast.AST) -> bool:
        for n in ast.walk(expr):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Attribute) and n.func.attr == "data_ptr":
                    return True
                if isinstance(n.func, ast.Name) and n.func.id == "id":
                    return True
        return False

    def has_torch_value(self, expr: ast.AST) -> bool:
        for n in ast.walk(expr):
            if isinstance(n, ast.Call):
                name = call_name(n, self.aliases)
                if name.startswith("torch.") and not any(name == t or name.startswith(t + ".") for t in _TORCH_NON_TENSOR):
                    return True
        return False

    def tainted_expr(self, expr: ast.AST | None, names: set[str]) -> bool:
        if expr is None:
            return False
        return self.refs(expr, names) or self.has_address(expr) or self.has_torch_value(expr)

    # -- propagation ------------------------------------------------------------
    @staticmethod
    def _params(fn: ast.FunctionDef) -> list[ast.arg]:
        return fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs

    def _seed_run(self) -> dict[str, set[str]]:
        seeds: dict[str, set[str]] = {k: set() for k in self.functions}
        run = self.functions.get("run")
        if run is not None:
            for a in self._params(run):
                ann = ast.unparse(a.annotation) if a.annotation is not None else None
                if ann is None or not any(ann == s or ann.endswith("." + s) for s in _SCALAR_ANNOTATIONS):
                    seeds["run"].add(a.arg)
        return seeds

    def _local_taint(self, fn: ast.FunctionDef, seed: set[str]) -> set[str]:
        names = set(seed)
        changed = True
        while changed:
            changed = False
            for node in ast.walk(fn):
                if getattr(node, "lineno", None) in self.kernel:
                    continue
                targets, value = [], None
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    value = node.value
                elif isinstance(node, ast.NamedExpr):
                    targets, value = [node.target], node.value
                elif isinstance(node, (ast.For, ast.comprehension)):
                    targets, value = [node.target], node.iter
                elif isinstance(node, ast.withitem) and node.optional_vars is not None:
                    targets, value = [node.optional_vars], node.context_expr
                if value is None or not self.tainted_expr(value, names):
                    continue
                for t in targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name) and x.id not in names:
                            names.add(x.id)
                            changed = True
        return names

    def _callee(self, call: ast.Call, owner: str) -> tuple[str | None, int]:
        """(function key, positional offset) of a call to a module function / class / method of the same class."""
        f = call.func
        if isinstance(f, ast.Name):
            if f.id in self.functions:
                return f.id, 0
            if f.id in self.classes and f"{f.id}.__init__" in self.functions:
                return f"{f.id}.__init__", 1
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self" and "." in owner:
            key = f"{owner.split('.')[0]}.{f.attr}"
            if key in self.functions:
                return key, 1
        return None, 0

    def _propagate(self) -> None:
        seeds = self._seed_run()
        for _ in range(12):
            changed = False
            for key, fn in self.functions.items():
                self.tainted[key] = self._local_taint(fn, seeds[key])
            for key, fn in self.functions.items():
                for call in ast.walk(fn):
                    if not isinstance(call, ast.Call):
                        continue
                    callee, offset = self._callee(call, key)
                    if callee is None:
                        continue
                    params = [a.arg for a in self._params(self.functions[callee])]
                    for i, arg in enumerate(call.args):
                        j = i + offset
                        if j < len(params) and self.tainted_expr(arg, self.tainted[key]) and params[j] not in seeds[callee]:
                            seeds[callee].add(params[j])
                            changed = True
                    for kw in call.keywords:
                        if kw.arg in params and self.tainted_expr(kw.value, self.tainted[key]) and kw.arg not in seeds[callee]:
                            seeds[callee].add(kw.arg)
                            changed = True
            if not changed:
                break
        for key, fn in self.functions.items():
            self.tainted[key] = self._local_taint(fn, seeds[key])

    # -- value kinds of persistent writes ---------------------------------------
    _COMPILE_TAILS = {"jit", "compile", "kernel", "prim_func", "JITKernel", "lower", "build", "function"}

    def _is_persistent_read(self, expr: ast.AST) -> bool:
        """`CACHE.get(k)`, `CACHE[k]`, `CACHE.setdefault(k, ...)` on a module-level container: a cache lookup."""
        node = expr
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("get", "setdefault", "pop"):
            node = node.func.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        else:
            return False
        return self._base_name(node) in self.module_names

    def _assigned_values(self, scope: ast.AST, name: str) -> list[ast.AST]:
        """Values assigned to `name` in `scope`, cache lookups from module-level containers left out."""
        vals = [n.value for n in ast.walk(scope) if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
        return [v for v in vals if not self._is_persistent_read(v)]

    def _is_compile_call(self, call: ast.Call) -> bool:
        name = call_name(call, self.aliases)
        if name in _ALL_KERNEL_DECORATORS:
            return True
        parts = name.split(".")
        return parts[0] in ("triton", "tilelang", "cuda", "tvm") and parts[-1] in self._COMPILE_TAILS

    def value_kind(self, expr: ast.AST | None, scope: ast.AST, depth: int = 0) -> str:
        """`kernel` (a compiled kernel / JIT object), `function` (a function object) or `value`
        (anything else: tensors, scalars, unknown). Generic, no operator- or candidate-specific names."""
        if expr is None or depth > 4:
            return "value"
        if isinstance(expr, ast.Lambda):
            return "function"
        if isinstance(expr, ast.Name):
            if expr.id in self.all_defs:
                return "function"
            vals = self._assigned_values(scope, expr.id)
            kinds = {self.value_kind(v, scope, depth + 1) for v in vals}
            if vals and kinds <= {"kernel", "function"}:
                return "kernel" if "kernel" in kinds else "function"
            return "value"
        if isinstance(expr, ast.Call):
            if self._is_compile_call(expr):
                return "kernel"
            f = expr.func
            # a kernel factory: a call to a DSL decorator's result, e.g. tilelang.jit(...)(fn)
            if isinstance(f, ast.Call) and self._is_compile_call(f):
                return "kernel"
            if isinstance(f, ast.Name) and f.id in self.all_defs:
                d = self.all_defs[f.id]
                if _is_kernel_decorated(d, self.aliases):
                    return "kernel"
                rets = [r.value for r in ast.walk(d) if isinstance(r, ast.Return) and r.value is not None]
                if rets and all(self.value_kind(r, d, depth + 1) in ("kernel", "function") for r in rets):
                    return "kernel"
        return "value"

    # -- persistent writes ------------------------------------------------------
    @staticmethod
    def mutable_defaults(fn: ast.FunctionDef) -> set[str]:
        """Parameters whose default is a mutable container: the default object persists across calls."""
        args = fn.args.posonlyargs + fn.args.args
        pairs = list(zip(args[len(args) - len(fn.args.defaults):], fn.args.defaults)) + \
            [(a, d) for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is not None]
        out = set()
        for a, d in pairs:
            if isinstance(d, (ast.Dict, ast.List, ast.Set)) or \
                    (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in ("dict", "list", "set")):
                out.add(a.arg)
        return out

    def _locals(self, fn: ast.FunctionDef) -> set[str]:
        out = {a.arg for a in self._params(fn)} - self.mutable_defaults(fn)
        globals_ = {n for g in ast.walk(fn) if isinstance(g, ast.Global) for n in g.names}
        for node in ast.walk(fn):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For)):
                targets = [node.target]
            elif isinstance(node, ast.With):
                targets = [i.optional_vars for i in node.items if i.optional_vars is not None]
            for t in targets:
                if isinstance(t, ast.Name) and t.id not in globals_:
                    out.add(t.id)
        return out

    @staticmethod
    def _base_name(node: ast.AST) -> str | None:
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def persistent_writes(self) -> list[dict]:
        out = []
        for key, fn in self.functions.items():
            local = self._locals(fn)
            globals_ = {n for g in ast.walk(fn) if isinstance(g, ast.Global) for n in g.names}
            names = self.tainted.get(key, set())
            for node in ast.walk(fn):
                if getattr(node, "lineno", None) in self.kernel:
                    continue
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for t in targets:
                        if isinstance(t, ast.Name) and t.id in globals_:
                            out.append({"line": node.lineno, "target": t.id, "kind": "global assignment",
                                        "tensor": self.tainted_expr(node.value, names)})
                        elif isinstance(t, (ast.Subscript, ast.Attribute)):
                            base = self._base_name(t)
                            persistent = self.module_names | self.mutable_defaults(fn)
                            if base and base in persistent and base not in local:
                                key_expr = t.slice if isinstance(t, ast.Subscript) else None
                                vk = self.value_kind(node.value, fn)
                                if vk in ("kernel", "function"):
                                    # a compiled-kernel / function cache keyed by shape metadata (or anything else):
                                    # it stores code, not tensor data; never a cross-call result cache
                                    out.append({"line": node.lineno, "target": ast.unparse(t), "kind": "write into module-level object",
                                                "tensor": False, "value_kind": vk,
                                                "keyed_by_address": self.has_address(key_expr) if key_expr is not None else False})
                                    continue
                                out.append({"line": node.lineno, "target": ast.unparse(t), "kind": "write into module-level object",
                                            "tensor": self.tainted_expr(node.value, names) or self.tainted_expr(key_expr, names),
                                            "value_kind": vk})
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _MUTATORS:
                    base = self._base_name(node.func.value)
                    if base and base in (self.module_names | self.mutable_defaults(fn)) and base not in local:
                        tensor = any(self.tainted_expr(a, names) for a in node.args) or \
                            any(self.tainted_expr(k.value, names) for k in node.keywords)
                        out.append({"line": node.lineno, "target": ast.unparse(node.func.value),
                                    "kind": f"{node.func.attr}() on module-level object", "tensor": tensor})
        return out

    def cache_decorated(self) -> list[tuple[str, int]]:
        out = []
        for key, fn in self.functions.items():
            for dec in fn.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                name = canonical(_dotted(target), self.aliases)
                if name in _CACHE_DECORATORS:
                    out.append((key, fn.lineno))
        return out

    def calls_with_tensors(self, callee: str) -> list[int]:
        lines = []
        short = callee.split(".")[-1]
        for key, fn in self.functions.items():
            names = self.tainted.get(key, set())
            for call in ast.walk(fn):
                if isinstance(call, ast.Call):
                    f = call.func
                    hit = (isinstance(f, ast.Name) and f.id == short) or \
                          (isinstance(f, ast.Attribute) and f.attr == short and "." in callee)
                    if hit and (any(self.tainted_expr(a, names) for a in call.args) or
                                any(self.tainted_expr(k.value, names) for k in call.keywords)):
                        lines.append(call.lineno)
        return lines

    def host_matmul_on_tensors(self) -> list[int]:
        lines = []
        for key, fn in self.functions.items():
            names = self.tainted.get(key, set())
            for node in ast.walk(fn):
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult) and node.lineno not in self.kernel:
                    if self.refs(node.left, names) and self.refs(node.right, names):
                        lines.append(node.lineno)
        return lines


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def analyze(source: str, dsl: str, *, allowed_torch_calls: tuple[str, ...] = ()) -> StaticReport:
    rep = StaticReport()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        rep.evidence.append(Evidence(AUDIT, "io", f"unparsable source: {e} (the evaluator reports the failure)", e.lineno))
        rep.code_text = source
        rep.code_text_with_strings = source
        return rep
    aliases = import_aliases(tree)
    aliases.update(_assignment_aliases(tree, aliases))
    rep.aliases = aliases
    rep.kernel_lines = kernel_line_set(tree, aliases)
    rep.code_text = code_only(source)
    rep.code_text_with_strings = code_only(source, keep_strings=True)
    rep.computational_lines = _computational_lines(tree)
    rep.run_params = _run_params(tree)
    rep.degenerate_exit_lines = degenerate_exit_lines(tree, rep.kernel_lines)
    rep.matmult_lines = matmult_lines(tree)
    rep.decorator_lines = decorator_line_set(tree)
    rep.defined_names = defined_name_set(tree, aliases)
    rep.imported_names = set(import_aliases(tree))
    rep.import_lines = {n.lineno for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))}
    rep.guarded_contiguous_lines = guarded_contiguous_line_set(tree)
    rep.name_tokens = name_token_spans(source)
    rep.keyword_names = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords if k.arg}
    allowed = set(allowed_torch_calls)
    imported_modules = {v for k, v in aliases.items() if "." not in v and k == v} | \
        {k for k, v in aliases.items() if v.split(".")[0] in ("torch", "triton", "tilelang", "cuda", "numpy", "tvm")}

    # imports
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for n in names:
            head = n.split(".")[0]
            if n in _FORBIDDEN_MODULES or n.startswith(_FORBIDDEN_MODULE_PREFIXES) or head in _FORBIDDEN_MODULES:
                rep.evidence.append(Evidence(CONFIRMED, "forbidden_import", f"import of {n!r}", node.lineno, "host"))
            if head in _REVIEW_IO_MODULES:
                rep.evidence.append(Evidence(REVIEW, "io", f"import of {n!r} (can load or inspect other code)", node.lineno, "host"))
            elif head in _AUDIT_IO_MODULES:
                rep.evidence.append(Evidence(AUDIT, "io", f"import of {n!r} (environment/path access)", node.lineno, "host"))
            if is_autotune_name(n) and head in ("triton", "tilelang", "cuda"):
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
            name = call_name(node, aliases)
            scope = rep.scope_of(node.lineno)
            if is_autotune_name(name) and (name.split(".")[0] in ("triton", "tilelang", "cuda") or name in _AUTOTUNE_CANONICAL
                                           or name.split(".")[-1] in ("exhaustive_search", "autotune_launch", "set_autotune_inputs")):
                rep.evidence.append(Evidence(CONFIRMED, "autotune", f"call to {name}", node.lineno, scope))
            if name.startswith("torch.") and name not in allowed and scope == "host":
                if any(name == c or name.startswith(c) for c in _DELEGATION_CALLS):
                    rep.evidence.append(Evidence(CONFIRMED, "delegation", f"reference-library computation {name}", node.lineno, scope))
                elif name.startswith("torch.ops."):
                    rep.evidence.append(Evidence(REVIEW, "delegation", f"operator-library call {name} (possible delegated computation)",
                                                 node.lineno, scope))
            if scope == "host" and any(name == t or name.startswith(t + ".") for t in _STREAM_GRAPH):
                rep.evidence.append(Evidence(REVIEW, "tampering", f"stream/graph primitive {name} (work may leave the evaluator's "
                                                                  "captured, timed stream)", node.lineno, scope))
            elif scope == "host" and any(name == t or name.startswith(t + ".") for t in _TIMING_SYNC):
                rep.evidence.append(Evidence(AUDIT, "tampering", f"timing/sync call {name}", node.lineno, scope))
            if name in _REVIEW_DYNAMIC:
                rep.evidence.append(Evidence(REVIEW, "io", f"dynamic code / IO call {name}", node.lineno, scope))
            elif name in _AUDIT_DYNAMIC:
                if len(node.args) >= 2 and not isinstance(node.args[1], ast.Constant) \
                        and canonical(_dotted(node.args[0]), aliases).split(".")[0] == "torch":
                    rep.evidence.append(Evidence(REVIEW, "delegation", "dynamic attribute lookup on torch (possible delegated "
                                                                       "computation through an ambiguous alias)", node.lineno, scope))
                else:
                    rep.evidence.append(Evidence(AUDIT, "io", f"dynamic attribute access {name}", node.lineno, scope))
            if name.startswith("torch.") and name.split(".")[-1] in ("manual_seed", "seed", "set_rng_state"):
                rep.evidence.append(Evidence(REVIEW, "tampering", "re-seeding the torch RNG (evaluator state)", node.lineno, scope))
            if name in _PRECISION_STATE_CALLS or (name.startswith("torch.backends.") and name.endswith((".flags", "set_flags"))):
                rep.evidence.append(Evidence(CONFIRMED, "tampering", f"call changing process-wide precision/backend state {name}",
                                             node.lineno, scope))
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                tname = canonical(_dotted(t), aliases) if isinstance(t, ast.Attribute) else None
                if tname and tname.startswith("torch.backends."):
                    rep.evidence.append(Evidence(CONFIRMED, "tampering", f"assignment to process-wide backend/precision state {tname}",
                                                 node.lineno, rep.scope_of(node.lineno)))
                elif isinstance(t, ast.Attribute) and (_dotted(t.value) or "").split(".")[0] in imported_modules:
                    rep.evidence.append(Evidence(REVIEW, "tampering", f"assignment into an imported module {tname} (monkey-patching)",
                                                 node.lineno, rep.scope_of(node.lineno)))

    # persistent state (data flow)
    flow = _Flow(tree, aliases, rep.kernel_lines)
    review_lines: set[int] = set()
    for w in flow.persistent_writes():
        if w["tensor"]:
            rep.evidence.append(Evidence(REVIEW, "cache", f"{w['kind']} `{w['target']}` holds tensor data or a tensor address/identity "
                                                          "across calls (possible cross-call result cache)", w["line"], "host"))
            review_lines.add(w["line"])
        elif w.get("value_kind") in ("kernel", "function"):
            rep.evidence.append(Evidence(AUDIT, "cache", f"{w['kind']} `{w['target']}` caches a compiled {w['value_kind']} "
                                                         "(code, not tensor data)" + (" keyed by a tensor address" if w.get("keyed_by_address") else ""),
                                         w["line"], "host"))
        elif not any(k in w["target"].lower() for k in ("config", "cfg")):
            rep.evidence.append(Evidence(AUDIT, "cache", f"{w['kind']} `{w['target']}` keeps non-tensor values across calls "
                                                         "(compiled-kernel / configuration cache)", w["line"], "host"))
    for key, line in flow.cache_decorated():
        tensor_calls = flow.calls_with_tensors(key)
        if tensor_calls:
            rep.evidence.append(Evidence(REVIEW, "cache", f"functools cache on {key} called with tensor data (lines {tensor_calls})",
                                         line, "host"))
        else:
            rep.evidence.append(Evidence(AUDIT, "cache", f"functools cache on {key} over non-tensor arguments", line, "host"))
    code = rep.code_text
    for m in re.finditer(r"\b(" + "|".join(_IDENTITY_DICTS) + r")\b", code):
        line = code.count("\n", 0, m.start()) + 1
        rep.evidence.append(Evidence(REVIEW, "cache", "tensor-identity dictionary", line, rep.scope_of(line), m.group(0)))
    for line in data_ptr_lines(tree):
        if line not in review_lines and line not in rep.kernel_lines:
            rep.evidence.append(Evidence(AUDIT, "cache", "tensor address (data_ptr) read; no persistent use found by the data-flow "
                                                         "analysis", line, "host", ".data_ptr()"))
    for line in flow.host_matmul_on_tensors():
        rep.evidence.append(Evidence(CONFIRMED, "delegation", "matrix-multiply operator `@` on tensors in host code (torch.matmul)",
                                     line, "host"))
    return rep


def data_ptr_lines(tree: ast.AST) -> list[int]:
    return sorted({n.lineno for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "data_ptr"})


def hidden_search_evidence(source: str, aliases: dict[str, str] | None = None) -> list[Evidence]:
    """Loops that iterate over several configurations and time them inside
    host code: high-risk (needs execution/human confirmation)."""
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
                out.append(Evidence(REVIEW, "autotune", "loop over configurations with timing calls", node.lineno, "host"))
    return out
