"""Static evidence collection on a generated file.

Static analysis can only PROVIDE EVIDENCE. The verdict levels are:

- confirmed  : unambiguous by construction (an autotuner decorator, a
               forbidden reference-library call that computes the operator,
               a subprocess / network / evaluator import). Triggers the
               same-round compliance repair.
- suspicious : patterns that need a human or execution-based decision
               (module-level caches keyed by tensor identity, loops that time
               several configurations, monkeypatching of torch.cuda).
               Verdict review_required.
- none       : no evidence found.

Operator-specific forbidden/required patterns come from the evaluator rules
of the contract (validation.contract_checks)."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

CONFIRMED = "confirmed"
SUSPICIOUS = "suspicious"

# Autotuning / hidden search entry points of each DSL.
_AUTOTUNE_ATTRS = {
    "triton": {("triton", "autotune"), ("triton", "testing", "do_bench"), ("triton", "testing", "do_bench_cudagraph")},
    "cutile": {("ct", "tune", "exhaustive_search"), ("cuda", "tile", "tune", "exhaustive_search"),
               ("ct_experimental", "autotune_launch"), ("cuda", "tile_experimental", "autotune_launch")},
    "tilelang": {("tilelang", "autotune"), ("tilelang", "autotuner", "AutoTuner"), ("set_autotune_inputs",)},
    "nki": set(),
}
_AUTOTUNE_NAMES = {"autotune", "do_bench", "exhaustive_search", "autotune_launch", "AutoTuner",
                   "set_autotune_inputs", "benchmark"}

# Modules a generated file must never import (evaluator, reference, I/O, process control).
_FORBIDDEN_MODULES = {
    "subprocess", "socket", "urllib", "requests", "http", "ctypes", "multiprocessing",
    "tilebench.llm", "tilebench.core.timer", "tilebench.core.verifier", "tilebench.core.engine",
    "tilebench.benchmarks", "impl_torch", "impl_triton", "impl_cutile", "impl_tilelang", "impl_nki",
    "triton.profiler", "proton",
}
_FORBIDDEN_MODULE_PREFIXES = ("tilebench.llm", "tilebench.benchmarks", "tilebench.profiling")

# torch calls that would delegate the operator's computation to a library.
_DELEGATION_CALLS = {
    "torch.nn.functional", "torch.matmul", "torch.mm", "torch.bmm", "torch.baddbmm", "torch.addmm",
    "torch.einsum", "torch.softmax", "torch.log_softmax", "torch.sort", "torch.argsort", "torch.topk",
    "torch.histc", "torch.bincount", "torch.cumsum", "torch.conv1d", "torch.conv2d", "torch.conv3d",
    "torch.nn.", "torch._scaled_mm", "torch.linalg", "torch.fft", "torch.scaled_dot_product_attention",
    "torch.layer_norm", "torch.rms_norm", "torch.batch_norm", "torch.kl_div", "torch.cross_entropy",
    "torch.max_pool2d", "torch.sigmoid", "torch.relu", "torch.nn.functional.",
}
_EVALUATOR_TAMPERING = {"torch.cuda.synchronize", "torch.cuda.Event", "torch.cuda.graph", "torch.cuda.CUDAGraph",
                        "setattr(torch", "torch.cuda.Stream", "time.perf_counter"}


@dataclass
class Evidence:
    level: str          # confirmed | suspicious
    category: str       # autotune | delegation | forbidden_import | cache | tampering | io | rule
    message: str
    line: int | None = None


@dataclass
class StaticReport:
    evidence: list[Evidence] = field(default_factory=list)

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

    def to_dict(self) -> dict:
        return {"verdict": self.verdict(), "evidence": [e.__dict__ for e in self.evidence]}


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def analyze(source: str, dsl: str, *, allowed_torch_calls: tuple[str, ...] = ()) -> StaticReport:
    rep = StaticReport()
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        rep.evidence.append(Evidence(SUSPICIOUS, "io", f"unparsable source: {e}", e.lineno))
        return rep

    # imports
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        for n in names:
            if n in _FORBIDDEN_MODULES or n.startswith(_FORBIDDEN_MODULE_PREFIXES) or n.split(".")[0] in _FORBIDDEN_MODULES:
                rep.evidence.append(Evidence(CONFIRMED, "forbidden_import", f"import of {n!r}", node.lineno))
            if n.split(".")[0] in ("os", "sys", "pathlib", "shutil", "importlib", "inspect"):
                rep.evidence.append(Evidence(SUSPICIOUS, "io", f"import of {n!r} (file/system access)", node.lineno))

    # decorators and calls
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                dotted = _dotted(target) or ""
                if dotted.split(".")[-1] in ("autotune",) or dotted in ("tilelang.autotune", "triton.autotune"):
                    rep.evidence.append(Evidence(CONFIRMED, "autotune", f"autotune decorator {dotted}", node.lineno))
        if isinstance(node, ast.Call):
            dotted = _dotted(node.func) or ""
            last = dotted.split(".")[-1]
            if last in _AUTOTUNE_NAMES or any(dotted.endswith(".".join(t)) for t in _AUTOTUNE_ATTRS.get(dsl, set())):
                rep.evidence.append(Evidence(CONFIRMED, "autotune", f"call to {dotted}", node.lineno))
            if dotted.startswith("torch.") and dotted not in allowed_torch_calls:
                if any(dotted == c or dotted.startswith(c) for c in _DELEGATION_CALLS):
                    rep.evidence.append(Evidence(CONFIRMED, "delegation", f"reference-library computation {dotted}", node.lineno))
            if any(dotted.startswith(t.rstrip("(")) for t in _EVALUATOR_TAMPERING if not t.startswith("setattr")):
                rep.evidence.append(Evidence(SUSPICIOUS, "tampering", f"timing/sync primitive {dotted}", node.lineno))
            if dotted in ("setattr", "eval", "exec", "compile", "open", "__import__"):
                rep.evidence.append(Evidence(SUSPICIOUS, "io", f"dynamic/IO call {dotted}", node.lineno))
            if dotted.startswith("torch.") and last in ("manual_seed", "seed"):
                rep.evidence.append(Evidence(SUSPICIOUS, "tampering", "re-seeding torch RNG", node.lineno))

    # module-level mutable caches keyed by tensors (output caching evidence)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, (ast.Dict, ast.List)) and len(getattr(value, "keys", getattr(value, "elts", []))) == 0:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        rep.evidence.append(Evidence(SUSPICIOUS, "cache", f"module-level mutable container {t.id!r} (possible result cache)", node.lineno))
    src_l = source
    for pat, msg in ((r"\.data_ptr\(\)", "tensor address used as a key"),
                     (r"WeakTensorKeyDictionary", "tensor-identity dictionary"),
                     (r"functools\.lru_cache|functools\.cache", "function result cache")):
        for m in re.finditer(pat, src_l):
            rep.evidence.append(Evidence(SUSPICIOUS, "cache", msg, src_l.count("\n", 0, m.start()) + 1))
    return rep


def hidden_search_evidence(source: str) -> list[Evidence]:
    """Loops that iterate over several configurations and time them inside
    run(): treated as suspicious (needs execution/human confirmation)."""
    out = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    timing_names = {"perf_counter", "time", "Event", "elapsed_time", "synchronize"}
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            body_src = ast.unparse(node)
            if any(t in body_src for t in timing_names) and ("config" in body_src.lower() or "BLOCK" in body_src):
                out.append(Evidence(SUSPICIOUS, "autotune", "loop over configurations with timing calls", node.lineno))
    return out
