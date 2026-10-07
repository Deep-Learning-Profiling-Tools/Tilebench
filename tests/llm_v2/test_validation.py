"""Strict parser, static evidence and contract checks."""
import pytest

from tilebench.llm.v2.validation.parser import FormatError, parse_single_file
from tilebench.llm.v2.validation.static_checks import analyze
from tilebench.llm.v2.validation.contract_checks import check_compliance

OK = '```python title="impl_triton.py"\nimport triton\ndef run(x):\n    return x\n```'


def test_parser_accepts_exactly_one_titled_block():
    p = parse_single_file(OK, "impl_triton.py")
    assert p.filename == "impl_triton.py" and p.source.startswith("import triton")


@pytest.mark.parametrize("text,msg", [
    ("", "empty"), ("no code here", "no fenced"),
    (OK + "\n" + OK, "exactly one"),
    ('```python\ndef run(): pass\n```', "title="),
    ('```python title="impl_cutile.py"\ndef run(): pass\n```', "!= requested"),
    ('```python title="impl_triton.py"\ndef run(:\n```', "not valid Python"),
])
def test_parser_rejects(text, msg):
    with pytest.raises(FormatError, match=msg):
        parse_single_file(text, "impl_triton.py")


def test_autotune_decorator_is_confirmed():
    src = "import triton\n@triton.autotune(configs=[], key=['N'])\n@triton.jit\ndef k(): pass\n"
    rep = analyze(src, "triton")
    assert rep.verdict() == "confirmed_violation" and any(e.category == "autotune" for e in rep.confirmed)


def test_tilelang_autotune_and_cutile_search_are_confirmed():
    assert analyze("import tilelang\n@tilelang.autotune(configs=[])\ndef k(): pass\n", "tilelang").verdict() == "confirmed_violation"
    assert analyze("import cuda.tile as ct\nr = ct.tune.exhaustive_search([], None)\n", "cutile").verdict() == "confirmed_violation"


def test_delegation_to_torch_is_confirmed_unless_allowed():
    src = "import torch\ndef run(a, b):\n    return torch.matmul(a, b)\n"
    assert analyze(src, "triton").verdict() == "confirmed_violation"
    src2 = "import torch\ndef run(a):\n    out = torch.empty_like(a)\n    return out\n"
    assert analyze(src2, "triton", allowed_torch_calls=("torch.empty_like",)).verdict() == "clear"


def test_forbidden_imports_and_suspicious_cache():
    rep = analyze("import subprocess\nfrom tilebench.llm import evaluator\n", "triton")
    assert len(rep.confirmed) >= 2
    # checker v2: a local / read-only use of a tensor address is audit-only; writing tensor identity or data into
    # persistent state is high-risk (review_required)
    rep2 = analyze("_cache = {}\ndef run(x):\n    k = x.data_ptr()\n    return _cache.get(k)\n", "triton")
    assert rep2.verdict() == "audit_only" and any(e.category == "cache" for e in rep2.audit)
    rep3 = analyze("_cache = {}\ndef run(x):\n    k = x.data_ptr()\n    _cache[k] = x\n    return x\n", "triton")
    assert rep3.verdict() == "review_required" and any(e.category == "cache" for e in rep3.review)


def test_contract_rules_add_evidence():
    rules = {"forbidden_substitutions": [{"pattern": r"torch\.sort", "message": "sort delegated", "level": "confirmed"}],
             "required_evidence": [{"any_of": [r"tl\.load"], "message": "tile load", "level": "suspicious"}],
             "allowed_torch_calls": ["torch.empty_like"]}
    bad = check_compliance("import torch\ndef run(x):\n    return torch.sort(x)[0]\n", "triton", rules)
    assert bad.verdict == "confirmed_violation" and any("sort delegated" in d for d in bad.diagnostics())
    weak = check_compliance("import triton\ndef run(x):\n    return x\n", "triton", rules)
    assert weak.verdict == "audit_only" and any("missing evidence: tile load" in f for f in weak.audit_flags())
    good = check_compliance("import triton\nimport triton.language as tl\n@triton.jit\ndef k(p):\n    tl.load(p)\ndef run(x):\n    return x\n", "triton", rules)
    assert good.verdict == "clear"
