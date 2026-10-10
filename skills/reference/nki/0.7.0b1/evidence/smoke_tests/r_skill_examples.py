"""(r) Executes the code examples of SKILL.md verbatim.

Every ```python block in ../../SKILL.md whose first line is `# example: <name>` is
written unchanged to smoke_tests/skill_snippets/<name>.py, imported as a module
(the NKI frontend needs real source files), and run on the device against a CPU
reference. Two short fragments of SKILL.md (section 7.1 row reductions on a 2-D
tile, section 7.5 fp16 HBM vector -> fp32 SBUF operand) are checked by the kernels
at the bottom of this file, which use exactly those lines."""

import importlib
import os
import re
import sys

import _common as C
from _common import nki, nl, nisa, torch

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.normpath(os.path.join(HERE, "..", "..", "SKILL.md"))
SNIP_DIR = os.path.join(HERE, "skill_snippets")
os.makedirs(SNIP_DIR, exist_ok=True)

text = open(SKILL, encoding="utf-8").read()
blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
examples = {}
for b in blocks:
    m = re.match(r"# example: (\w+)\n", b)
    if m:
        examples[m.group(1)] = b
for name, code in examples.items():
    with open(os.path.join(SNIP_DIR, name + ".py"), "w", encoding="utf-8") as f:
        f.write(code)
sys.path.insert(0, SNIP_DIR)


def mod(name):
    return importlib.import_module(name)


torch.manual_seed(16)


def ex_add_kernel():
    a_c = torch.randn(256, 512, dtype=torch.float16)
    b_c = torch.randn(256, 512, dtype=torch.float16)
    out = mod("add_kernel").add_kernel(C.to_dev(a_c), C.to_dev(b_c))
    C.sync()
    ok, err = C.allclose(out, a_c.float() + b_c.float(), 1e-3, 1e-3)
    return {"ok": ok, "max_abs_err": err}


def ex_split_add():
    k = mod("split_add").split_add
    a_c = torch.randn(256, 512, dtype=torch.float16)
    b_c = torch.randn(256, 512, dtype=torch.float16)
    a, b = C.to_dev(a_c), C.to_dev(b_c)
    o2 = k(a, b)
    o1 = k[1](a, b)
    C.sync()
    ok2, e2 = C.allclose(o2, a_c.float() + b_c.float(), 1e-3, 1e-3)
    ok1, e1 = C.allclose(o1, a_c.float() + b_c.float(), 1e-3, 1e-3)
    return {"ok": ok1 and ok2, "bare_call_err": e2, "lnc1_err": e1}


def ex_barrier():
    out = mod("barrier_exchange").barrier_exchange(C.to_dev(torch.zeros(1, 1)))
    C.sync()
    o = out.cpu()
    ok = bool((o[:, 0, :] == 1.0).all() and (o[:, 1, :] == 2.0).all())
    return {"ok": ok}


def ex_any_shape():
    k = mod("add_one_any_shape").add_one_any_shape
    res = {}
    for shp in [(200, 300), (77, 513)]:
        x_c = torch.randn(*shp)
        out = k(C.to_dev(x_c))
        C.sync()
        res[str(shp)] = C.allclose(out, x_c + 1, 1e-6, 1e-6)[0]
    return {"ok": all(res.values()), **res}


def ex_fori():
    x_c = torch.randn(512, 256, dtype=torch.float32)
    out = mod("fori_rows").fori_rows(C.to_dev(x_c))
    C.sync()
    ok, err = C.allclose(out, (x_c + 1) * 2 - 1, 1e-5, 1e-5)
    return {"ok": ok, "max_abs_err": err}


def ex_colsum():
    x_c = torch.randn(128, 256, dtype=torch.float32)
    out = mod("colsum").colsum(C.to_dev(x_c))
    C.sync()
    ok, err = C.allclose(out, x_c.sum(dim=0, keepdim=True), 1e-3, 1e-2)
    return {"ok": ok, "max_abs_err": err}


def ex_matmul():
    k = mod("matmul_lhsT").matmul_lhsT
    res = {}
    for dt, out_dt, name in [(torch.bfloat16, nl.float32, "bf16->fp32"), (torch.float16, nl.float16, "fp16->fp16"),
                             (torch.float32, nl.float32, "fp32->fp32")]:
        lhsT = torch.randn(384, 128).to(dt)
        rhs = torch.randn(384, 512).to(dt)
        out = k(C.to_dev(lhsT), C.to_dev(rhs), out_dt)
        C.sync()
        ok, err = C.allclose(out, lhsT.float().T @ rhs.float(), 2e-2, 1e-1)
        res[name] = {"ok": ok, "max_abs_err": err}
    return {"ok": all(v["ok"] for v in res.values()), **res}


def ex_transpose():
    x_c = torch.randn(128, 64).to(torch.bfloat16)
    out = mod("transpose_tile").transpose_tile(C.to_dev(x_c))
    C.sync()
    ok = bool(torch.equal(out.cpu(), x_c.T.contiguous()))
    return {"ok": ok}


# ---- fragments of SKILL.md 7.1 and 7.5, used verbatim inside kernels -------------
@nki.jit
def frag_row_stats(x):
    P = 128
    out = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.shared_hbm)
    x_tile = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=x)
    m = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=m, op=nl.maximum, data=x_tile, axis=1)        # row max (x_tile: [P, F])
    s = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=s, op=nl.add, data=x_tile, axis=1)            # row sum
    nisa.dma_copy(dst=out[:, 0:1], src=m)
    nisa.dma_copy(dst=out[:, 1:2], src=s)
    return out


@nki.jit
def frag_row_scale(x, v_hbm):
    P = 128
    out = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    x_tile = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_tile, src=x)
    r = nl.ndarray(x.shape, dtype=nl.float32, buffer=nl.sbuf)
    v = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)            # MUST be fp32 for arithmetic ops
    nisa.dma_copy(dst=v, src=v_hbm)                                       # DMA converts dtype if v_hbm is fp16/bf16
    nisa.tensor_scalar(dst=r, data=x_tile, op0=nl.multiply, operand0=v, op1=nl.add, operand1=1.0)
    nisa.dma_copy(dst=out, src=r)
    return out


def frag_stats():
    x_c = torch.randn(128, 300)
    out = frag_row_stats(C.to_dev(x_c))
    C.sync()
    o = out.cpu()
    ok1 = torch.allclose(o[:, 0], x_c.amax(dim=1))
    ok2 = torch.allclose(o[:, 1], x_c.sum(dim=1), 1e-4, 1e-3)
    return {"ok": bool(ok1 and ok2)}


def frag_scale():
    x_c = torch.randn(128, 64)
    v_c = torch.randn(128, 1).to(torch.float16)
    out = frag_row_scale(C.to_dev(x_c), C.to_dev(v_c))
    C.sync()
    ok, err = C.allclose(out, x_c * v_c.float() + 1.0, 1e-5, 1e-5)
    return {"ok": ok, "max_abs_err": err}


RUNNERS = {
    "add_kernel": ex_add_kernel, "split_add": ex_split_add, "barrier_exchange": ex_barrier,
    "add_one_any_shape": ex_any_shape, "fori_rows": ex_fori, "colsum": ex_colsum,
    "matmul_lhsT": ex_matmul, "transpose_tile": ex_transpose,
}
missing = sorted(set(RUNNERS) - set(examples))
extra = sorted(set(examples) - set(RUNNERS))
C.run_case("example_inventory", "SKILL.md `# example:` blocks", "every example block has a runner and vice versa",
           lambda: {"ok": not missing and not extra, "examples": sorted(examples), "missing": missing, "unrunnable": extra})
for name, fn in RUNNERS.items():
    if name in examples:
        C.run_case(f"skill_example_{name}", f"SKILL.md example {name}", "verbatim SKILL.md code compiles, runs and matches CPU", fn)
C.run_case("skill_fragment_7_1_row_reductions", "SKILL.md 7.1 lines", "tensor_reduce max/sum with axis=1 on a 2-D tile", frag_stats)
C.run_case("skill_fragment_7_5_row_scale", "SKILL.md 7.5 lines", "fp16 HBM [P,1] -> fp32 SBUF via DMA, tensor_scalar with vector operand", frag_scale)
C.emit({"skill_md": SKILL, "examples_extracted": sorted(examples)})
