"""(l) Which Python constructs the default kernel frontend (NKI_COMPILER_FRONTEND
unset -> 'custom' = parser, a Python subset evaluated by NKI's partial evaluator)
accepts inside an @nki.jit body. Each construct is a separate tiny kernel that
computes out = x + <value> where <value> is built with the construct."""

import math

import _common as C
from _common import nki, nl, nisa, torch

# Observed on device (see smoke_results.json): these constructs are rejected.
EXPECT_ERROR = {"generator_expression", "list_comprehension", "lambda", "try_except",
                "nested_def_direct_call", "module_helper_keyword_only_arg"}


def _add_value(x, v):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=x)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=v)
    nisa.dma_copy(dst=out, src=t)
    return out


def helper_with_defaults(a, b=2, c=3):
    return a + b * c


def helper_kw_only(a, *, c=3):
    return a + c


@nki.jit
def k_module_helper(x):
    return _add_value(x, float(helper_with_defaults(1, c=3)))     # 7


@nki.jit
def k_module_helper_kwonly(x):
    return _add_value(x, float(helper_kw_only(1, c=3)))           # 4


@nki.jit
def k_tuple_unpack_shape(x):
    P, F = x.shape
    return _add_value(x, float(P // 128 + F % 7 + len(x.shape)))  # 1 + 1 + 2 = 4 (F=64)


@nki.jit
def k_list_comprehension(x):
    vals = [i * 2 for i in range(4)]
    return _add_value(x, float(vals[3]))                          # 6


@nki.jit
def k_generator_expression(x):
    v = sum(i for i in range(4))
    return _add_value(x, float(v))                                # 6


@nki.jit
def k_builtins(x):
    v = max(1, 5) + min(3, 4) + abs(-2) + int(2.7)
    return _add_value(x, float(v))                                # 12


@nki.jit
def k_math_module(x):
    return _add_value(x, float(math.ceil(2.1) + math.sqrt(16.0)))  # 7


@nki.jit
def k_python_while(x):
    i = 0
    s = 0
    while i < 4:
        s = s + i
        i = i + 1
    return _add_value(x, float(s))                                # 6


@nki.jit
def k_if_elif(x):
    F = x.shape[1]
    if F > 100:
        v = 1.0
    elif F == 64:
        v = 2.0
    else:
        v = 3.0
    return _add_value(x, v)                                       # 2


@nki.jit
def k_dict(x):
    d = {"a": 1.0, "b": 5.0}
    return _add_value(x, d["b"])                                  # 5


@nki.jit
def k_lambda(x):
    f = lambda z: z + 1.0  # noqa: E731
    return _add_value(x, f(1.0))                                  # 2


@nki.jit
def k_try_except(x):
    try:
        v = 1.0
    except ValueError:
        v = 2.0
    return _add_value(x, v)


@nki.jit
def k_is_not_none(x, opt=None):
    v = 1.0
    if opt is not None:
        v = 2.0
    return _add_value(x, v)                                       # 1


@nki.jit
def k_is_none(x, opt=None):
    v = 1.0
    if opt is None:
        v = 4.0
    return _add_value(x, v)                                       # 4


@nki.jit
def k_eq_none(x, opt=None):
    v = 1.0
    if opt != None:  # noqa: E711
        v = 2.0
    return _add_value(x, v)                                       # 1


@nki.jit
def k_fstring_assert(x):
    P = x.shape[0]
    assert P <= 128, f"partition dim {P} too large"
    return _add_value(x, 1.0)                                     # 1


@nki.jit
def k_fstring_value(x):
    s = f"{x.shape[0]}"
    return _add_value(x, float(len(s)))                           # 3


@nki.jit
def k_print(x):
    print("trace-time print", x.shape)
    return _add_value(x, 1.0)                                     # 1


@nki.jit
def k_nested_def(x):
    def inner(a):
        return a * 2.0
    return _add_value(x, inner(2.0))                              # 4


@nki.jit
def k_tile_size_props(x):
    v = nl.tile_size.sbuf_fmax_bytes // 1000 + nl.tile_size.psum_num_banks
    return _add_value(x, float(v))                                # 212 + 8 = 220


@nki.jit
def k_setitem(x):
    # t[...] = value  -> memset ; out[...] = sbuf_tile -> dma_copy
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.sbuf)
    t[...] = 9.0
    out[...] = t
    return out


@nki.jit
def sub_kernel_add(x, v):
    return _add_value(x, v)


@nki.jit
def k_nested_jit(x):
    # a @nki.jit kernel called from another kernel is inlined into the caller's trace
    return sub_kernel_add(x, 3.0)                                 # 3


@nki.jit
def k_memset_float_into_int(x):
    out = nl.ndarray(x.shape, dtype=nl.int32, buffer=nl.shared_hbm)
    t = nl.ndarray(x.shape, dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=t, value=2.0)
    nisa.dma_copy(dst=out, src=t)
    return out


@nki.jit
def k_multiple_returns(x):
    a = _add_value(x, 1.0)
    b = _add_value(x, 2.0)
    return a, b


CASES = [
    ("module_helper_defaults_and_keyword_call", k_module_helper, 7.0),
    ("module_helper_keyword_only_arg", k_module_helper_kwonly, 4.0),
    ("tuple_unpack_shape_int_ops", k_tuple_unpack_shape, 4.0),
    ("list_comprehension", k_list_comprehension, 6.0),
    ("generator_expression", k_generator_expression, 6.0),
    ("builtins_max_min_abs_int", k_builtins, 12.0),
    ("math_module", k_math_module, 7.0),
    ("python_while_int", k_python_while, 6.0),
    ("if_elif_else_on_shape", k_if_elif, 2.0),
    ("dict_literal", k_dict, 5.0),
    ("lambda", k_lambda, 2.0),
    ("try_except", k_try_except, 1.0),
    ("is_not_none_user_code", k_is_not_none, 1.0),
    ("is_none_user_code", k_is_none, 4.0),
    ("ne_none", k_eq_none, 1.0),
    ("fstring_in_assert", k_fstring_assert, 1.0),
    ("fstring_value", k_fstring_value, 3.0),
    ("print_in_kernel", k_print, 1.0),
    ("nested_def_direct_call", k_nested_def, 4.0),
    ("tile_size_properties_in_kernel", k_tile_size_props, 220.0),
    ("setitem_memset_and_store", k_setitem, 9.0),
    ("nested_jit_subkernel_call", k_nested_jit, 3.0),
    ("memset_float_value_into_int32_tile", k_memset_float_into_int, 2.0),
]

xs = torch.zeros(128, 64, dtype=torch.float32)
x = C.to_dev(xs)

for name, kern, expect in CASES:
    def f(kern=kern, expect=expect):
        out = kern(x)
        C.sync()
        v = float(out.cpu()[0, 0])
        return {"ok": v == expect, "value": v, "expected_value": expect}
    neg = name in EXPECT_ERROR
    C.run_case(name, "kernel body construct: " + name,
               "rejected by the default frontend (error text recorded)" if neg
               else "accepted by the default frontend and gives the expected value",
               f, expect_error=neg)


def multi_ret():
    a, b = k_multiple_returns(x)
    C.sync()
    va, vb = float(a.cpu()[0, 0]), float(b.cpu()[0, 0])
    return {"ok": (va, vb) == (1.0, 2.0), "values": [va, vb]}


C.run_case("multiple_return_tensors", "return a, b", "kernel returns a tuple of two shared_hbm tensors", multi_ret)
C.emit()
