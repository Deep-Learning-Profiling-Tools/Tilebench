"""Device-independent benchmark case identity for the TileArena paper-figure data packages.

The same rule is used for every device (B200, GH200, MI300X, Trainium2) so that the central
plotting phase can join cases across devices without relying on row order:

    operator    = normalize_operator(name)            lower case, '-' -> '_', stripped
    dtype       = normalize_dtype(name)               lower case, stripped; no aliasing, so
                                                      fp8_e4m3fn and fp8_e4m3fnuz stay distinct
    params_json = canonical_params_json(params)       json.dumps(sort_keys=True,
                                                      separators=(",", ":"), ensure_ascii=False)
    case_id     = sha256(f"{operator}|{dtype}|{params_json}".encode("utf-8")).hexdigest()

`params` is the input-parameter dictionary recorded in the `params` column of the benchmark
summary CSV (results/<device>/csv/<op>_<mode>.csv), i.e. the swept parameters of the case. It is
parsed with `parse_params_cell`, which splits on top-level commas only, so quoted strings,
lists, tuples and dicts that themselves contain commas survive.
"""
from __future__ import annotations

import ast
import hashlib
import json

__all__ = ["normalize_operator", "normalize_dtype", "parse_params_cell", "canonical_params_json", "case_id"]


def normalize_operator(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def normalize_dtype(name: str) -> str:
    # Deliberately no alias table: e4m3fn vs e4m3fnuz, bf16 vs fp16 etc. must never collapse.
    return name.strip().lower()


def _split_top_level(s: str, sep: str = ","):
    parts, depth, quote, buf = [], 0, None, []
    i = 0
    while i < len(s):
        ch = s[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(s):
                buf.append(s[i + 1]); i += 2; continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch; buf.append(ch)
        elif ch in "([{":
            depth += 1; buf.append(ch)
        elif ch in ")]}":
            depth -= 1; buf.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(buf)); buf = []
        else:
            buf.append(ch)
        i += 1
    if quote or depth:
        raise ValueError(f"unbalanced params cell: {s!r}")
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _parse_value(v: str):
    v = v.strip()
    try:
        return ast.literal_eval(v)  # ints, floats, quoted strings, lists, tuples, dicts, True/False/None
    except (ValueError, SyntaxError):
        return v  # bare token, kept verbatim as a string


def _jsonable(v):
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def parse_params_cell(cell: str) -> dict:
    """'cols=512, rows=512' -> {'cols': 512, 'rows': 512}. A JSON object cell is also accepted."""
    cell = cell.strip()
    if cell.startswith("{"):
        return _jsonable(json.loads(cell))
    out = {}
    for part in _split_top_level(cell):
        if "=" not in part:
            raise ValueError(f"params item without '=': {part!r} in {cell!r}")
        k, v = part.split("=", 1)
        k = k.strip()
        if k in out:
            raise ValueError(f"duplicate params key {k!r} in {cell!r}")
        out[k] = _jsonable(_parse_value(v))
    return out


def canonical_params_json(params: dict) -> str:
    return json.dumps(_jsonable(params), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def case_id(operator: str, dtype: str, params: dict) -> str:
    key = f"{normalize_operator(operator)}|{normalize_dtype(dtype)}|{canonical_params_json(params)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
