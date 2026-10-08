"""Strict single-file response parser.

A valid response contains exactly ONE fenced Python block whose info string
carries the requested file name, e.g.

    ```python title="impl_triton.py"
    ...
    ```

Anything else (zero blocks, several blocks, a different title, an untitled
block) is a FormatError. A format error is an ordinary failure: it consumes
the round and is reported as a diagnostic; it never triggers a same-round
repair."""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)\n```", re.S)
_TITLE = re.compile(r'title\s*=\s*"([^"]+)"')


class FormatError(ValueError):
    pass


@dataclass
class ParsedResponse:
    filename: str
    source: str
    n_blocks: int


def parse_single_file(text: str | None, expected_filename: str) -> ParsedResponse:
    if not text or not text.strip():
        raise FormatError("empty response")
    blocks = _FENCE.findall(text)
    if not blocks:
        raise FormatError("no fenced code block found")
    titled = []
    for info, body in blocks:
        m = _TITLE.search(info)
        titled.append((m.group(1).strip() if m else None, info.strip(), body))
    code_blocks = [b for b in titled if b[1].split()[:1] in (["python"], ["py"]) or b[0]]
    if len(blocks) != 1:
        names = [t[0] or "<untitled>" for t in titled]
        raise FormatError(f"expected exactly one fenced block, found {len(blocks)}: {names}")
    title, info, body = titled[0]
    if title is None:
        raise FormatError(f'the code block must carry title="{expected_filename}"')
    if title != expected_filename:
        raise FormatError(f"code block title {title!r} != requested file {expected_filename!r}")
    if not info.startswith("python"):
        raise FormatError("the code block must be a ```python block")
    try:
        ast.parse(body)
    except SyntaxError as e:
        raise FormatError(f"{expected_filename} is not valid Python: {e.msg} (line {e.lineno})") from e
    return ParsedResponse(filename=expected_filename, source=body + ("\n" if not body.endswith("\n") else ""),
                          n_blocks=len(code_blocks))
