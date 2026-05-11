"""Static safety checks for LLM-generated kernel code.

Before writing generated code to disk or loading it into the process,
run these checks to catch obviously dangerous patterns (file I/O, shell
execution, network access, etc.) that should never appear in a GPU kernel.
"""

from __future__ import annotations

import ast
import re

# ---------------------------------------------------------------------------
# Forbidden literal snippets (fast substring scan).
# ---------------------------------------------------------------------------
_FORBIDDEN_SNIPPETS: list[tuple[str, str]] = [
    ("subprocess", "subprocess module is not allowed"),
    ("os.system", "os.system shell execution is not allowed"),
    ("os.popen", "os.popen shell execution is not allowed"),
    ("shutil.rmtree", "shutil.rmtree filesystem removal is not allowed"),
    ("requests.", "HTTP requests library is not allowed"),
    ("socket.", "socket module usage is not allowed"),
    ("import urllib", "urllib is not allowed"),
    ("__import__", "dynamic __import__ is not allowed"),
    ("eval(", "eval() is not allowed"),
    ("exec(", "exec() is not allowed"),
    ("compile(", "compile() is not allowed"),
    ("ctypes.", "ctypes module is not allowed"),
    ("cffi.", "cffi module is not allowed"),
]

# Allowed open() patterns: triton/cuda sometimes use open() for cache files;
# we flag bare open() calls that are NOT inside a known safe context.
# For strictness we ban all file I/O in kernel files.
_FORBIDDEN_FILE_IO: list[tuple[str, str]] = [
    (r"\bopen\s*\(", "open() file I/O is not allowed in kernel files"),
    (r"\bpathlib\b", "pathlib is not allowed in kernel files"),
]


def static_safety_check(code: str, strict_file_io: bool = True) -> None:
    """Raise :class:`ValueError` if *code* contains forbidden patterns.

    Parameters
    ----------
    code:
        Source code string to inspect.
    strict_file_io:
        When ``True`` (default), also reject ``open()`` and ``pathlib``
        usage.  Set to ``False`` if the target DSL legitimately needs
        file access (e.g. reading autotune cache).

    Raises
    ------
    ValueError
        On the first forbidden pattern detected.
    """
    for snippet, reason in _FORBIDDEN_SNIPPETS:
        if snippet in code:
            raise ValueError(f"Static safety check failed – {reason}")

    if strict_file_io:
        for pattern, reason in _FORBIDDEN_FILE_IO:
            if re.search(pattern, code):
                raise ValueError(f"Static safety check failed – {reason}")


def syntax_check(code: str) -> None:
    """Raise :class:`SyntaxError` if *code* does not parse as valid Python.

    Parameters
    ----------
    code:
        Source code string.

    Raises
    ------
    SyntaxError
        With line/column information from the Python parser.
    """
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise SyntaxError(
            f"Generated code has a syntax error at line {exc.lineno}: {exc.msg}"
        ) from exc


def validate_code(code: str, strict_file_io: bool = True) -> None:
    """Run all static validations in order: syntax → safety.

    Parameters
    ----------
    code:
        Source code string.
    strict_file_io:
        Passed through to :func:`static_safety_check`.

    Raises
    ------
    SyntaxError | ValueError
        On the first validation failure.
    """
    syntax_check(code)
    static_safety_check(code, strict_file_io=strict_file_io)
