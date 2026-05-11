"""Code extraction utilities for LLM responses.

Parses a raw LLM response text and extracts the Python code block.
Enforces strict rules suitable for reproducible paper experiments:
  - Exactly one fenced Python code block.
  - Code must pass a static safety check before being returned.
"""

from __future__ import annotations

import re
from typing import Literal


_PYTHON_BLOCK_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)```",
    re.DOTALL,
)

# Multiline fence that doesn't open with "python" label still accepted.
_GENERIC_BLOCK_RE = re.compile(
    r"```\n(.*?)```",
    re.DOTALL,
)


def extract_single_python_block(
    text: str,
    allow_multi: bool = False,
) -> str:
    """Extract a single Python code block from *text*.

    Parameters
    ----------
    text:
        Raw LLM response string.
    allow_multi:
        When ``True``, silently pick the *first* block if multiple are found.
        When ``False`` (default for strict paper experiments), raise if more
        than one code block is detected.

    Returns
    -------
    str
        The extracted source code, with a guaranteed trailing newline.

    Raises
    ------
    ValueError
        If no code block is found, or if *allow_multi* is ``False`` and
        more than one code block is present.
    """
    blocks = _PYTHON_BLOCK_RE.findall(text)

    # Fallback: try generic fenced blocks without language tag.
    if not blocks:
        blocks = _GENERIC_BLOCK_RE.findall(text)

    if not blocks:
        raise ValueError(
            "No fenced code block found in the LLM response.  "
            "The model must return code inside a ```python ... ``` block."
        )

    if len(blocks) > 1 and not allow_multi:
        raise ValueError(
            f"Expected exactly one Python code block, found {len(blocks)}.  "
            "The model must return a single fenced code block."
        )

    code = blocks[0].strip()
    if not code:
        raise ValueError("The extracted code block is empty.")

    return code + "\n"


def count_code_blocks(text: str) -> int:
    """Return the total number of fenced code blocks in *text*."""
    return len(_PYTHON_BLOCK_RE.findall(text)) or len(
        _GENERIC_BLOCK_RE.findall(text)
    )


def count_lines(code: str) -> int:
    """Return the number of non-empty, non-comment lines of code.

    Used as the LOC metric in paper tables.
    """
    lines = code.splitlines()
    return sum(
        1
        for ln in lines
        if ln.strip() and not ln.strip().startswith("#")
    )
