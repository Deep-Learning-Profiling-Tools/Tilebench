"""Tile-language backend selection, shared by every entry point.

One definition of the backend names, their canonical order and how a
``--tile-language`` value is parsed, so that a selection always resolves to the
same set, and to the same file names (see ``tilebench.paths.timing_log_path``),
whatever order the user typed it in. torch is the implicit baseline and is never
part of a selection.
"""

from __future__ import annotations

#: Canonical order: drives CSV column order and the backend tag in file names.
BACKEND_ORDER = ("triton", "cutile", "tilelang", "nki")

#: Backends measured on the GPU itself: the default selection, and what ``all``
#: means. NKI runs on AWS Trainium and only when it is named explicitly.
GPU_BACKENDS = ("triton", "cutile", "tilelang")

#: Run modes, as they appear in result file names.
MODES = ("default", "autotune")


def canonical_backends(backends) -> list[str]:
    """The selection as a duplicate-free list in canonical order."""
    chosen = set(backends)
    unknown = sorted(chosen - set(BACKEND_ORDER))
    if unknown:
        raise ValueError(f"unknown backend(s) {unknown}; choose from {', '.join(BACKEND_ORDER)}")
    return [b for b in BACKEND_ORDER if b in chosen]


def parse_backends(text: str | None) -> list[str]:
    """Parse a ``--tile-language`` value. None or ``all`` selects the GPU
    backends; ``torch`` is accepted and ignored because it always runs."""
    if text is None:
        return list(GPU_BACKENDS)
    tokens = [t.strip().lower() for t in text.split(",") if t.strip()]
    if "all" in tokens:
        return list(GPU_BACKENDS)
    return canonical_backends(t for t in tokens if t != "torch")


def backend_tag(backends) -> str:
    """File-name tag of a selection: ``triton-cutile``, ``tilelang``, ``nki``...
    Order-independent, so ``cutile,triton`` and ``triton,cutile`` share one tag.
    A torch-only run is tagged ``torch``."""
    return "-".join(canonical_backends(backends)) or "torch"
