"""Canonical filesystem locations, resolved from this file rather than the CWD.

Every package resource (operator configs, peak-performance JSON, LLM prompt
inputs, the NCU catalogue) is addressed through these constants, so the
benchmark behaves identically no matter which directory it is started from.

``REPO_ROOT`` is the only entry that is not a package resource: results, logs
and generated artifacts live in the working tree next to the package. When
TileBench is installed outside a checkout (``pip install`` rather than
``pip install -e``), that checkout does not exist; ``TILEBENCH_REPO_ROOT``
overrides the guess, and callers that write results should honour their own
CLI arguments first.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

CORE_ROOT = PACKAGE_ROOT / "core"
DATA_ROOT = PACKAGE_ROOT / "data"
BENCHMARK_ROOT = PACKAGE_ROOT / "benchmarks"
OPERATOR_ROOT = BENCHMARK_ROOT / "operators"
LLM_GENERATED_ROOT = BENCHMARK_ROOT / "llm_generated"
PEAK_PERFORMANCE_ROOT = DATA_ROOT / "peak_performance"
PROFILING_ROOT = PACKAGE_ROOT / "profiling"
LLM_CODEGEN_ROOT = PACKAGE_ROOT / "llm_codegen"

#: NCU catalogue consumed by the profiling driver and the figure scripts.
NCU_CATALOGUE = PROFILING_ROOT / "ncu_catalogue.json"

#: Probed kernel launch counts and kernel names per (op, dtype, backend).
#: Canonical profiling metadata, not a report: the NCU harness validates every
#: capture against it. Regenerate with tilebench/profiling/probe_kernel_count.py.
KERNEL_COUNTS = PROFILING_ROOT / "kernel_counts.json"

REPO_ROOT = Path(os.environ.get("TILEBENCH_REPO_ROOT") or PACKAGE_ROOT.parent).resolve()

#: Generated artifacts (NCU reports, measured peak sweeps). Not package data.
OUTPUT_ROOT = REPO_ROOT / "outputs"
NCU_OUTPUT_ROOT = OUTPUT_ROOT / "ncu"


def operator_dir(operator: str) -> Path:
    """Directory holding one operator's config.yaml and impl_*.py files."""
    return OPERATOR_ROOT / operator


def operator_config(operator: str) -> Path:
    return operator_dir(operator) / "config.yaml"


def list_operators() -> list[str]:
    """Operator names, i.e. every directory under operators/ with a config.yaml."""
    if not OPERATOR_ROOT.is_dir():
        return []
    return sorted(d.name for d in OPERATOR_ROOT.iterdir()
                  if (d / "config.yaml").is_file())
