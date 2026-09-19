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
import re
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

#: Benchmark results, one namespace per hardware campaign:
#:     results/<hardware>/{csv,logs,figures,aggregate,runs}/
#: The GPU backends in a namespace were measured on that hardware. NKI (AWS
#: Trainium) measurements are recorded alongside, in their own columns and under
#: logs/nki_profiles/, as cross-hardware data. Only csv/ is version-controlled.
#: Build paths with the helpers below, never by hand.
RESULTS_ROOT = REPO_ROOT / "results"

_HARDWARE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


def hardware_label(label: str) -> str:
    """Validate a hardware label (B200, GH200, MI300X, ...) as one safe path component.

    There is deliberately no list of supported devices: a new platform needs a
    new label, not a code change. Usable as an argparse ``type=``.
    """
    if not isinstance(label, str) or not _HARDWARE_LABEL.fullmatch(label):
        raise ValueError(
            f"invalid hardware label {label!r}: use one path component made of letters, "
            f"digits, '.', '_', '+' or '-', starting with a letter or digit (e.g. B200)")
    return label


def results_root(hardware: str) -> Path:
    return RESULTS_ROOT / hardware_label(hardware)


def results_csv_dir(hardware: str) -> Path:
    """Summary CSVs: the only version-controlled results."""
    return results_root(hardware) / "csv"


def results_logs_dir(hardware: str) -> Path:
    return results_root(hardware) / "logs"


def results_figures_dir(hardware: str) -> Path:
    return results_root(hardware) / "figures"


def results_aggregate_dir(hardware: str) -> Path:
    return results_root(hardware) / "aggregate"


def results_runs_dir(hardware: str) -> Path:
    return results_root(hardware) / "runs"


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
