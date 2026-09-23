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

from tilebench.backends import MODES, backend_tag

PACKAGE_ROOT = Path(__file__).resolve().parent

CORE_ROOT = PACKAGE_ROOT / "core"
DATA_ROOT = PACKAGE_ROOT / "data"
BENCHMARK_ROOT = PACKAGE_ROOT / "benchmarks"
OPERATOR_ROOT = BENCHMARK_ROOT / "operators"
LLM_GENERATED_ROOT = BENCHMARK_ROOT / "llm_generated"
PEAK_PERFORMANCE_ROOT = DATA_ROOT / "peak_performance"
PROFILING_ROOT = PACKAGE_ROOT / "profiling"
LLM_ROOT = PACKAGE_ROOT / "llm"
#: Task descriptions, one <operator>_current.md per operator: the problem
#: statement the LLM pipeline puts in its prompts.
PROBLEMS_ROOT = PACKAGE_ROOT / "problems"

REPO_ROOT = Path(os.environ.get("TILEBENCH_REPO_ROOT") or PACKAGE_ROOT.parent).resolve()

#: Generated artifacts (NCU reports, measured peak sweeps). Not package data.
OUTPUT_ROOT = REPO_ROOT / "outputs"
#: Generated NCU reports, one directory per hardware: outputs/ncu/<hardware>/.
NCU_OUTPUT_ROOT = OUTPUT_ROOT / "ncu"
#: NCU profiling metadata, one directory per hardware (see the helpers below):
#:     outputs/profiling/<hardware>/{ncu_catalogue.json,kernel_counts.json}
#: Autotune winners, kernel launch counts and kernel names are measured on one
#: GPU, so this is generated experiment data, not a package resource: it lives
#: with the other generated outputs, there is no global copy, and another GPU's
#: files are never used as a fallback.
PROFILING_METADATA_ROOT = OUTPUT_ROOT / "profiling"

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


def _log_name(operator: str, mode: str, backends) -> str:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
    return f"{operator}_{mode}_{backend_tag(backends)}.json"


def timing_log_path(hardware: str, operator: str, mode: str, backends) -> Path:
    """Raw timing JSON of one run:
        results/<hardware>/logs/time_measurement_logs/<operator>_<mode>_<backend-tag>.json
    The name carries the mode and the backend selection, so a default run, an
    autotune run, a TileLang-only run and an NKI run of the same operator never
    overwrite each other. `backends` may be in any order."""
    return results_logs_dir(hardware) / "time_measurement_logs" / _log_name(operator, mode, backends)


def autotune_log_path(hardware: str, operator: str, mode: str, backends) -> Path:
    """Selected-config log of one run; same naming rule as timing_log_path."""
    return results_logs_dir(hardware) / "autotune_logs" / _log_name(operator, mode, backends)


def profiling_metadata_dir(hardware: str) -> Path:
    return PROFILING_METADATA_ROOT / hardware_label(hardware)


def ncu_catalogue_path(hardware: str) -> Path:
    """Sweep-max cases and autotune winners profiled by NCU on this hardware."""
    return profiling_metadata_dir(hardware) / "ncu_catalogue.json"


def kernel_counts_path(hardware: str) -> Path:
    """Probed kernel launch counts and kernel names per (op, dtype, backend) on
    this hardware. The NCU harness validates every capture against it."""
    return profiling_metadata_dir(hardware) / "kernel_counts.json"


def ncu_output_dir(hardware: str) -> Path:
    """Generated NCU reports of this hardware: outputs/ncu/<hardware>/."""
    return NCU_OUTPUT_ROOT / hardware_label(hardware)


def ncu_report_path(hardware: str, operator: str, backend: str, dtype: str) -> Path:
    """One NCU report: outputs/ncu/<hardware>/<operator>/<backend>_<dtype>.ncu-rep.
    The single rule shared by the sweep driver and the one-operator tool, so
    the reports of two GPUs can never collide."""
    return ncu_output_dir(hardware) / operator / f"{backend}_{dtype}.ncu-rep"


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
