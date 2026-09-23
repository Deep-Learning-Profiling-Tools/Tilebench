"""TileBench: controlled performance evaluation of tile-based programming models.

Submodules are imported explicitly (``from tilebench.core.engine import ...``);
this module deliberately exposes no symbols of its own so that importing the
package never pulls in torch, triton or the vendor backends.
"""

from tilebench.paths import (  # noqa: F401
    BENCHMARK_ROOT,
    DATA_ROOT,
    LLM_ROOT,
    OPERATOR_ROOT,
    PACKAGE_ROOT,
    PEAK_PERFORMANCE_ROOT,
    PROBLEMS_ROOT,
    PROFILING_ROOT,
    REPO_ROOT,
)
