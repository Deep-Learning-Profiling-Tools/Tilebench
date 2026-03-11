"""pytest unit tests for metrics reporting correctness.

Two levels of tests:

  1. Unit tests for core/metrics.py helpers (_dtype_bytes, _eval_expr,
     compute_derived formula correctness, edge-case guard-rails).

  2. Operator-specific integration tests that load each operator's actual
     config.yaml and verify that metrics are computed correctly for every
     dtype the operator declares.  These catch:
       • wrong bytes_expr / flops_expr in the config
       • wrong dtype_size lookup (e.g. bf16 → 4 instead of 2)
       • unit conversion bugs (GB/s vs MB/s, TFLOPS vs GFLOPS)
       • wrong peak ceiling used per dtype

Run from the Tilebench/ directory:
    PYTHONPATH=. pytest tests/test_metrics.py -v
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from core.metrics import _dtype_bytes, _eval_expr, compute_derived

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent  # Tilebench/


def _load_operator_config(operator_name: str) -> dict:
    path = _REPO_ROOT / "benchmarks" / "operators" / operator_name / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _make_result(
    params: dict,
    dtype: str,
    problem_size: int,
    torch_ms: float = 1.0,
    triton_ms: float = 0.5,
    cutile_ms: float = 0.25,
) -> dict:
    """Minimal benchmark result dict that mirrors engine.py output."""
    return {
        "params": params,
        "problem_size": problem_size,
        "dtype": dtype,
        "torch_ms": torch_ms,
        "triton_ms": triton_ms,
        "cutile_ms": cutile_ms,
    }


def _expected_bw(bytes_expr: str, n: int, dtype_size: int, ms: float) -> float:
    """Reference bandwidth computation, independent of core/metrics.py."""
    ctx = {"n": n, "dtype_size": dtype_size}
    bytes_val = eval(bytes_expr, {"__builtins__": {}, "math": math}, ctx)  # noqa: S307
    return bytes_val / (ms * 1e-3) / 1e9


def _expected_tflops(flops_expr: str, n: int, dtype_size: int, ms: float) -> float:
    ctx = {"n": n, "dtype_size": dtype_size}
    flops = eval(flops_expr, {"__builtins__": {}, "math": math}, ctx)  # noqa: S307
    return flops / (ms * 1e-3) / 1e12


# ===========================================================================
# 1.  _dtype_bytes – the foundation of every per-dtype metric
# ===========================================================================

class TestDtypeBytes:
    """_dtype_bytes underpins all bandwidth/AI calculations.
    If this mapping is wrong, every metric for that dtype is wrong.
    """

    @pytest.mark.parametrize("name,expected_bytes", [
        ("fp16",     2),
        ("bf16",     2),
        ("fp32",     4),
        ("int8",     1),
        ("fp8",      1),
        ("fp8_e5m2", 1),
        ("int32",    4),
        ("int64",    8),
    ])
    def test_dtype_byte_sizes(self, name: str, expected_bytes: int):
        assert _dtype_bytes(name) == expected_bytes, (
            f"_dtype_bytes('{name}') should be {expected_bytes} bytes per element"
        )

    def test_unknown_dtype_defaults_to_4(self):
        assert _dtype_bytes("mystery_type") == 4

    def test_lookup_is_case_insensitive(self):
        for name in ("FP16", "BF16", "INT8", "FP32"):
            assert _dtype_bytes(name) == _dtype_bytes(name.lower())


# ===========================================================================
# 2.  _eval_expr – the safe eval that turns config strings into numbers
# ===========================================================================

class TestEvalExpr:
    """_eval_expr bridges config.yaml expression strings → float values."""

    CTX = {"n": 1024 * 1024, "dtype_size": 2}  # 1M fp16 elements

    def test_mul2_bytes_expr(self):
        # "n * dtype_size * 2" with n=1M, dtype_size=2 → 4 MB
        assert _eval_expr("n * dtype_size * 2", self.CTX) == pytest.approx(4 * 1024 * 1024)

    def test_flops_proxy_expr(self):
        # flops_expr: "n" → element count as proxy
        assert _eval_expr("n", self.CTX) == pytest.approx(1024 * 1024)

    def test_none_returns_none(self):
        assert _eval_expr(None, self.CTX) is None

    @pytest.mark.parametrize("sentinel", ["", "null", "None", "none"])
    def test_empty_or_null_returns_none(self, sentinel):
        assert _eval_expr(sentinel, self.CTX) is None

    def test_invalid_expr_returns_none_not_exception(self):
        assert _eval_expr("os.system('rm -rf /')", self.CTX) is None

    def test_syntax_error_returns_none_not_exception(self):
        assert _eval_expr("n ** ** 2", self.CTX) is None

    def test_result_is_always_float(self):
        result = _eval_expr("n", self.CTX)
        assert isinstance(result, float)


# ===========================================================================
# 3.  mul2 – operator-specific metric tests using the real config.yaml
# ===========================================================================

@pytest.fixture(scope="module")
def mul2_cfg():
    return _load_operator_config("mul2")


@pytest.fixture(scope="module")
def mul2_metrics(mul2_cfg):
    return mul2_cfg["metrics"]


@pytest.fixture(scope="module")
def mul2_dtypes(mul2_cfg):
    return mul2_cfg["case_grid"]["dtype"]


class TestMul2Metrics:
    """Verify that compute_derived produces correct metrics for mul2's actual
    config and every dtype declared in its case_grid.
    mul2 params: n (element count). bytes_expr = "n * dtype_size * 2".
    """

    # A fixed problem size used across all sub-tests.
    N = 4 * 1024 * 1024   # 4M elements
    TORCH_MS = 1.0
    TRITON_MS = 0.4
    CUTILE_MS = 0.25

    def _result(self, dtype: str) -> dict:
        return _make_result(
            params={"n": self.N},
            dtype=dtype,
            problem_size=self.N,
            torch_ms=self.TORCH_MS,
            triton_ms=self.TRITON_MS,
            cutile_ms=self.CUTILE_MS,
        )

    @pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "int8"])
    def test_bandwidth_per_dtype(self, dtype: str, mul2_metrics: dict):
        """bandwidth_GBs must be computed with the correct dtype_size for each dtype."""
        ds = _dtype_bytes(dtype)
        expected = _expected_bw(
            mul2_metrics["bytes_expr"], self.N, ds, self.TORCH_MS
        )
        derived = compute_derived(self._result(dtype), mul2_metrics)
        assert derived["torch"]["bandwidth_GBs"] == pytest.approx(expected, rel=1e-6), (
            f"mul2 torch bandwidth wrong for dtype={dtype} "
            f"(dtype_size={ds}): got {derived['torch']['bandwidth_GBs']:.3f}, "
            f"expected {expected:.3f} GB/s"
        )

    @pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "int8"])
    def test_pct_peak_bw_per_dtype(self, dtype: str, mul2_metrics: dict):
        """pct_peak_bw = bandwidth_GBs / peak_bw_GBs * 100, per dtype."""
        ds = _dtype_bytes(dtype)
        bw = _expected_bw(mul2_metrics["bytes_expr"], self.N, ds, self.TORCH_MS)
        expected_pct = bw / mul2_metrics["peak_bw_GBs"] * 100.0
        derived = compute_derived(self._result(dtype), mul2_metrics)
        assert derived["torch"]["pct_peak_bw"] == pytest.approx(expected_pct, rel=1e-5)

    @pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "int8"])
    def test_peak_tflops_ceiling_per_dtype(self, dtype: str, mul2_metrics: dict):
        """pct_peak_tflops must use the ceiling for this dtype, not another dtype's."""
        peak_tflops_map = mul2_metrics["peak_tflops"]
        if dtype not in peak_tflops_map:
            pytest.skip(f"{dtype} not in mul2 peak_tflops map")
        ds = _dtype_bytes(dtype)
        tf = _expected_tflops(mul2_metrics["flops_expr"], self.N, ds, self.TRITON_MS)
        expected_pct = tf / peak_tflops_map[dtype] * 100.0
        derived = compute_derived(self._result(dtype), mul2_metrics)
        assert derived["triton"]["pct_peak_tflops"] == pytest.approx(expected_pct, rel=1e-5), (
            f"mul2 triton pct_peak_tflops wrong for dtype={dtype}: "
            f"expected to use peak={peak_tflops_map[dtype]} TFLOPS"
        )

    @pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "int8"])
    def test_triton_speedup_gt1_when_faster(self, dtype: str, mul2_metrics: dict):
        """speedup > 1 when triton_ms < torch_ms."""
        derived = compute_derived(self._result(dtype), mul2_metrics)
        assert derived["triton"]["speedup"] > 1.0, (
            f"Expected triton speedup > 1 for dtype={dtype} "
            f"(torch={self.TORCH_MS}ms, triton={self.TRITON_MS}ms)"
        )

    def test_torch_has_no_speedup(self, mul2_metrics: dict):
        derived = compute_derived(self._result("fp32"), mul2_metrics)
        assert "speedup" not in derived["torch"]

    def test_all_dtypes_declared_in_config(self, mul2_dtypes: list):
        """Sanity check: the four standard dtypes are all in mul2's config."""
        for dtype in ("fp16", "bf16", "fp32", "int8"):
            assert dtype in mul2_dtypes, f"Expected dtype '{dtype}' in mul2 case_grid"


# ===========================================================================
# 4.  destindex – operator-specific metric tests using the real config.yaml
# ===========================================================================

@pytest.fixture(scope="module")
def destindex_cfg():
    return _load_operator_config("destindex")


@pytest.fixture(scope="module")
def destindex_metrics(destindex_cfg):
    return destindex_cfg["metrics"]


@pytest.fixture(scope="module")
def destindex_dtypes(destindex_cfg):
    return destindex_cfg["case_grid"]["dtype"]


class TestDestindexMetrics:
    """Verify metrics for destindex, which uses seq_len (not n) as its sweep
    parameter.  problem_size = seq_len * 1600 is used as 'n' in expressions.
    """

    # Realistic MLA-typical shapes from destindex config defaults
    SEQ_LEN = 1024
    # n = seq_len * (kv_nope_head_num*kv_nope_head_dim + kv_rope_head_num*kv_rope_head_dim)
    #   = 1024 * (12*128 + 1*64) = 1024 * 1600
    N = SEQ_LEN * 1600
    TORCH_MS = 2.0
    TRITON_MS = 0.8
    CUTILE_MS = 0.4

    def _result(self, dtype: str) -> dict:
        return _make_result(
            params={"seq_len": self.SEQ_LEN},   # note: NOT "n"
            dtype=dtype,
            problem_size=self.N,                # engine.py sets this
            torch_ms=self.TORCH_MS,
            triton_ms=self.TRITON_MS,
            cutile_ms=self.CUTILE_MS,
        )

    @pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "int8"])
    def test_bandwidth_uses_problem_size_not_seq_len(
        self, dtype: str, destindex_metrics: dict
    ):
        """n must come from problem_size (= total elements), not from seq_len."""
        ds = _dtype_bytes(dtype)
        # Using N (total elements), NOT seq_len
        expected = _expected_bw(
            destindex_metrics["bytes_expr"], self.N, ds, self.TORCH_MS
        )
        derived = compute_derived(self._result(dtype), destindex_metrics)
        assert derived["torch"]["bandwidth_GBs"] == pytest.approx(expected, rel=1e-6), (
            f"destindex bandwidth wrong for dtype={dtype}: "
            f"n should be problem_size={self.N}, not seq_len={self.SEQ_LEN}"
        )

    @pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "int8"])
    def test_pct_peak_bw_per_dtype(self, dtype: str, destindex_metrics: dict):
        ds = _dtype_bytes(dtype)
        bw = _expected_bw(destindex_metrics["bytes_expr"], self.N, ds, self.TORCH_MS)
        expected_pct = bw / destindex_metrics["peak_bw_GBs"] * 100.0
        derived = compute_derived(self._result(dtype), destindex_metrics)
        assert derived["torch"]["pct_peak_bw"] == pytest.approx(expected_pct, rel=1e-5)

    @pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "int8"])
    def test_peak_tflops_ceiling_per_dtype(self, dtype: str, destindex_metrics: dict):
        peak_tflops_map = destindex_metrics["peak_tflops"]
        if dtype not in peak_tflops_map:
            pytest.skip(f"{dtype} not in destindex peak_tflops map")
        ds = _dtype_bytes(dtype)
        tf = _expected_tflops(
            destindex_metrics["flops_expr"], self.N, ds, self.TRITON_MS
        )
        expected_pct = tf / peak_tflops_map[dtype] * 100.0
        derived = compute_derived(self._result(dtype), destindex_metrics)
        assert derived["triton"]["pct_peak_tflops"] == pytest.approx(expected_pct, rel=1e-5), (
            f"destindex triton pct_peak_tflops wrong for dtype={dtype}"
        )

    @pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32", "int8"])
    def test_triton_speedup_gt1_when_faster(self, dtype: str, destindex_metrics: dict):
        derived = compute_derived(self._result(dtype), destindex_metrics)
        assert derived["triton"]["speedup"] > 1.0

    def test_all_dtypes_declared_in_config(self, destindex_dtypes: list):
        for dtype in ("fp16", "bf16", "fp32", "int8"):
            assert dtype in destindex_dtypes, (
                f"Expected dtype '{dtype}' in destindex case_grid"
            )

    def test_fp16_bandwidth_gt_fp32_bandwidth(self, destindex_metrics: dict):
        """fp16 (2 bytes) transfers less data per element than fp32 (4 bytes),
        so given the same latency, fp16 should report lower bandwidth than fp32."""
        d16 = compute_derived(
            _make_result({"seq_len": self.SEQ_LEN}, "fp16", self.N, torch_ms=self.TORCH_MS),
            destindex_metrics,
        )
        d32 = compute_derived(
            _make_result({"seq_len": self.SEQ_LEN}, "fp32", self.N, torch_ms=self.TORCH_MS),
            destindex_metrics,
        )
        assert d16["torch"]["bandwidth_GBs"] == pytest.approx(
            d32["torch"]["bandwidth_GBs"] / 2, rel=1e-5
        ), "fp16 bandwidth should be half of fp32 bandwidth (same n, same latency)"


# ===========================================================================
# 5.  Speedup formula invariants (cross-operator)
# ===========================================================================

class TestSpeedupInvariants:
    """speedup = torch_ms / backend_ms.  These hold for any operator."""

    _CFG = {"flops_expr": "n", "bytes_expr": "n * dtype_size * 2", "peak_bw_GBs": 8000.0}

    def _r(self, torch_ms, triton_ms):
        return _make_result({"n": 1024}, "fp32", 1024, torch_ms, triton_ms)

    def test_triton_2x_faster_yields_speedup_2(self):
        derived = compute_derived(self._r(2.0, 1.0), self._CFG)
        assert derived["triton"]["speedup"] == pytest.approx(2.0)

    def test_triton_2x_slower_yields_speedup_0_5(self):
        derived = compute_derived(self._r(1.0, 2.0), self._CFG)
        assert derived["triton"]["speedup"] == pytest.approx(0.5)

    def test_speedup_equals_1_when_latency_equal(self):
        derived = compute_derived(self._r(1.5, 1.5), self._CFG)
        assert derived["triton"]["speedup"] == pytest.approx(1.0)

    def test_torch_backend_never_has_speedup(self):
        derived = compute_derived(self._r(1.0, 0.5), self._CFG)
        assert "speedup" not in derived["torch"]


# ===========================================================================
# 6.  Guard-rails: bad latency values must not produce silent wrong metrics
# ===========================================================================

class TestBadLatencyGuardRails:

    _CFG = {"flops_expr": "n", "bytes_expr": "n * dtype_size * 2", "peak_bw_GBs": 8000.0}
    _BASE = {"params": {"n": 1024}, "problem_size": 1024, "dtype": "fp32"}

    def test_zero_latency_backend_excluded(self):
        result = {**self._BASE, "torch_ms": 0.0, "triton_ms": 0.5}
        derived = compute_derived(result, self._CFG)
        assert "torch" not in derived

    def test_nan_latency_backend_excluded(self):
        result = {**self._BASE, "torch_ms": 1.0, "triton_ms": float("nan")}
        derived = compute_derived(result, self._CFG)
        assert "triton" not in derived

    def test_negative_latency_backend_excluded(self):
        result = {**self._BASE, "torch_ms": 1.0, "cutile_ms": -0.5}
        derived = compute_derived(result, self._CFG)
        assert "cutile" not in derived

    def test_latency_ms_always_present_when_valid(self):
        result = {**self._BASE, "torch_ms": 1.23}
        derived = compute_derived(result, self._CFG)
        assert derived["torch"]["latency_ms"] == pytest.approx(1.23)
