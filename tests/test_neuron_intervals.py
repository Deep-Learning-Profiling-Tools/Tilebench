"""Busy-sum aggregation of the native Trn2 timing (tilebench/core/neuron_intervals.py): the per-core
copies of one device execution count once, distinct executions add up, and union / span are kept apart."""
import pytest

from tilebench.core import neuron_intervals as intervals


def test_intervals_merge_cores_and_separate_sum_union_span():
    ex = [
        {"start_ns": 0, "end_ns": 10, "execution_id": "a"},
        {"start_ns": 2, "end_ns": 12, "execution_id": "a"},   # second core of "a"
        {"start_ns": 8, "end_ns": 20, "execution_id": "b"},   # overlaps "a"
        {"start_ns": 30, "end_ns": 35, "execution_id": "c"},  # after a gap
    ]
    agg = intervals.aggregate(ex)
    assert agg["n_executions"] == 3
    assert agg["busy_sum_ms"] == pytest.approx((12 + 12 + 5) / 1e6)
    assert agg["union_ms"] == pytest.approx((20 + 5) / 1e6)
    assert agg["span_ms"] == pytest.approx(35 / 1e6)
    assert agg["overlap_between_executions"]


def test_intervals_empty_and_invalid():
    assert intervals.aggregate([])["busy_sum_ms"] is None
    with pytest.raises(ValueError):
        intervals.aggregate([{"start_ns": 5, "end_ns": 1}])
