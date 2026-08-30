"""CPU-only tests for runtime-trace window timing (core/nki_timer.py)."""
import os

import pytest

from core.nki_timer import (NkiTraceError, find_session_dir, session_neffs,
                            time_windows)


def ex(model, start, end):
    return {"flow_id": [start], "model_id": model, "start_ns": start, "end_ns": end,
            "pcores": 2}


def test_single_graph_per_window_mean():
    execs = [ex("A", 50, 60),                 # verification run: outside every window
             ex("A", 110, 130), ex("A", 210, 240), ex("A", 310, 320)]
    stats = time_windows(execs, [[100, 200], [200, 300], [300, 400]], tag="nki")
    assert stats["repeat"] == 3
    assert stats["per_iteration_ms"] == pytest.approx([20e-6, 30e-6, 10e-6])
    assert stats["mean"] == pytest.approx(20e-6)
    assert stats["min"] == pytest.approx(10e-6) and stats["max"] == pytest.approx(30e-6)
    assert stats["executions_per_iteration"] == 1
    assert stats["per_model"] == {"A": {"count_per_iteration": 1,
                                        "mean_ms": pytest.approx(20e-6)}}
    assert stats["method"] == "neuron_rt_inspect"


def test_multi_graph_iteration_is_summed():
    # radix-sort shape: two kernel passes + one XLA helper graph per run()
    execs = [ex("K", 110, 120), ex("K", 125, 135), ex("X", 140, 141),
             ex("K", 210, 220), ex("K", 225, 235), ex("X", 240, 241)]
    stats = time_windows(execs, [[100, 200], [200, 300]], tag="nki")
    assert stats["executions_per_iteration"] == 3
    assert stats["mean"] == pytest.approx(21e-6)
    assert stats["per_model"]["K"]["count_per_iteration"] == 2
    assert stats["per_model"]["X"]["count_per_iteration"] == 1


def test_empty_window_fails():
    with pytest.raises(NkiTraceError, match="contains no"):
        time_windows([ex("A", 10, 20)], [[100, 200]], tag="torch")


def test_execution_straddling_a_window_boundary_fails():
    with pytest.raises(NkiTraceError, match="cuts through"):
        time_windows([ex("A", 150, 250), ex("A", 260, 270)], [[100, 200], [200, 300]])


def test_varying_graph_pattern_fails():
    execs = [ex("K", 110, 120), ex("X", 130, 131), ex("K", 210, 220)]  # X missing in iter 2
    with pytest.raises(NkiTraceError, match="pattern differs"):
        time_windows(execs, [[100, 200], [200, 300]], tag="nki")


def test_no_windows_fails():
    with pytest.raises(NkiTraceError, match="no timed windows"):
        time_windows([ex("A", 1, 2)], [])


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")


def test_find_session_dir_requires_exactly_one(tmp_path):
    root = str(tmp_path)
    with pytest.raises(NkiTraceError, match="found 0"):
        find_session_dir(root)
    _touch(os.path.join(root, "i-abc_pid_1", "111", "ntrace.pb"))
    assert find_session_dir(root) == os.path.join(root, "i-abc_pid_1", "111")
    _touch(os.path.join(root, "i-abc_pid_2", "222", "ntrace.pb"))
    with pytest.raises(NkiTraceError, match="found 2"):
        find_session_dir(root)


def test_session_neffs_parses_model_ids(tmp_path):
    sess = str(tmp_path)
    _touch(os.path.join(sess, "neff_400072599073199_vnc_0.neff"))
    _touch(os.path.join(sess, "neff_860884412189317_vnc_0.neff"))
    _touch(os.path.join(sess, "400072599073199_vnc_0.ntff"))
    out = session_neffs(sess)
    assert set(out) == {"400072599073199", "860884412189317"}
    _touch(os.path.join(sess, "neff_400072599073199_vnc_1.neff"))
    with pytest.raises(NkiTraceError, match="several NEFF files"):
        session_neffs(sess)
