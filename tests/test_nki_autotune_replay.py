"""CPU-only tests for NkiAutotuner canonicalization, winner trace, and replay.

No torch-xla, Neuron SDK, or Trainium required.
"""
import dataclasses
from types import SimpleNamespace

import pytest

from core import nki_autotune as na
from core.nki_autotune import (NkiAutotuner, NkiAutotuneReplayError,
                               NkiAutotuneSerializationError, canonical_config,
                               canonical_shape_key)


def fake_kernel(a, tile):  # a plain function is a valid "kernel" for identity
    return a


def other_kernel(a, tile):
    return a


SPACE = [SimpleNamespace(tile=t) for t in (16, 32, 64, 128, 256)]  # A..E
KEY = ((128, 8192), "torch.float16")


@pytest.fixture(autouse=True)
def _reset_registry():
    na.clear_tuning_trace()
    na.clear_tuning_replay()
    yield
    na.clear_tuning_trace()
    na.clear_tuning_replay()


def make_tuner(kernel=fake_kernel, times=None, **kw):
    """Tuner whose candidate timer is a stub (no Neuron SDK involved)."""
    tuner = NkiAutotuner(kernel, quiet=True, **kw)
    calls = []
    times = times or {}

    def stub(args):
        cfg_tile = args[-1]
        calls.append(cfg_tile)
        return times.get(cfg_tile, float(cfg_tile))

    tuner._time_candidate = stub
    tuner._timer_calls = calls
    return tuner


def record(tuner, cfg, key=KEY):
    return {"tuner_name": tuner.name,
            "shape_key": canonical_shape_key(key),
            "config": canonical_config(cfg)}


# ---------------------------------------------------------------------- replay

def test_replay_returns_recorded_winner_not_last_candidate():
    tuner = make_tuner()
    winner = SPACE[1]  # B — deliberately not the final candidate E
    na.install_tuning_replay([record(tuner, winner)])
    got = tuner.tune_or_cached(shape_key=KEY, search_space=SPACE,
                               args_fn=lambda c: ("x", c.tile))
    assert got is SPACE[1]
    na.assert_tuning_replay_consumed()


def test_replay_invokes_zero_timer_calls():
    tuner = make_tuner()
    na.install_tuning_replay([record(tuner, SPACE[1])])
    tuner.tune_or_cached(shape_key=KEY, search_space=SPACE,
                         args_fn=lambda c: ("x", c.tile))
    assert tuner._timer_calls == []


def test_replay_missing_config_in_search_space_fails():
    tuner = make_tuner()
    na.install_tuning_replay([record(tuner, SimpleNamespace(tile=999))])
    with pytest.raises(NkiAutotuneReplayError, match="no longer exists"):
        tuner.tune_or_cached(shape_key=KEY, search_space=SPACE,
                             args_fn=lambda c: ("x", c.tile))


def test_replay_duplicate_canonical_candidates_ambiguous():
    tuner = make_tuner()
    dup_space = SPACE + [SimpleNamespace(tile=32)]  # second canonical B
    na.install_tuning_replay([record(tuner, SPACE[1])])
    with pytest.raises(NkiAutotuneReplayError, match="ambiguous"):
        tuner.tune_or_cached(shape_key=KEY, search_space=dup_space,
                             args_fn=lambda c: ("x", c.tile))


def test_replay_wrong_shape_key_fails_strict():
    tuner = make_tuner()
    na.install_tuning_replay([record(tuner, SPACE[1], key=((64, 64), "fp32"))])
    with pytest.raises(NkiAutotuneReplayError, match="no record"):
        tuner.tune_or_cached(shape_key=KEY, search_space=SPACE,
                             args_fn=lambda c: ("x", c.tile))


def test_replay_wrong_tuner_name_fails_strict():
    tuner = make_tuner()
    rec = record(tuner, SPACE[1])
    rec["tuner_name"] = "somebody.else.kernel"
    na.install_tuning_replay([rec])
    with pytest.raises(NkiAutotuneReplayError, match="no record"):
        tuner.tune_or_cached(shape_key=KEY, search_space=SPACE,
                             args_fn=lambda c: ("x", c.tile))


def test_unused_replay_record_fails_consumption_check():
    tuner = make_tuner()
    na.install_tuning_replay([record(tuner, SPACE[1]),
                              record(tuner, SPACE[2], key=((1, 1), "fp32"))])
    tuner.tune_or_cached(shape_key=KEY, search_space=SPACE,
                         args_fn=lambda c: ("x", c.tile))
    with pytest.raises(NkiAutotuneReplayError, match="never consumed"):
        na.assert_tuning_replay_consumed()


def test_multiple_tuners_replay_independently():
    t1 = make_tuner(fake_kernel)
    t2 = make_tuner(other_kernel)
    assert t1.name != t2.name
    na.install_tuning_replay([record(t1, SPACE[1]), record(t2, SPACE[3])])
    assert t1.tune_or_cached(shape_key=KEY, search_space=SPACE,
                             args_fn=lambda c: ("x", c.tile)) is SPACE[1]
    assert t2.tune_or_cached(shape_key=KEY, search_space=SPACE,
                             args_fn=lambda c: ("x", c.tile)) is SPACE[3]
    na.assert_tuning_replay_consumed()
    assert t1._timer_calls == [] and t2._timer_calls == []


def test_explicit_name_collision_support():
    t1 = make_tuner(fake_kernel, name="op.kernel_a")
    assert t1.name == "op.kernel_a"


# --------------------------------------------------------- canonicalization

def test_dict_config_canonicalization():
    assert canonical_config({"b": 2, "a": (1, 2)}) == {"a": [1, 2], "b": 2}


def test_dataclass_config_canonicalization():
    @dataclasses.dataclass
    class Cfg:
        tile: int
        stages: int

    assert canonical_config(Cfg(tile=64, stages=3)) == {"stages": 3, "tile": 64}


def test_simplenamespace_config_canonicalization():
    assert canonical_config(SimpleNamespace(z=1, a=2)) == {"a": 2, "z": 1}


def test_equivalent_structures_canonicalize_identically():
    @dataclasses.dataclass
    class Cfg:
        a: int
        b: int

    d = canonical_config({"b": 2, "a": 1})
    assert d == canonical_config(SimpleNamespace(b=2, a=1)) == canonical_config(Cfg(1, 2))


def test_unsupported_config_fails_loudly():
    with pytest.raises(NkiAutotuneSerializationError):
        canonical_config(object())
    with pytest.raises(NkiAutotuneSerializationError):
        canonical_config({"a": object()})
    with pytest.raises(NkiAutotuneSerializationError):
        canonical_shape_key({1: "non-str-key"})


# ------------------------------------------------- non-replay behavior intact

def test_normal_tuning_still_works_and_records_trace():
    tuner = make_tuner(times={16: 5.0, 32: 1.0, 64: 2.0, 128: 3.0, 256: 4.0})
    got = tuner.tune_or_cached(shape_key=KEY, search_space=SPACE,
                               args_fn=lambda c: ("x", c.tile))
    assert got is SPACE[1]  # tile=32 fastest
    assert len(tuner._timer_calls) == len(SPACE)
    trace = na.export_tuning_trace()
    assert trace == [{"tuner_name": tuner.name,
                      "shape_key": canonical_shape_key(KEY),
                      "config": {"tile": 32}}]


def test_cached_winner_outside_replay_mode():
    tuner = make_tuner(times={16: 5.0, 32: 1.0, 64: 2.0, 128: 3.0, 256: 4.0})
    first = tuner.tune_or_cached(shape_key=KEY, search_space=SPACE,
                                 args_fn=lambda c: ("x", c.tile))
    n_calls = len(tuner._timer_calls)
    second = tuner.tune_or_cached(shape_key=KEY, search_space=SPACE,
                                  args_fn=lambda c: ("x", c.tile))
    assert second is first
    assert len(tuner._timer_calls) == n_calls  # cache hit: no re-timing


def test_replay_record_authoritative_over_inprocess_cache():
    tuner = make_tuner(times={16: 5.0, 32: 1.0, 64: 2.0, 128: 3.0, 256: 4.0})
    tuner.tune_or_cached(shape_key=KEY, search_space=SPACE,
                         args_fn=lambda c: ("x", c.tile))  # caches B (tile=32)
    na.install_tuning_replay([record(tuner, SPACE[3])])    # replay says D
    got = tuner.tune_or_cached(shape_key=KEY, search_space=SPACE,
                               args_fn=lambda c: ("x", c.tile))
    assert got is SPACE[3]
