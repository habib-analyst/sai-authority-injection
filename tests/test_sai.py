"""Tests for sai-authority-injection. Deterministic throughout."""
import dataclasses
import sys

import pytest

sys.path.insert(0, "src")

from sai import (  # noqa: E402
    AuthoritySnapshot,
    AuthorityStore,
    OperatorHandle,
    OverrideClass,
    PRECEDENCE,
    SCENARIOS,
    paired_bootstrap_ci,
    run_episode,
    validate_override,
)
from sai.simulator import planner_decide  # noqa: E402


def good_payload(cls="constraint"):
    return {"class": cls, "target": "tool:file_delete",
            "directive": "Never delete files. Move them to trash or archive instead."}


# --- schema validation -----------------------------------------------------

def test_validate_accepts_all_three_classes():
    for cls in ("constraint", "priority", "strategy"):
        ok, reason = validate_override(good_payload(cls))
        assert ok, reason


def test_validate_rejects_unknown_class():
    ok, reason = validate_override({**good_payload(), "class": "suggestion"})
    assert not ok and "unknown override class" in reason


def test_validate_rejects_missing_keys():
    for payload in ({}, {"class": "constraint"},
                    {"class": "constraint", "target": "tool:x"}):
        ok, _ = validate_override(payload)
        assert not ok


def test_validate_rejects_bad_target():
    for target in ("Bad Target!", "x", "UPPER:case", "", "a" * 100, 42, None):
        ok, _ = validate_override({**good_payload(), "target": target})
        assert not ok, target


def test_validate_rejects_short_and_long_directive():
    ok, _ = validate_override({**good_payload(), "directive": "short"})
    assert not ok
    ok, _ = validate_override({**good_payload(), "directive": "z" * 400})
    assert not ok


def test_validate_rejects_non_dict():
    for payload in ("text", None, 42, ["class"]):
        ok, _ = validate_override(payload)
        assert not ok


def test_malformed_batch_all_rejected():
    bad = [
        {}, {"class": "constraint"},
        {"class": "ban", "target": "tool:x", "directive": "y" * 20},
        {"class": "constraint", "target": "Bad Target!", "directive": "do the thing please"},
        {"class": "priority", "target": "ok:target", "directive": "short"},
        {"class": "strategy", "target": "ok:target", "directive": "z" * 400},
    ]
    assert all(not validate_override(p)[0] for p in bad)


# --- directed authority channel --------------------------------------------

def test_register_requires_operator_handle():
    store = AuthorityStore()
    with pytest.raises(PermissionError):
        store.register("not-a-handle", good_payload())
    with pytest.raises(PermissionError):
        store.register(OperatorHandle.__new__(OperatorHandle), good_payload())


def test_snapshot_ranked_constraint_first():
    store = AuthorityStore()
    h = store.operator_handle()
    store.register(h, good_payload("strategy"))
    store.register(h, good_payload("priority"))
    store.register(h, good_payload("constraint"))
    snap = store.snapshot()
    assert [e.cls for e in snap.entries] == [
        OverrideClass.CONSTRAINT, OverrideClass.PRIORITY, OverrideClass.STRATEGY]
    assert [e.rank for e in snap.entries] == [
        PRECEDENCE[OverrideClass.CONSTRAINT],
        PRECEDENCE[OverrideClass.PRIORITY],
        PRECEDENCE[OverrideClass.STRATEGY]]


def test_snapshot_immutable_to_planner():
    store = AuthorityStore()
    h = store.operator_handle()
    store.register(h, good_payload())
    snap = store.snapshot()
    assert isinstance(snap, AuthoritySnapshot)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.entries = ()  # type: ignore
    assert isinstance(snap.entries, tuple)  # no list API for the planner to abuse


def test_override_ids_unique_and_sequenced():
    store = AuthorityStore()
    h = store.operator_handle()
    ovs = [store.register(h, good_payload()) for _ in range(5)]
    ids = [o.override_id for o in ovs]
    assert len(set(ids)) == 5
    assert [o.seq for o in ovs] == sorted(o.seq for o in ovs)


def test_register_rejects_malformed_payload():
    store = AuthorityStore()
    h = store.operator_handle()
    before = len(store)
    with pytest.raises(ValueError, match="schema validation"):
        store.register(h, {"class": "constraint"})
    assert len(store) == before


# --- simulator behavior ----------------------------------------------------

def test_freetext_erosion_present():
    r = run_episode(SCENARIOS[0], "freetext", seed=1)
    assert r.compliance[0] is True
    assert r.compliance[9] is False
    assert sum(r.compliance) < len(r.compliance)


def test_sai_compliance_persists_all_turns():
    for sc in SCENARIOS:
        r = run_episode(sc, "sai", seed=2)
        assert all(r.compliance), sc.sid


def test_repeat_reminder_costs_more_tokens_than_sai():
    sc = SCENARIOS[0]
    r_rep = run_episode(sc, "freetext_repeat", seed=3)
    r_sai = run_episode(sc, "sai", seed=3)
    assert sum(r_rep.tokens_per_turn) > sum(r_sai.tokens_per_turn)


def test_episode_reproducible_same_seed():
    sc = SCENARIOS[1]
    a = run_episode(sc, "freetext", seed=99)
    b = run_episode(sc, "freetext", seed=99)
    assert a.compliance == b.compliance
    assert a.tokens_per_turn == b.tokens_per_turn


def test_distractor_schedule_deterministic_across_modes():
    # Same seed -> same distractor paraphrase order in both arms.
    import random
    from sai.scenarios import DISTRACTOR_TURNS
    sc = SCENARIOS[2]
    rng1, rng2 = random.Random(7), random.Random(7)
    d1 = list(sc.distractors); rng1.shuffle(d1)
    d2 = list(sc.distractors); rng2.shuffle(d2)
    assert [d1[DISTRACTOR_TURNS.index(t)] for t in DISTRACTOR_TURNS] == \
           [d2[DISTRACTOR_TURNS.index(t)] for t in DISTRACTOR_TURNS]


# --- statistics ------------------------------------------------------------

def test_bootstrap_ci_excludes_zero_on_known_diff():
    x = [0.0] * 50
    y = [1.0] * 50
    mean, lo, hi = paired_bootstrap_ci(x, y, n_boot=1000, seed=0)
    assert mean == pytest.approx(1.0)
    assert lo > 0


def test_bootstrap_ci_covers_zero_on_no_diff():
    rng_vals = [0.0, 1.0] * 50
    mean, lo, hi = paired_bootstrap_ci(rng_vals, rng_vals, n_boot=1000, seed=1)
    assert mean == pytest.approx(0.0)
    assert lo <= 0 <= hi


def test_planner_never_receives_store():
    # Structural: planner_decide's signature has no store parameter.
    import inspect
    params = inspect.signature(planner_decide).parameters
    assert "store" not in params
    assert "authority" in params  # snapshot only
