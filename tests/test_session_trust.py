import threading

import pytest

from app.rag.session_trust import (
    ALPHA, BLOCK_FLOOR, STATIC_POLICY, SessionTrustStore, policy_for,
)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def store():
    clock = FakeClock()
    s = SessionTrustStore(secret_key="test-secret", clock=clock)
    s.clock = clock
    return s


def test_new_user_has_zero_suspicion_and_static_policy(store):
    key = store.key("t", "u", "10.0.0.1")
    assert store.suspicion(key) == 0.0
    assert store.policy(key) == STATIC_POLICY


def test_policy_at_zero_equals_the_static_settings():
    p = policy_for(0.0)
    assert (p.block_threshold, p.flag_threshold, p.allow_tier2, p.entailment_threshold, p.min_grounding) == (0.80, 0.30, True, 0.60, 0.0)


def test_policy_tightens_monotonically_and_respects_floors():
    prev = policy_for(0.0)
    for s in (0.2, 0.4, 0.6, 0.8, 1.0):
        p = policy_for(s)
        assert p.block_threshold <= prev.block_threshold
        assert p.flag_threshold <= prev.flag_threshold
        assert p.entailment_threshold >= prev.entailment_threshold
        assert p.block_threshold >= BLOCK_FLOOR
        prev = p
    assert policy_for(1.0).allow_tier2 is False and policy_for(1.0).min_grounding > 0
    assert policy_for(0.4).allow_tier2 is True and policy_for(0.4).min_grounding == 0.0


def test_evidence_raises_suspicion_with_saturation(store):
    key = store.key("t", "u", "10.0.0.1")
    s1 = store.observe(key, 1.0)
    assert s1 == pytest.approx(ALPHA)
    s2 = store.observe(key, 1.0)
    assert s1 < s2 < 1.0
    assert s2 == pytest.approx(s1 + (1 - s1) * ALPHA)


def test_suspicion_decays_by_half_life(store):
    key = store.key("t", "u", "10.0.0.1")
    s = store.observe(key, 1.0)
    store.clock.t += store.half_life_sec
    assert store.suspicion(key) == pytest.approx(s / 2)


def test_clean_turns_let_suspicion_recover(store):
    key = store.key("t", "u", "10.0.0.1")
    s = store.observe(key, 1.0)
    for _ in range(5):
        s = store.observe_layer1(key, base_risk=0.0)
    assert s < 0.6 * ALPHA


def test_layer1_evidence_scales_with_risk(store):
    a, b = store.key("t", "a", "10.0.0.1"), store.key("t", "b", "10.0.0.1")
    low = store.observe_layer1(a, base_risk=0.35)
    high = store.observe_layer1(b, base_risk=0.70)
    blocked = store.observe_layer1(store.key("t", "c", "10.0.0.1"), base_risk=0.9, blocked=True)
    assert 0 < low < high < blocked == pytest.approx(ALPHA)


def test_layer2_and_layer3_evidence(store):
    key = store.key("t", "u", "10.0.0.1")
    assert store.observe_layer2(key, quarantined=0) == 0.0
    assert store.observe_layer3(key, safety_hit=False, contradictions=0) == 0.0
    after_l2 = store.observe_layer2(key, quarantined=2)
    assert after_l2 > 0
    assert store.observe_layer3(key, safety_hit=True) > after_l2
    assert store.observe_layer3(store.key("t", "v", "10.0.0.1"), contradictions=2) > 0


def test_state_is_keyed_on_tenant_user_subnet_not_session(store):
    assert store.key("t", "u", "10.0.0.1") == store.key("t", "u", "10.0.0.200")   # same /24
    assert store.key("t", "u", "10.0.0.1") != store.key("t", "u", "10.0.1.1")
    assert store.key("t", "u", "10.0.0.1") != store.key("t", "other", "10.0.0.1")
    assert store.key("t", "u", "10.0.0.1") != store.key("t2", "u", "10.0.0.1")


def test_concurrent_updates_do_not_lose_evidence():
    store = SessionTrustStore(secret_key="k", alpha=0.01)
    key = store.key("t", "u", "10.0.0.1")
    threads = [threading.Thread(target=lambda: [store.observe(key, 1.0) for _ in range(50)]) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    expected = 1 - (1 - 0.01) ** 200
    assert store.suspicion(key) == pytest.approx(expected, rel=1e-3)
