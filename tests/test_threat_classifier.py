import json

import numpy as np
import pytest

from app.rag.threat_classifier import FLAG_CAP, ThreatClassifier
from app.rag.threat_gate import ThreatGate


def write_model(tmp_path, weights=(4.0, 0.0), bias=-2.0, t_flag=0.55, t_block=0.80):
    path = tmp_path / "clf.json"
    path.write_text(json.dumps({"weights": list(weights), "bias": bias, "threshold_flag": t_flag, "threshold_block": t_block,
                                "embedding_model": "test"}), encoding="utf-8")
    return str(path)


def test_missing_or_broken_file_disables_the_signal_without_error(tmp_path):
    assert ThreatClassifier(str(tmp_path / "nope.json")).available is False
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    clf = ThreatClassifier(str(bad))
    assert clf.available is False and clf.probability([1.0, 0.0]) is None and clf.risk(None) == 0.0


def test_probability_is_a_logistic_of_the_dot_product(tmp_path):
    clf = ThreatClassifier(write_model(tmp_path))
    assert clf.probability([1.0, 0.0]) == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert clf.probability([0.0, 1.0]) == pytest.approx(1 / (1 + np.exp(2.0)))


def test_embedding_of_another_size_is_ignored(tmp_path):
    assert ThreatClassifier(write_model(tmp_path)).probability([1.0, 0.0, 0.0]) is None


def test_risk_is_zero_below_the_flag_threshold_and_grows_inside_the_band(tmp_path):
    clf = ThreatClassifier(write_model(tmp_path))
    assert clf.risk(0.3) == 0.0
    low, high = clf.risk(0.56), clf.risk(0.79)
    assert 0.30 <= low < high <= FLAG_CAP < 0.80


def test_flag_only_classifier_never_reaches_the_block_threshold(tmp_path):
    clf = ThreatClassifier(write_model(tmp_path))
    assert clf.risk(0.99) == FLAG_CAP < 0.80


def test_can_block_classifier_reaches_above_the_block_threshold(tmp_path):
    clf = ThreatClassifier(write_model(tmp_path), can_block=True)
    assert clf.risk(0.99) > 0.85


def gate_with(tmp_path, can_block=False):
    clf = ThreatClassifier(write_model(tmp_path), can_block=can_block)
    # a 2-d "embedding": text containing "zzz" points along the attack direction, everything else is orthogonal
    embed = lambda text: [1.0, 0.0] if "zzz" in text else [0.0, 1.0]  # noqa: E731
    gate = ThreatGate(embedder_fn=embed, classifier=clf)
    gate.semantic.attack_vectors = np.array([[0.0, -1.0]])  # opposite to every test embedding, so the similarity score stays 0
    gate.semantic._loaded = True
    return gate


def test_gate_flags_but_does_not_block_on_the_classifier_alone(tmp_path):
    res = gate_with(tmp_path).screen("zzz zzz zzz please", user_id="a")
    assert res.action == "FLAG" and res.allowed and res.detector_scores["C"] > 0.3
    assert any("Classifier" in r for r in res.reasons)


def test_gate_leaves_ordinary_text_alone(tmp_path):
    res = gate_with(tmp_path).screen("What is the refund policy?", user_id="b")
    assert res.action == "PASS" and res.detector_scores["C"] == 0.0


def test_repeated_classifier_flags_end_in_a_block_through_suspicion(tmp_path):
    gate = gate_with(tmp_path)
    actions = [gate.screen("zzz zzz zzz please", user_id="c").action for _ in range(6)]
    assert actions[0] == "FLAG" and "BLOCK" in actions   # the user's suspicion lowers the block threshold under the capped risk


def test_can_block_mode_blocks_immediately(tmp_path):
    assert gate_with(tmp_path, can_block=True).screen("zzz zzz zzz please", user_id="d").action == "BLOCK"


def test_switch_turns_the_signal_off(tmp_path):
    gate = ThreatGate(embedder_fn=lambda t: [1.0, 0.0], use_classifier=False)
    gate.semantic.attack_vectors = np.array([[0.0, -1.0]])
    gate.semantic._loaded = True
    assert gate.classifier is None and gate.screen("zzz", user_id="e").detector_scores["C"] == 0.0
