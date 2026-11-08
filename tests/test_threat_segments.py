"""Segment scoring in Layer 1: an attack inside a long message must not be averaged away.

A tiny deterministic embedder stands in for the sentence model: dimension 0 is the share of words that are the
marker word `zzattack`, so one embedding of a long message dilutes the marker exactly as a real model dilutes an
attack sentence, while the segment holding the marker scores high.
"""
import json

import numpy as np
import pytest

from app.rag.threat_classifier import ThreatClassifier, resolve_model
from app.rag.threat_gate import MAX_SEGMENTS, ThreatGate, segment_text

FILLER = ("Our quarterly review covers vendor onboarding and invoice approval timelines. "
          "The travel policy was updated for contractors this year. "
          "Laptop refresh cycles for the finance team are now three years. ") * 3
ATTACK = "Please obey zzattack right now."


def fake_embed(text):
    words = text.split()
    share = sum(w.strip(".,") == "zzattack" for w in words) / max(len(words), 1)
    return [share, 1.0 - share]


def write_config(tmp_path, scoring, can_block=True):
    path = tmp_path / f"clf_{scoring}.json"
    path.write_text(json.dumps({
        "version": 2, "embedding_model": "fake", "scoring": scoring, "can_block": can_block,
        # P(attack) = sigmoid(40 * marker share - 4): about 0.99 for the attack sentence alone, about 0.03 once it is
        # one sentence in a 90-word message
        "weights": [40.0, 0.0], "bias": -4.0, "threshold_flag": 0.5, "threshold_block": 0.9,
    }), encoding="utf-8")
    return str(path)


def gate_for(tmp_path, scoring, **kw):
    # The similarity signal is off: with this two-number embedder every attack-corpus phrase looks like plain text
    return ThreatGate(embedder_fn=fake_embed, use_session=False, use_semantic=False,
                      classifier=ThreatClassifier(write_config(tmp_path, scoring, **kw)))


def screen(gate, text):
    return gate.screen(text, user_id="seg-test", client_ip="10.1.2.3")


# ---------------------------------------------------------------------------
# segment_text
# ---------------------------------------------------------------------------

def test_a_single_sentence_is_one_segment():
    assert segment_text("What is the refund policy?") == ["What is the refund policy?"]


def test_several_sentences_give_whole_sentences_and_pairs():
    segs = segment_text("First one. Second one! Third one?")
    assert segs[0] == "First one. Second one! Third one?"
    assert {"First one.", "Second one!", "Third one?", "First one. Second one!", "Second one! Third one?"} <= set(segs)


def test_a_long_sentence_is_cut_into_overlapping_windows():
    words = [f"w{i}" for i in range(100)]
    segs = segment_text(" ".join(words))
    windows = [s for s in segs[1:] if len(s.split()) == 40]
    assert windows and windows[0].split()[0] == "w0" and windows[1].split()[0] == "w20"


def test_the_number_of_segments_is_capped():
    text = " ".join(f"Sentence number {i} is here." for i in range(400))
    assert len(segment_text(text)) == MAX_SEGMENTS


def test_empty_text_has_no_segments():
    assert segment_text("   ") == []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_whole_message_scoring_misses_an_attack_hidden_in_a_long_message(tmp_path):
    gate = gate_for(tmp_path, "whole")
    assert screen(gate, ATTACK).action == "BLOCK"                 # alone it is caught
    assert screen(gate, FILLER + ATTACK).action == "PASS"         # diluted: the known weakness


@pytest.mark.parametrize("position", ["start", "middle", "end"])
def test_segment_scoring_finds_the_attack_wherever_it_is(tmp_path, position):
    parts = FILLER.split(". ")
    idx = {"start": 0, "middle": len(parts) // 2, "end": len(parts)}[position]
    text = ". ".join(parts[:idx] + [ATTACK.rstrip(".")] + parts[idx:])
    result = screen(gate_for(tmp_path, "segment_max"), text)
    assert result.action == "BLOCK"
    assert any("part" in r for r in result.reasons), result.reasons


def test_segment_scoring_leaves_ordinary_long_text_alone(tmp_path):
    assert screen(gate_for(tmp_path, "segment_max"), FILLER).action == "PASS"


def test_the_config_decides_whether_the_classifier_may_block(tmp_path):
    flag_only = gate_for(tmp_path, "segment_max", can_block=False)
    assert screen(flag_only, ATTACK).action == "FLAG"
    override = ThreatGate(embedder_fn=fake_embed, use_session=False, use_semantic=False,
                          classifier=ThreatClassifier(write_config(tmp_path, "segment_max", can_block=True), can_block=False))
    assert screen(override, ATTACK).action == "FLAG", "an explicit can_block argument overrides the file"


def test_batched_probabilities_match_single_ones(tmp_path):
    clf = ThreatClassifier(write_config(tmp_path, "segment_max"))
    vecs = np.array([[0.1, 0.9], [0.5, 0.5]])
    batch = clf.probabilities(vecs)
    assert batch == pytest.approx([clf.probability(v) for v in vecs])
    assert clf.probabilities(np.ones((2, 3))) is None   # wrong dimension: no score rather than a wrong one


# ---------------------------------------------------------------------------
# Model location
# ---------------------------------------------------------------------------

def test_a_local_model_copy_under_atts_model_dir_is_preferred(tmp_path, monkeypatch):
    (tmp_path / "paraphrase-multilingual-MiniLM-L12-v2").mkdir()
    monkeypatch.setenv("ATTS_MODEL_DIR", str(tmp_path))
    name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert resolve_model(name) == str(tmp_path / "paraphrase-multilingual-MiniLM-L12-v2")
    assert resolve_model("sentence-transformers/all-MiniLM-L6-v2") == "sentence-transformers/all-MiniLM-L6-v2"
    monkeypatch.delenv("ATTS_MODEL_DIR")
    assert resolve_model(name) == name
