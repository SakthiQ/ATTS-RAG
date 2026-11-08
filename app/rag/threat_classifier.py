"""Layer 1 classifier signal (C): a small logistic regression over the query embedding.

The rules (H) are precise but only recognise phrasings someone wrote down, and the similarity score (V) is capped
because raw MiniLM similarity does not separate attacks from benign text. C is trained on public prompt-injection and
jailbreak data plus in-domain benign questions (scripts/train_threat_classifier.py), with thresholds calibrated on a
held-out dev split.

C is a FLAG-ONLY signal by default (can_block=False): its risk never exceeds FLAG_CAP (0.50), well under the block
threshold. Evaluated on hand-written benign questions it still flags some ordinary ones (for example "Show me the
documents about data privacy and security"), and a false flag costs the user nothing, while a false block costs an
answer. FLAG_CAP is deliberately low: a higher cap made those false flags raise honest users' suspicion enough to cause
extra blocks in the dev sessions. Repeated flags from a user who keeps probing are turned into a block by the user's
suspicion value (app/rag/session_trust.py), which lowers the block threshold. Set can_block=True only for experiments.

The model is stored as plain numbers in config/threat_classifier.json (no pickle), and scoring is one dot product over the
embedding Layer 1 already computes for V.
"""
import json
import os
from typing import Optional, Sequence

import numpy as np
from loguru import logger

DEFAULT_PATH = os.path.join("config", "threat_classifier.json")
FLAG_RISK = 0.30      # risk at the flag threshold (the static flag cut-off)
BLOCK_RISK = 0.85     # risk at the block threshold (above the static block cut-off of 0.80)
FLAG_CAP = 0.50       # highest risk a flag-only classifier can report. Calibrated on the dev sessions of eval/adaptive_sessions.json: 0.79 caused an
                      # extra benign block, 0.60 let honest users reach suspicion 0.53, 0.50 is the lowest-release setting that met the dev rule
STRONG_RISK = 0.60    # a score above the block threshold that no other signal backs: flagged, and blocked only for a user whose
                      # suspicion has already lowered the block threshold to 0.60 (about three such hits in a row)
SUSPICION_FULL = 0.50   # suspicion at which the flag threshold reaches its floor (linear from 0); see `flag_floor`


def resolve_model(name: str) -> str:
    """The local copy of a model under ATTS_MODEL_DIR (e.g. D:\\ATTS-RAG-models\\<model folder>) if there is one,
    else the Hugging Face name (loaded from, or downloaded into, the Hugging Face cache)."""
    base = os.getenv("ATTS_MODEL_DIR")
    if base and name:
        local = os.path.join(base, name.split("/")[-1])
        if os.path.isdir(local):
            return local
    return name


class ThreatClassifier:
    """Logistic regression over the query embedding.

    The JSON file decides how the gate uses it: `embedding_model` (the gate must embed with the same model),
    `scoring` ("whole": score the whole message; "segment_max": score every sentence / window and keep the highest,
    so an attack hidden in a long message is not averaged away), `can_block` (whether a score above the block
    threshold may block on its own) and optional `semantic` calibration of the similarity signal for that model.

    Thresholds may depend on message length (`length_thresholds`: a list of {max_segments, flag, block, block_alone},
    max_segments null for the last bucket). With segment scoring the gate keeps the highest score over a message's
    pieces, and the more pieces, the more chances one ordinary sentence scores high, so one threshold for all lengths
    is either too strict on long text or too lenient on short attacks. Each bucket is calibrated on dev to the same
    benign false-positive rates. `block_alone` (default: the block threshold) is the stricter level at which the
    classifier blocks with no other signal; between block and block_alone it blocks only when the rules, the
    similarity check or the authority-claim check also fired.

    `flag_floor` (per bucket, and `threshold_flag_floor` globally; default: the flag threshold, i.e. off) is a lower
    flag threshold, calibrated on dev to a looser benign false-positive rate, that the classifier moves towards as the
    user's suspicion grows. Without it a reworded attack scoring just below the flag threshold reports risk 0, and
    suspicion, which only lowers the gate's thresholds on that risk, cannot catch it however often the user has probed.
    """

    def __init__(self, path: str = DEFAULT_PATH, can_block: Optional[bool] = None):
        self.can_block = bool(can_block)
        self.available = False
        self.weights = np.zeros(0)
        self.bias = 0.0
        self.t_flag = self.t_block = self.t_block_alone = 1.0
        self.t_flag_floor = 1.0
        self.buckets: list = []   # [(max_segments or None, flag, block, block_alone)], ascending
        self.floors: dict = {}    # max_segments or None -> flag floor
        self.embedding_model = ""
        self.scoring = "whole"
        self.semantic: dict = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.weights = np.asarray(cfg["weights"], dtype=float)
            self.bias = float(cfg["bias"])
            self.t_flag, self.t_block = float(cfg["threshold_flag"]), float(cfg["threshold_block"])
            self.t_block_alone = float(cfg.get("threshold_block_alone", self.t_block))
            self.t_flag_floor = float(cfg.get("threshold_flag_floor", self.t_flag))
            for b in cfg.get("length_thresholds", []) or []:
                block = float(b["block"])
                self.buckets.append((b.get("max_segments"), float(b["flag"]), block, float(b.get("block_alone", block))))
                self.floors[b.get("max_segments")] = float(b.get("flag_floor", b["flag"]))
            self.buckets.sort(key=lambda b: float("inf") if b[0] is None else b[0])
            self.embedding_model = cfg.get("embedding_model", "")
            self.scoring = cfg.get("scoring", "whole")
            self.semantic = cfg.get("semantic", {}) or {}
            # an explicit can_block argument (ablations) overrides the file's setting
            self.can_block = bool(cfg.get("can_block", False)) if can_block is None else bool(can_block)
            ordered = (all(f <= b <= a and self.floors[m] <= f for m, f, b, a in self.buckets)
                       and self.t_flag_floor <= self.t_flag <= self.t_block <= self.t_block_alone)
            self.available = self.weights.size > 0 and ordered
        except FileNotFoundError:
            logger.info(f"No Layer 1 classifier at {path}; running without the C signal.")
        except Exception as exc:
            logger.warning(f"Layer 1 classifier could not be loaded ({exc}); running without the C signal.")

    def thresholds(self, n_segments: Optional[int] = None):
        """(flag, block, block_alone) for a message cut into n_segments pieces (the global ones without buckets)."""
        if self.buckets and n_segments is not None:
            for max_segments, flag, block, alone in self.buckets:
                if max_segments is None or n_segments <= max_segments:
                    return flag, block, alone
        return self.t_flag, self.t_block, self.t_block_alone

    def flag_floor(self, n_segments: Optional[int] = None) -> float:
        """The lowest flag threshold suspicion can move the classifier to, for a message of n_segments pieces."""
        if self.buckets and n_segments is not None:
            for max_segments, *_ in self.buckets:
                if max_segments is None or n_segments <= max_segments:
                    return self.floors.get(max_segments, self.thresholds(n_segments)[0])
        return self.t_flag_floor

    def probabilities(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        """P(attack) for each row of a matrix of normalised embeddings (one per segment)."""
        mat = np.asarray(embeddings, dtype=float)
        if not self.available or mat.ndim != 2 or mat.shape[1] != self.weights.shape[0]:
            return None
        z = np.clip(mat @ self.weights + self.bias, -700, 700)
        return 1.0 / (1.0 + np.exp(-z))

    def probability(self, embedding: Sequence[float]) -> Optional[float]:
        """P(attack) for a normalised query embedding; None if the classifier cannot score this embedding."""
        vec = np.asarray(embedding, dtype=float)
        if not self.available or vec.shape != self.weights.shape:
            return None   # e.g. a test embedder with another dimension
        z = float(vec @ self.weights + self.bias)
        return 1.0 / (1.0 + np.exp(-z)) if z > -700 else 0.0

    def risk(self, probability: Optional[float], n_segments: Optional[int] = None, corroborated: bool = True,
             suspicion: float = 0.0) -> float:
        """Maps P(attack) to a Layer 1 risk: 0 below the flag threshold, 0.30 up to FLAG_CAP in the flag band, 0.85-1.0
        from the block threshold. Above the block threshold but below block_alone the classifier blocks only when
        `corroborated` (another detector fired); otherwise it reports STRONG_RISK, a flag. `suspicion` (the user's,
        0-1) lowers the flag threshold towards the flag floor; the block thresholds never move here."""
        t_flag, t_block, t_alone = self.thresholds(n_segments)
        if suspicion > 0:
            t_flag -= min(1.0, suspicion / SUSPICION_FULL) * (t_flag - self.flag_floor(n_segments))
        if probability is None or probability < t_flag:
            return 0.0
        if probability >= t_block:
            if not self.can_block:
                return FLAG_CAP
            if probability < t_alone and not corroborated:
                return STRONG_RISK
            return min(1.0, BLOCK_RISK + (1.0 - BLOCK_RISK) * (probability - t_block) / max(1.0 - t_block, 1e-9))
        span = max(t_block - t_flag, 1e-9)
        return FLAG_RISK + (FLAG_CAP - FLAG_RISK) * (probability - t_flag) / span
