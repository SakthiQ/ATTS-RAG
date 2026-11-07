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
    """

    def __init__(self, path: str = DEFAULT_PATH, can_block: Optional[bool] = None):
        self.can_block = bool(can_block)
        self.available = False
        self.weights = np.zeros(0)
        self.bias = 0.0
        self.t_flag = self.t_block = 1.0
        self.embedding_model = ""
        self.scoring = "whole"
        self.semantic: dict = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.weights = np.asarray(cfg["weights"], dtype=float)
            self.bias = float(cfg["bias"])
            self.t_flag, self.t_block = float(cfg["threshold_flag"]), float(cfg["threshold_block"])
            self.embedding_model = cfg.get("embedding_model", "")
            self.scoring = cfg.get("scoring", "whole")
            self.semantic = cfg.get("semantic", {}) or {}
            # an explicit can_block argument (ablations) overrides the file's setting
            self.can_block = bool(cfg.get("can_block", False)) if can_block is None else bool(can_block)
            self.available = self.weights.size > 0 and self.t_flag <= self.t_block
        except FileNotFoundError:
            logger.info(f"No Layer 1 classifier at {path}; running without the C signal.")
        except Exception as exc:
            logger.warning(f"Layer 1 classifier could not be loaded ({exc}); running without the C signal.")

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

    def risk(self, probability: Optional[float]) -> float:
        """Maps P(attack) to a Layer 1 risk: 0 below the flag threshold, 0.30 up to FLAG_CAP in the flag band, 0.85-1.0 from the block threshold."""
        if probability is None or probability < self.t_flag:
            return 0.0
        if probability >= self.t_block:
            if not self.can_block:
                return FLAG_CAP
            return min(1.0, BLOCK_RISK + (1.0 - BLOCK_RISK) * (probability - self.t_block) / max(1.0 - self.t_block, 1e-9))
        span = max(self.t_block - self.t_flag, 1e-9)
        return FLAG_RISK + (FLAG_CAP - FLAG_RISK) * (probability - self.t_flag) / span
