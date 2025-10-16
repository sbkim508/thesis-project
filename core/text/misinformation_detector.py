"""Textual misinformation detector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from config import CONFIG
from utils.hf_pipeline import run_pipeline


@dataclass
class MisinformationDetector:
    """Score text segments for potential misinformation."""

    model_name: str = CONFIG.models.text_misinformation_model
    device: str = CONFIG.resolve_device()

    def predict(self, texts: Sequence[str]) -> float:
        if not texts:
            raise ValueError("No transcripts provided for misinformation detection")
        outputs = run_pipeline(
            "zero-shot-classification",
            self.model_name,
            inputs=list(texts),
            device=self.device,
            candidate_labels=["misinformation", "factual"],
        )
        if isinstance(outputs, dict):  # single item
            scores = outputs["scores"]
            labels = [label.lower() for label in outputs["labels"]]
            score = next(
                (score for label, score in zip(labels, scores) if "misinformation" in label),
                0.0,
            )
            return float(score)
        scores = []
        for output in outputs:
            labels = [label.lower() for label in output["labels"]]
            scores.append(
                next((score for label, score in zip(labels, output["scores"]) if "misinformation" in label), 0.0)
            )
        return float(np.mean(scores))


__all__ = ["MisinformationDetector"]
