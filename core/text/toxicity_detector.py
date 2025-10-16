"""Toxicity classifier for harmfulness signals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from config import CONFIG
from utils.hf_pipeline import run_pipeline


@dataclass
class ToxicityDetector:
    """Aggregate toxicity scores across text segments."""

    model_name: str = CONFIG.models.text_toxicity_model
    device: str = CONFIG.resolve_device()

    def predict(self, texts: Sequence[str]) -> float:
        if not texts:
            raise ValueError("No text provided for toxicity detection")
        outputs = run_pipeline(
            "text-classification",
            self.model_name,
            inputs=list(texts),
            device=self.device,
            truncation=True,
        )
        if isinstance(outputs, dict):
            return float(outputs.get("score", 0.0))
        scores = [float(result.get("score", 0.0)) for result in outputs]
        return float(np.mean(scores))


__all__ = ["ToxicityDetector"]
