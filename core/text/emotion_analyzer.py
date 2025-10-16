"""Text emotion analyzer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from config import CONFIG
from utils.hf_pipeline import run_pipeline


@dataclass
class TextEmotionAnalyzer:
    """Return aggregated emotion scores for transcripts."""

    model_name: str = CONFIG.models.text_emotion_model
    device: str = CONFIG.resolve_device()

    def predict(self, texts: Sequence[str]) -> Dict[str, float]:
        if not texts:
            raise ValueError("No text provided for emotion analysis")
        outputs = run_pipeline(
            "text-classification",
            self.model_name,
            inputs=list(texts),
            device=self.device,
            truncation=True,
            top_k=None,
        )
        aggregated: Dict[str, List[float]] = {}
        if isinstance(outputs, dict):
            outputs = [outputs]
        for result in outputs:
            entries = result if isinstance(result, list) else [result]
            for entry in entries:
                label = entry["label"].lower()
                aggregated.setdefault(label, []).append(float(entry["score"]))
        return {label: float(np.mean(scores)) for label, scores in aggregated.items()}


__all__ = ["TextEmotionAnalyzer"]
