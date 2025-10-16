"""Visual emotion recognizer for harmfulness assessment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from config import CONFIG
from utils.hf_pipeline import run_pipeline


@dataclass
class VisualEmotionRecognizer:
    """Infer facial emotion distributions across frames."""

    model_name: str = CONFIG.models.visual_emotion_model
    device: str = CONFIG.resolve_device()

    def predict(self, frames: Sequence[Path]) -> Dict[str, float]:
        if not frames:
            raise ValueError("No frames provided for emotion recognition")
        outputs = run_pipeline(
            "image-classification",
            self.model_name,
            inputs=[str(frame) for frame in frames],
            device=self.device,
        )
        aggregated: Dict[str, List[float]] = {}
        for output in outputs:
            if isinstance(output, list):
                entries = output
            else:  # pragma: no cover
                entries = [output]
            for entry in entries:
                label = entry["label"].lower()
                aggregated.setdefault(label, []).append(float(entry["score"]))
        return {label: float(np.mean(scores)) for label, scores in aggregated.items()}


__all__ = ["VisualEmotionRecognizer"]
