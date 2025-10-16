"""Deepfake detector leveraging image classification pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from config import CONFIG
from utils.hf_pipeline import run_pipeline


@dataclass
class DeepfakeDetector:
    """Predict deepfake likelihood for a list of video frames."""

    model_name: str = CONFIG.models.visual_deepfake_model
    device: str = CONFIG.resolve_device()

    def predict(self, frames: Sequence[Path]) -> float:
        """Return the mean probability of the "fake" class across frames."""
        if not frames:
            raise ValueError("No frames provided for deepfake detection")
        results = run_pipeline(
            "image-classification",
            self.model_name,
            inputs=[str(frame) for frame in frames],
            device=self.device,
        )
        fake_scores: List[float] = []
        for output in results:
            if isinstance(output, list):
                scores = {item["label"].lower(): item["score"] for item in output}
            else:  # pragma: no cover - huggingface may return dict
                scores = {output["label"].lower(): output["score"]}
            fake_score = max(
                (score for label, score in scores.items() if "fake" in label),
                default=0.0,
            )
            fake_scores.append(fake_score)
        return float(np.mean(fake_scores))


__all__ = ["DeepfakeDetector"]
