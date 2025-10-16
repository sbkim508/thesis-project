"""NSFW detector mapping to harmfulness signals."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from config import CONFIG
from utils.hf_pipeline import run_pipeline


@dataclass
class NSFWDetector:
    """Estimate NSFW probability from frames."""

    model_name: str = CONFIG.models.visual_nsfw_model
    device: str = CONFIG.resolve_device()

    def predict(self, frames: Sequence[Path]) -> float:
        if not frames:
            raise ValueError("No frames provided for NSFW detection")
        results = run_pipeline(
            "image-classification",
            self.model_name,
            inputs=[str(frame) for frame in frames],
            device=self.device,
        )
        nsfw_scores = []
        for output in results:
            if isinstance(output, list):
                scores = {item["label"].lower(): item["score"] for item in output}
            else:  # pragma: no cover
                scores = {output["label"].lower(): output["score"]}
            nsfw_scores.append(max(scores.get("nsfw", 0.0), scores.get("porn", 0.0)))
        return float(np.mean(nsfw_scores))


__all__ = ["NSFWDetector"]
