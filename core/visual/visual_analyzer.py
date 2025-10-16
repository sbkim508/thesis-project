"""Aggregate visual modality detectors to produce feature vectors."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

from .deepfake_detector import DeepfakeDetector
from .emotion_recognizer import VisualEmotionRecognizer
from .nsfw_detector import NSFWDetector


@dataclass
class VisualAnalyzer:
    """Run all visual detectors and produce structured output."""

    deepfake_detector: DeepfakeDetector = field(default_factory=DeepfakeDetector)
    nsfw_detector: NSFWDetector = field(default_factory=NSFWDetector)
    emotion_recognizer: VisualEmotionRecognizer = field(default_factory=VisualEmotionRecognizer)

    def analyze(self, frames: Sequence[Path]) -> Dict[str, Dict[str, float]]:
        """Return a dictionary summarizing visual signals."""
        deepfake_score = self.deepfake_detector.predict(frames)
        nsfw_score = self.nsfw_detector.predict(frames)
        emotion_scores = self.emotion_recognizer.predict(frames)
        return {
            "deceptiveness": {"visual_deepfake_score": deepfake_score},
            "harmfulness": {
                "visual_nsfw_score": nsfw_score,
                **{f"visual_emotion_{label}": score for label, score in emotion_scores.items()},
            },
        }


__all__ = ["VisualAnalyzer"]
