"""Aggregate textual detectors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

from .emotion_analyzer import TextEmotionAnalyzer
from .misinformation_detector import MisinformationDetector
from .toxicity_detector import ToxicityDetector


@dataclass
class TextAnalyzer:
    """Run textual detectors on transcripts."""

    misinformation_detector: MisinformationDetector = field(default_factory=MisinformationDetector)
    toxicity_detector: ToxicityDetector = field(default_factory=ToxicityDetector)
    emotion_analyzer: TextEmotionAnalyzer = field(default_factory=TextEmotionAnalyzer)

    def analyze(self, transcripts: Sequence[str]) -> Dict[str, Dict[str, float]]:
        misinformation_score = self.misinformation_detector.predict(transcripts)
        toxicity_score = self.toxicity_detector.predict(transcripts)
        emotion_scores = self.emotion_analyzer.predict(transcripts)
        return {
            "deceptiveness": {"text_misinformation_score": misinformation_score},
            "harmfulness": {
                "text_toxicity_score": toxicity_score,
                **{f"text_emotion_{label}": score for label, score in emotion_scores.items()},
            },
        }


__all__ = ["TextAnalyzer"]
