"""Aggregate audio modality detectors."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence

from .audio_emotion import AudioEmotionAnalyzer
from .voice_authenticity import VoiceAuthenticityDetector


@dataclass
class AudioAnalyzer:
    """Run audio detectors and return structured features."""

    authenticity_detector: VoiceAuthenticityDetector = field(default_factory=VoiceAuthenticityDetector)
    emotion_analyzer: AudioEmotionAnalyzer = field(default_factory=AudioEmotionAnalyzer)

    def analyze(self, audio_segments: Sequence[Path]) -> Dict[str, Dict[str, float]]:
        authenticity_scores = self.authenticity_detector.predict(audio_segments)
        emotion_scores = self.emotion_analyzer.predict(audio_segments)
        return {
            "deceptiveness": authenticity_scores,
            "harmfulness": {
                **{f"audio_emotion_{label}": score for label, score in emotion_scores.items()},
            },
        }


__all__ = ["AudioAnalyzer"]
