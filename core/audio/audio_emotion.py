"""Audio emotion recognition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import librosa
import numpy as np

from config import CONFIG
from utils.hf_pipeline import run_pipeline


@dataclass
class AudioEmotionAnalyzer:
    """Return aggregated emotion distribution for speech segments."""

    model_name: str = CONFIG.models.audio_emotion_model
    device: str = CONFIG.resolve_device()
    sample_rate: int = 16000

    def _load_audio(self, path: Path) -> np.ndarray:
        audio, _ = librosa.load(path, sr=self.sample_rate)
        return audio

    def predict(self, audio_segments: Sequence[Path]) -> Dict[str, float]:
        if not audio_segments:
            raise ValueError("No audio provided for emotion analysis")
        aggregated: Dict[str, List[float]] = {}
        for segment in audio_segments:
            audio = self._load_audio(segment)
            outputs = run_pipeline(
                "audio-classification",
                self.model_name,
                inputs=audio,
                device=self.device,
                sampling_rate=self.sample_rate,
            )
            if isinstance(outputs, list):
                entries = outputs
            else:  # pragma: no cover
                entries = [outputs]
            for entry in entries:
                label = entry["label"].lower()
                aggregated.setdefault(label, []).append(float(entry["score"]))
        return {label: float(np.mean(scores)) for label, scores in aggregated.items()}


__all__ = ["AudioEmotionAnalyzer"]
