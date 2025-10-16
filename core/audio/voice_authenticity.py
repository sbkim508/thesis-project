"""Audio-based authenticity estimation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import librosa
import numpy as np

from config import CONFIG
from utils.hf_pipeline import run_pipeline


@dataclass
class VoiceAuthenticityDetector:
    """Estimate the probability that speech is manipulated."""

    model_name: str = CONFIG.models.audio_authenticity_model
    device: str = CONFIG.resolve_device()
    sample_rate: int = 16000

    def _load_audio(self, path: Path) -> np.ndarray:
        audio, _ = librosa.load(path, sr=self.sample_rate)
        return audio

    def predict(self, audio_segments: Sequence[Path]) -> Dict[str, float]:
        if not audio_segments:
            raise ValueError("No audio segments provided")
        predictions = []
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
                predictions.append({item["label"].lower(): float(item["score"]) for item in outputs})
            else:  # pragma: no cover
                predictions.append({outputs["label"].lower(): float(outputs["score"])})
        spoof_scores = [max(pred.get("spoof", 0.0), pred.get("fake", 0.0)) for pred in predictions]
        return {
            "audio_spoof_score_mean": float(np.mean(spoof_scores)),
            "audio_spoof_score_max": float(np.max(spoof_scores)),
        }


__all__ = ["VoiceAuthenticityDetector"]
