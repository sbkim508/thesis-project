"""Main orchestration engine for multi-modal analysis."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from config import CONFIG
from data.csv_loader import CSVLoader
from .video_decomposer import VideoDecomposer
from .visual.visual_analyzer import VisualAnalyzer
from .audio.audio_analyzer import AudioAnalyzer
from .text.text_analyzer import TextAnalyzer
from utils.hf_pipeline import run_pipeline

LOGGER = logging.getLogger(__name__)


@dataclass
class DetectionEngine:
    """Run the full pipeline on videos referenced in a metadata CSV."""

    output_dir: Path
    visual_analyzer: VisualAnalyzer = field(default_factory=VisualAnalyzer)
    audio_analyzer: AudioAnalyzer = field(default_factory=AudioAnalyzer)
    text_analyzer: TextAnalyzer = field(default_factory=TextAnalyzer)
    asr_model: str = CONFIG.models.audio_asr_model
    device: str = CONFIG.resolve_device()

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _transcribe(self, audio_segments: Sequence[Path]) -> List[str]:
        transcripts: List[str] = []
        for segment in audio_segments:
            result = run_pipeline(
                "automatic-speech-recognition",
                self.asr_model,
                inputs=str(segment),
                device=self.device,
            )
            if isinstance(result, dict):
                transcripts.append(result.get("text", ""))
            else:  # pragma: no cover
                transcripts.append(str(result))
        return transcripts

    def process_video(self, video_path: Path, *, transcripts: Optional[Sequence[str]] = None) -> pd.Series:
        LOGGER.info("Processing video %s", video_path)
        frame_dir = self.output_dir / video_path.stem / "frames"
        audio_dir = self.output_dir / video_path.stem / "audio"
        frame_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        decomposer = VideoDecomposer(frame_dir)
        frames = decomposer.extract_frames(video_path)
        audio_path = decomposer.extract_audio(video_path)
        audio_segments = VideoDecomposer(audio_dir).chunk_audio(audio_path)
        text_segments = list(transcripts) if transcripts is not None else self._transcribe(audio_segments)

        visual_features = self.visual_analyzer.analyze(frames)
        audio_features = self.audio_analyzer.analyze(audio_segments)
        text_features = self.text_analyzer.analyze(text_segments)

        fusion_features = {"filename": video_path.name}
        for category, metrics in visual_features.items():
            for key, value in metrics.items():
                fusion_features[f"{category}__{key}"] = value
        for category, metrics in audio_features.items():
            for key, value in metrics.items():
                fusion_features[f"{category}__{key}"] = value
        for category, metrics in text_features.items():
            for key, value in metrics.items():
                fusion_features[f"{category}__{key}"] = value
        return pd.Series(fusion_features)

    def run_from_csv(self, csv_path: Path, video_root: Path) -> pd.DataFrame:
        loader = CSVLoader(csv_path)
        metadata = loader.load()
        features: List[pd.Series] = []
        for _, row in metadata.iterrows():
            video_path = video_root / row["Filename"]
            transcripts = row.get("Transcript")
            transcript_segments: Optional[Sequence[str]] = None
            if isinstance(transcripts, str) and transcripts.strip():
                transcript_segments = [transcripts]
            features.append(self.process_video(video_path, transcripts=transcript_segments))
        df = pd.DataFrame(features)
        if "Label" in metadata.columns:
            df["Label"] = metadata["Label"]
        return df


__all__ = ["DetectionEngine"]
