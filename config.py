"""Global configuration for thesis project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Paths:
    """Convenience container for frequently used project paths."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    data_dir: Path = field(init=False)
    video_dir: Path = field(init=False)
    metadata_dir: Path = field(init=False)
    experiments_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.root / "data"
        self.video_dir = self.data_dir / "deepfakes" / "video"
        self.metadata_dir = self.data_dir / "metadata"
        self.experiments_dir = self.root / "experiments"
        self.data_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        self.experiments_dir.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """Configuration for the pretrained models used across modalities."""

    visual_deepfake_model: str = "Bingsu/DeepFake-Detection-CLIP-ViT-B-32"
    visual_nsfw_model: str = "Falconsai/nsfw_image_detection"
    visual_emotion_model: str = "trpakov/vit-face-expression"

    audio_authenticity_model: str = "microsoft/wavlm-base-plus-sv"
    audio_emotion_model: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    audio_asr_model: str = "openai/whisper-small"

    text_misinformation_model: str = "facebook/bart-large-mnli"
    text_toxicity_model: str = "unitary/toxic-bert"
    text_emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"


@dataclass
class ExperimentConfig:
    """Configuration for evaluation metrics and experiment options."""

    classification_metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "f1", "precision", "recall", "roc_auc"]
    )
    cross_validation_folds: int = 5
    random_state: int = 42
    enable_hyperparameter_search: bool = True


@dataclass
class Config:
    """Top level configuration object used throughout the repository."""

    paths: Paths = field(default_factory=Paths)
    models: ModelConfig = field(default_factory=ModelConfig)
    experiments: ExperimentConfig = field(default_factory=ExperimentConfig)
    device: str = "cuda"
    fallback_device: str = "cpu"
    cache_dir: Optional[Path] = None

    def resolve_device(self) -> str:
        """Return an available device string."""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return self.fallback_device


CONFIG = Config()

__all__ = ["Config", "CONFIG", "Paths", "ModelConfig", "ExperimentConfig"]
