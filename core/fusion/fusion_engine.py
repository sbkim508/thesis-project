"""Fuse modality outputs into a single feature dictionary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import pandas as pd


@dataclass
class FusionEngine:
    """Combine modality-specific dictionaries into tabular features."""

    def fuse(self, *modalities: Mapping[str, Mapping[str, float]]) -> pd.Series:
        feature_dict: Dict[str, float] = {}
        for modality in modalities:
            for category, metrics in modality.items():
                for name, value in metrics.items():
                    feature_dict[f"{category}__{name}"] = float(value)
        return pd.Series(feature_dict)


__all__ = ["FusionEngine"]
