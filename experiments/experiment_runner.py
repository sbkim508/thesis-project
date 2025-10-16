"""Utilities to evaluate multiple models on extracted features."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from config import CONFIG


@dataclass
class ExperimentRunner:
    """Run cross-validated experiments for a collection of models."""

    metrics: tuple[str, ...] = field(
        default_factory=lambda: tuple(CONFIG.experiments.classification_metrics)
    )
    cv_folds: int = CONFIG.experiments.cross_validation_folds
    random_state: int = CONFIG.experiments.random_state

    def evaluate(self, models: Dict[str, object], features: pd.DataFrame, label_column: str = "Label") -> pd.DataFrame:
        if label_column not in features.columns:
            raise KeyError(f"Label column '{label_column}' not found in features DataFrame")
        X = features.drop(columns=[label_column]).to_numpy()
        y = features[label_column].to_numpy()
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        records = []
        for name, model in models.items():
            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=self.metrics,
                n_jobs=-1,
                return_train_score=False,
            )
            summary = {metric: float(np.mean(scores[f"test_{metric}"])) for metric in self.metrics}
            summary["model"] = name
            records.append(summary)
        return pd.DataFrame(records)


__all__ = ["ExperimentRunner"]
