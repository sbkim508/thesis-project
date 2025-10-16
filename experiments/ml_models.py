"""Classical machine learning models for baseline comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class MLModelFactory:
    """Factory that returns a dictionary of ML models."""

    random_state: int = 42

    def create(self) -> Dict[str, Pipeline]:
        return {
            "logistic_regression": Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1000,
                            random_state=self.random_state,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                class_weight="balanced",
            ),
        }


__all__ = ["MLModelFactory"]
