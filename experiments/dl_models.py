"""Deep learning style models using neural networks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sklearn.neural_network import MLPClassifier


@dataclass
class DLModelFactory:
    """Return neural-network-based estimators for tabular features."""

    random_state: int = 42

    def create(self) -> Dict[str, MLPClassifier]:
        return {
            "mlp_classifier": MLPClassifier(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=64,
                max_iter=300,
                random_state=self.random_state,
            )
        }


__all__ = ["DLModelFactory"]
