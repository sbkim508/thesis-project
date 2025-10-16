"""Utilities for interacting with the Hugging Face Hub."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from huggingface_hub import HfApi, ModelFilter


@dataclass
class ModelHubExplorer:
    """Search for models matching modality and task requirements."""

    api: HfApi = HfApi()

    def search(self, task: str, query: str, limit: int = 10) -> List[str]:
        models = self.api.list_models(filter=ModelFilter(task=task), search=query, limit=limit)
        return [model.modelId for model in models]


__all__ = ["ModelHubExplorer"]
