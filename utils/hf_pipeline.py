"""Utilities for creating Hugging Face pipelines with consistent options."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from transformers import pipeline

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_pipeline(task: str, model: str, device: int | str = "cpu", **kwargs: Any):
    """Return a cached Hugging Face pipeline for the given task and model."""
    LOGGER.info("Loading pipeline for task=%s model=%s", task, model)
    return pipeline(task=task, model=model, device=device, **kwargs)


def run_pipeline(task: str, model: str, inputs: Any, device: int | str = "cpu", **kwargs: Any):
    """Execute a pipeline in a single call with caching."""
    pipe = get_pipeline(task, model, device, **kwargs)
    return pipe(inputs)


__all__ = ["get_pipeline", "run_pipeline"]
