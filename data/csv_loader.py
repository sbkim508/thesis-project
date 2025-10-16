"""Utilities for loading and validating metadata CSV files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = {
    "Filename",
    "Label",
    "Date",
    "Video Ground Truth",
    "Audio Ground Truth",
}


@dataclass
class CSVLoader:
    """Load CSV metadata and perform basic validation."""

    path: Path
    validate_columns: bool = True

    def load(self) -> pd.DataFrame:
        """Read the CSV into a DataFrame and validate columns if requested."""
        df = pd.read_csv(self.path)
        if self.validate_columns:
            missing = REQUIRED_COLUMNS.difference(df.columns)
            if missing:
                raise ValueError(
                    f"Missing required columns in {self.path}: {', '.join(sorted(missing))}"
                )
        df["Filename"] = df["Filename"].astype(str)
        df["Label"] = df["Label"].astype(str)
        return df

    def subset(self, df: pd.DataFrame, *, split: Optional[str] = None) -> pd.DataFrame:
        """Return a subset of the dataframe filtered by split if provided."""
        if split is None:
            return df
        if "Finetuning Set" not in df.columns:
            raise KeyError("Finetuning Set column not present in metadata")
        return df[df["Finetuning Set"].str.lower() == split.lower()].reset_index(drop=True)


__all__ = ["CSVLoader", "REQUIRED_COLUMNS"]
