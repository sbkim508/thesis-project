"""Transformer-based text classification utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


@dataclass
class TransformerExperiment:
    """Fine-tune transformer models on transcript data."""

    model_name: str
    text_column: str = "Transcript"
    label_column: str = "Label"
    num_labels: int = 2
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 8
    weight_decay: float = 0.01
    output_dir: str = "transformer_outputs"
    seed: int = 42

    def _tokenize(self, examples: Dict[str, list[str]], tokenizer: AutoTokenizer) -> Dict[str, list[int]]:
        return tokenizer(examples[self.text_column], truncation=True, padding="max_length")

    def _label_encoder(self, labels: pd.Series) -> Dict[str, int]:
        unique = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
        return unique

    def fine_tune(self, df: pd.DataFrame) -> Trainer:
        if self.text_column not in df.columns:
            raise KeyError(f"DataFrame must contain {self.text_column} column")
        if self.label_column not in df.columns:
            raise KeyError(f"DataFrame must contain {self.label_column} column")
        label_mapping = self._label_encoder(df[self.label_column])
        df = df.copy()
        df[self.label_column] = df[self.label_column].map(label_mapping)
        dataset = Dataset.from_pandas(df[[self.text_column, self.label_column]])
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dataset = dataset.map(lambda examples: self._tokenize(examples, tokenizer), batched=True)
        dataset = dataset.rename_column(self.label_column, "labels")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            evaluation_strategy="no",
            save_strategy="epoch",
            seed=self.seed,
            logging_steps=50,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        trainer.train()
        return trainer


__all__ = ["TransformerExperiment"]
