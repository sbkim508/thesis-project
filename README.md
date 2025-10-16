# Thesis Project

Multi-modal framework for detecting deceptive and harmful deepfake content.

## Structure

- `config.py`: Centralized configuration for paths, models, and experiment defaults.
- `core/`: Video decomposition and modality-specific analyzers.
  - `video_decomposer.py`: Extracts frames and audio segments from raw videos.
  - `visual/`: Deepfake, NSFW, and emotion detectors for visual frames.
  - `audio/`: Voice authenticity and emotion analyzers for speech tracks.
  - `text/`: Misinformation, toxicity, and emotion detectors for transcripts.
  - `fusion/`: Combines modality outputs into tabular features.
- `data/csv_loader.py`: Metadata ingestion and validation.
- `models/model_hub.py`: Helper to explore Hugging Face models.
- `experiments/`: Training utilities for ML, DL, and Transformer baselines.
  - `experiment_runner.py`: Cross-validated evaluation with configurable metrics.
  - `statistical_tests.py`: Significance testing helpers for model comparisons.
- `requirements.txt`: Python dependencies.

## Usage

1. Install dependencies: `pip install -r requirements.txt`.
2. Place videos under `data/deepfakes/video` and metadata CSV files in `data/metadata`.
3. Use `DetectionEngine` to extract modality features into a CSV-ready DataFrame.
4. Run `ExperimentRunner` with `MLModelFactory` / `DLModelFactory` to benchmark classifiers.
5. Fine-tune transcript-focused models via `TransformerExperiment` when raw transcripts are available.

