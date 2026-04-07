# 🔊 CNN-Based Spectrogram Model for Machine Sound Analysis

Anomalous sound detection system using a **CNN Autoencoder** trained on **log-mel spectrograms** of normal machine operating sounds. Detects anomalies by measuring how poorly the autoencoder reconstructs unseen sounds, combined with **Mahalanobis distance** scoring in the latent feature space.

## Architecture

```
Audio (.wav)
    │
    ▼
Log-Mel Spectrogram (128 mel bins, 16kHz, 10s clips)
    │
    ▼
CNN Autoencoder (trained on NORMAL sounds only)
    │
    ├── Reconstruction Error (MSE between input and output)
    │
    └── Encoder → Latent Vector (256-dim)
                    │
                    ├── PCA Reduction (64-dim)
                    │
                    └── Mahalanobis Distance from "normal" distribution
    │
    ▼
Combined Score → NORMAL / NEEDS MAINTENANCE / ANOMALY
```

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
uv venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Prepare Data

Download the [MIMII Dataset](https://zenodo.org/record/3384388) and place audio files:

```
data/
├── raw_audio/
│   ├── train/          ← normal .wav files (for training)
│   ├── source_test/    ← test .wav files (source domain)
│   └── target_test/    ← test .wav files (target domain)
```

### 3. Convert Audio to Spectrograms

```bash
python -m src.preprocessing
```

This converts all `.wav` files to log-mel spectrogram `.png` images in `data/spectrograms/`.

### 4. Train the Model

```bash
python -m src.train
# Or with custom settings:
python -m src.train --epochs 100 --batch-size 16
```

Training features:
- Early stopping (patience=15)
- Learning rate reduction on plateau
- Best model checkpointing
- Saves both autoencoder and encoder separately

### 5. Fit Anomaly Detector

```bash
python -m src.evaluate --fit
```

This extracts features from training data, fits PCA, and computes adaptive thresholds.

### 6. Evaluate

```bash
# Evaluate test sets
python -m src.evaluate --test

# Score a single audio file
python -m src.evaluate --score path/to/audio.wav
```

### 7. Web Interface

```bash
python -m app.app
# Open http://localhost:5000
```

Upload a `.wav` file and get instant classification with spectrogram visualization.

## Project Structure

```
Anomalous-Sound-Detection/
├── config.py                 # All paths, hyperparameters, settings
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── preprocessing.py      # Audio → Log-Mel spectrogram (single source of truth)
│   ├── model.py              # CNN Autoencoder architecture
│   ├── train.py              # Training pipeline with callbacks
│   ├── evaluate.py           # Anomaly scoring + batch evaluation
│   └── utils.py              # Visualization utilities
│
├── app/
│   ├── app.py                # Flask web application
│   └── templates/
│       └── index.html        # Upload + results UI
│
├── data/
│   ├── raw_audio/            # Place .wav files here
│   └── spectrograms/         # Generated .png spectrograms
│
└── models/                   # Saved model artifacts
    ├── autoencoder.keras
    ├── encoder.keras
    ├── pca_model.joblib
    └── anomaly_stats.joblib
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Autoencoder** (not classifier) | Only needs normal data for training; detects any anomaly, even unseen failure types |
| **Log-Mel Spectrogram** | Matches human auditory perception; standard in DCASE benchmarks |
| **128×128 input** | Mel spectrograms are 128 bins tall; no information wasted |
| **Dual scoring** (recon error + Mahalanobis) | More robust than either method alone |
| **Adaptive thresholds** | Derived from training data distribution, not hardcoded magic numbers |
| **Single preprocessing module** | Train and inference use identical conversion — no train/serve skew |

## Model Details

- **Encoder**: 4 Conv2D blocks (32→64→128→256 filters) with BatchNorm + MaxPool → Dense(512) → Dense(256)
- **Decoder**: Dense → Reshape → 4 Conv2DTranspose blocks → sigmoid output
- **Loss**: MSE (reconstruction error)
- **Optimizer**: Adam with ReduceLROnPlateau

## API Reference

### REST API

```bash
# Score audio via API
curl -X POST http://localhost:5000/api/predict \
  -F "audio_file=@machine_sound.wav"
```

Response:
```json
{
    "classification": "NORMAL",
    "mahalanobis_score": 5.234,
    "reconstruction_error": 0.001234,
    "file": "machine_sound.wav"
}
```

## Requirements

- Python 3.10+
- TensorFlow 2.14+
- See `requirements.txt` for full list
