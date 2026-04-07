"""
Centralized configuration for the Anomalous Sound Detection system.
All paths, hyperparameters, and spectrogram settings live here.
Every module imports from this file — single source of truth.
"""

import os

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Raw audio (place MIMII .wav files here)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_AUDIO_DIR = os.path.join(DATA_DIR, "raw_audio")
RAW_TRAIN_DIR = os.path.join(RAW_AUDIO_DIR, "train")
RAW_SOURCE_TEST_DIR = os.path.join(RAW_AUDIO_DIR, "source_test")
RAW_TARGET_TEST_DIR = os.path.join(RAW_AUDIO_DIR, "target_test")

# Generated spectrograms (PNG — used for visualization / Flask app only)
SPECTROGRAM_DIR = os.path.join(DATA_DIR, "spectrograms")
SPEC_TRAIN_DIR = os.path.join(SPECTROGRAM_DIR, "train")
SPEC_SOURCE_TEST_DIR = os.path.join(SPECTROGRAM_DIR, "source_test")
SPEC_TARGET_TEST_DIR = os.path.join(SPECTROGRAM_DIR, "target_test")

# Lossless spectrogram arrays (.npy — used for training / evaluation)
NPY_DIR = os.path.join(DATA_DIR, "npy")
NPY_TRAIN_DIR = os.path.join(NPY_DIR, "train")
NPY_SOURCE_TEST_DIR = os.path.join(NPY_DIR, "source_test")
NPY_TARGET_TEST_DIR = os.path.join(NPY_DIR, "target_test")

# Model artifacts
MODEL_DIR = os.path.join(BASE_DIR, "models")
AUTOENCODER_PATH = os.path.join(MODEL_DIR, "autoencoder.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.keras")
PCA_PATH = os.path.join(MODEL_DIR, "pca_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
ANOMALY_STATS_PATH = os.path.join(MODEL_DIR, "anomaly_stats.joblib")

# ──────────────────────────────────────────────
# AUDIO / SPECTROGRAM PARAMETERS
# ──────────────────────────────────────────────
SAMPLE_RATE = 16000          # Hz — MIMII dataset standard
AUDIO_DURATION = 10          # seconds — clip/pad to this length
N_MELS = 128                 # mel frequency bins
N_FFT = 2048                 # FFT window size
HOP_LENGTH = 512             # hop between FFT windows
F_MIN = 50                   # minimum frequency (Hz) — cuts low rumble
F_MAX = 7500                 # maximum frequency (Hz) — most machine sounds

# ──────────────────────────────────────────────
# MODEL INPUT
# ──────────────────────────────────────────────
IMG_HEIGHT = 128             # spectrogram resize height
IMG_WIDTH = 128              # spectrogram resize width  (128×128 is faster than 224×224
                             # and loses minimal info for mel spectrograms)
IMG_CHANNELS = 1             # grayscale

# ──────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ──────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15      # 15% of training data held out
EARLY_STOPPING_PATIENCE = 15 # stop if val_loss doesn't improve for 15 epochs
LR_REDUCE_PATIENCE = 7       # reduce LR if val_loss stalls for 7 epochs
LR_REDUCE_FACTOR = 0.5       # halve the LR when reducing
MIN_LR = 1e-6                # floor for LR reduction

# ──────────────────────────────────────────────
# ANOMALY DETECTION
# ──────────────────────────────────────────────
PCA_COMPONENTS = 64          # dimensions to keep after PCA
ANOMALY_PERCENTILE_WARNING = 90   # scores above this → "NEEDS MAINTENANCE"
ANOMALY_PERCENTILE_CRITICAL = 99  # scores above this → "ANOMALY"

# ──────────────────────────────────────────────
# DATA AUGMENTATION (Denoising Autoencoder)
# ──────────────────────────────────────────────
AUGMENT_ENABLED = True           # enable augmentation during training
AUGMENT_NOISE_STD = 0.02         # Gaussian noise standard deviation
AUGMENT_TIME_SHIFT_MAX = 0.1     # max fraction of time axis to roll
AUGMENT_FREQ_MASK_WIDTH = 8      # max mel bins to zero out (SpecAugment)
AUGMENT_TIME_MASK_WIDTH = 8      # max time frames to zero out (SpecAugment)

# ──────────────────────────────────────────────
# FLASK APP
# ──────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac", "ogg"}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max upload

# ──────────────────────────────────────────────
# Ensure critical directories exist at import time
# ──────────────────────────────────────────────
for _dir in [RAW_TRAIN_DIR, RAW_SOURCE_TEST_DIR, RAW_TARGET_TEST_DIR,
             SPEC_TRAIN_DIR, SPEC_SOURCE_TEST_DIR, SPEC_TARGET_TEST_DIR,
             NPY_TRAIN_DIR, NPY_SOURCE_TEST_DIR, NPY_TARGET_TEST_DIR,
             MODEL_DIR, UPLOAD_FOLDER]:
    os.makedirs(_dir, exist_ok=True)

