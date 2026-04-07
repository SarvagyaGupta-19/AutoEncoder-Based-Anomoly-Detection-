"""
Preprocessing module — SINGLE SOURCE OF TRUTH for all audio → spectrogram conversion.

This module is used by:
  - Batch conversion (raw audio → saved spectrogram images)
  - Training data loader
  - Evaluation / scoring
  - Flask inference app

NEVER duplicate this logic elsewhere. If you need to preprocess audio,
import from here.
"""

import os
import numpy as np
import librosa
import cv2
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ────────────────────────────────────────────────────────────
# CORE: Audio → Log-Mel Spectrogram (NumPy array)
# ────────────────────────────────────────────────────────────

def audio_to_log_mel(audio_path):
    """
    Load an audio file and convert to a log-mel spectrogram (2D float array).

    Steps:
        1. Load audio at fixed sample rate
        2. Pad or truncate to fixed duration
        3. Compute mel spectrogram
        4. Convert to log (dB) scale
        5. Normalize to [0, 1]

    Args:
        audio_path: Path to .wav (or any librosa-supported format)

    Returns:
        np.ndarray of shape (n_mels, time_frames), values in [0, 1], dtype float32
    """
    # Step 1: Load
    y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)

    # Step 2: Pad or truncate to fixed length
    target_length = config.SAMPLE_RATE * config.AUDIO_DURATION
    if len(y) < target_length:
        # Pad with zeros (silence) at the end
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        # Truncate
        y = y[:target_length]

    # Step 3: Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLE_RATE,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        fmin=config.F_MIN,
        fmax=config.F_MAX,
        power=2.0,
    )

    # Step 4: Log scale (power → dB)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    # Now values are in range [~-80, 0] dB

    # Step 5: Normalize to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    log_mel = log_mel.astype(np.float32)

    return log_mel


# ────────────────────────────────────────────────────────────
# Spectrogram → Model-Ready Tensor
# ────────────────────────────────────────────────────────────

def spectrogram_to_model_input(spec_2d):
    """
    Resize a 2D spectrogram array to model input dimensions and add channel axis.

    Args:
        spec_2d: np.ndarray of shape (n_mels, time_frames), values in [0, 1]

    Returns:
        np.ndarray of shape (IMG_HEIGHT, IMG_WIDTH, 1), dtype float32
    """
    resized = cv2.resize(spec_2d, (config.IMG_WIDTH, config.IMG_HEIGHT),
                         interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32)
    # Ensure values still in [0, 1] after resize
    resized = np.clip(resized, 0.0, 1.0)
    # Add channel dimension
    return np.expand_dims(resized, axis=-1)


def audio_to_model_input(audio_path):
    """
    End-to-end: audio file → model-ready tensor.

    Args:
        audio_path: Path to audio file

    Returns:
        np.ndarray of shape (IMG_HEIGHT, IMG_WIDTH, 1)
    """
    log_mel = audio_to_log_mel(audio_path)
    return spectrogram_to_model_input(log_mel)


# ────────────────────────────────────────────────────────────
# Save spectrogram as image (for visualization / caching)
# ────────────────────────────────────────────────────────────

def save_spectrogram_image(spec_2d, save_path):
    """
    Save a [0,1] spectrogram as a grayscale PNG image.

    Args:
        spec_2d: np.ndarray of shape (n_mels, time_frames), values in [0, 1]
        save_path: Output .png path
    """
    # Scale to 0-255 for image format
    img = (spec_2d * 255).astype(np.uint8)
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT),
                     interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(save_path, img)


def load_spectrogram_image(image_path):
    """
    Load a saved spectrogram image and prepare for model input.
    Used when spectrograms are pre-cached as .png files.

    Args:
        image_path: Path to .png spectrogram

    Returns:
        np.ndarray of shape (IMG_HEIGHT, IMG_WIDTH, 1)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (config.IMG_WIDTH, config.IMG_HEIGHT),
                     interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=-1)


# ────────────────────────────────────────────────────────────
# Save / Load spectrograms as lossless .npy arrays
# ────────────────────────────────────────────────────────────

def save_spectrogram_npy(spec_2d, save_path):
    """
    Save a [0,1] spectrogram as a lossless float32 .npy file.
    Resizes to model input dimensions before saving.

    Args:
        spec_2d: np.ndarray of shape (n_mels, time_frames), values in [0, 1]
        save_path: Output .npy path
    """
    resized = cv2.resize(spec_2d, (config.IMG_WIDTH, config.IMG_HEIGHT),
                         interpolation=cv2.INTER_LINEAR)
    resized = np.clip(resized, 0.0, 1.0).astype(np.float32)
    np.save(save_path, resized)


def load_spectrogram_npy(npy_path):
    """
    Load a saved .npy spectrogram and prepare for model input.

    Args:
        npy_path: Path to .npy spectrogram file

    Returns:
        np.ndarray of shape (IMG_HEIGHT, IMG_WIDTH, 1), dtype float32
    """
    spec = np.load(npy_path).astype(np.float32)
    if spec.shape != (config.IMG_HEIGHT, config.IMG_WIDTH):
        spec = cv2.resize(spec, (config.IMG_WIDTH, config.IMG_HEIGHT),
                          interpolation=cv2.INTER_LINEAR)
    spec = np.clip(spec, 0.0, 1.0)
    return np.expand_dims(spec, axis=-1)


# ────────────────────────────────────────────────────────────
# Batch conversion: Directory of .wav → Directory of .npy
# ────────────────────────────────────────────────────────────

def convert_directory_npy(audio_dir, output_dir, skip_existing=True):
    """
    Convert all audio files in a directory to lossless .npy spectrograms.

    Args:
        audio_dir: Directory containing .wav files
        output_dir: Directory to save .npy spectrograms
        skip_existing: If True, skip files that already have a corresponding .npy

    Returns:
        int: Number of files converted
    """
    os.makedirs(output_dir, exist_ok=True)

    audio_files = [f for f in sorted(os.listdir(audio_dir))
                   if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    if not audio_files:
        print(f"  ⚠ No audio files found in {audio_dir}")
        return 0

    converted = 0
    for fname in tqdm(audio_files, desc=f"Converting {os.path.basename(audio_dir)}"):
        stem = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, f"{stem}.npy")

        if skip_existing and os.path.exists(out_path):
            continue

        try:
            log_mel = audio_to_log_mel(os.path.join(audio_dir, fname))
            save_spectrogram_npy(log_mel, out_path)
            converted += 1
        except Exception as e:
            print(f"  ✗ Failed on {fname}: {e}")

    return converted


def convert_all_datasets_npy():
    """
    Convert train, source_test, and target_test audio directories
    to lossless .npy spectrograms. Uses paths from config.py.
    """
    print("=" * 60)
    print("  AUDIO → LOG-MEL SPECTROGRAM (.npy) CONVERSION")
    print("=" * 60)

    pairs = [
        (config.RAW_TRAIN_DIR, config.NPY_TRAIN_DIR),
        (config.RAW_SOURCE_TEST_DIR, config.NPY_SOURCE_TEST_DIR),
        (config.RAW_TARGET_TEST_DIR, config.NPY_TARGET_TEST_DIR),
    ]

    total = 0
    for audio_dir, npy_dir in pairs:
        n = convert_directory_npy(audio_dir, npy_dir)
        total += n
        print(f"  ✓ {os.path.basename(audio_dir)}: {n} new .npy spectrograms")

    print(f"\n  Total converted: {total}")
    print("=" * 60)
    return total


# ────────────────────────────────────────────────────────────
# Load a full .npy dataset as numpy array (for evaluation)
# ────────────────────────────────────────────────────────────

def load_npy_dataset(npy_dir):
    """
    Load all .npy spectrogram files from a directory into a single numpy array.

    Args:
        npy_dir: Directory containing .npy spectrogram files

    Returns:
        np.ndarray of shape (N, IMG_HEIGHT, IMG_WIDTH, 1), dtype float32
    """
    files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")])
    if not files:
        raise FileNotFoundError(f"No .npy files found in {npy_dir}")

    images = []
    for fname in tqdm(files, desc=f"Loading {os.path.basename(npy_dir)}"):
        img = load_spectrogram_npy(os.path.join(npy_dir, fname))
        images.append(img)

    dataset = np.array(images, dtype=np.float32)
    print(f"  Loaded {len(dataset)} spectrograms, shape: {dataset.shape}")
    return dataset


# ────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this script directly to convert all datasets:
        python -m src.preprocessing           # convert to .npy (default)
        python -m src.preprocessing --png      # convert to .png (for visualization)
    """
    import argparse
    parser = argparse.ArgumentParser(description="Convert audio to spectrograms")
    parser.add_argument("--png", action="store_true",
                        help="Convert to PNG images instead of .npy arrays")
    args = parser.parse_args()

    if args.png:
        convert_all_datasets()
    else:
        convert_all_datasets_npy()

